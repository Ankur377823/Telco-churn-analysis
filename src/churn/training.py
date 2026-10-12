"""Training pipeline: compare the model zoo, tune, build ensembles, select, calibrate, score."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

from . import config as C
from .data_cleaning import load_clean, split_xy
from .metrics import CV_SCORING, prob_metrics
from .preprocessing import make_pipeline
from .model_zoo import COMPLEXITY, SEARCH_SPACES, base_models, ensembles


def cv_splitter():
    return StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def train_test(df):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


def cv_evaluate(pipes: dict, X, y, verbose=True) -> pd.DataFrame:
    """5-fold stratified CV of each pipeline on the training set."""
    rows = []
    for name, pipe in pipes.items():
        t0 = time.time()
        res = cross_validate(pipe, X, y, cv=cv_splitter(), scoring=CV_SCORING, n_jobs=-1)
        row = {"model": name, "fit_seconds": time.time() - t0}
        for k in CV_SCORING:
            s = res[f"test_{k}"]
            s = -s if k in ("brier", "log_loss") else s
            row[k], row[f"{k}_sd"] = s.mean(), s.std(ddof=1)
        rows.append(row)
        if verbose:
            print(f"{name:<20} AUC {row['roc_auc']:.4f} ± {row['roc_auc_sd']:.4f}  "
                  f"Brier {row['brier']:.4f}  ({row['fit_seconds']:.0f}s)")
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)


def tune(X, y, n_iter=25, verbose=True) -> dict:
    """RandomizedSearchCV over the four models that feed the ensembles."""
    tuned, report = {}, []
    for name, space in SEARCH_SPACES.items():
        search = RandomizedSearchCV(make_pipeline(base_models()[name]), space, n_iter=n_iter,
                                    scoring="roc_auc", cv=cv_splitter(), n_jobs=-1,
                                    random_state=C.RANDOM_STATE)
        search.fit(X, y)
        tuned[name] = search.best_estimator_.named_steps["model"]
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        report.append({"model": name, "best_cv_auc": search.best_score_, "best_params": params})
        if verbose:
            print(f"tuned {name:<20} AUC {search.best_score_:.4f}  {params}")
    return tuned, pd.DataFrame(report).set_index("model")


def one_se_choice(comparison: pd.DataFrame) -> str:
    """Simplest model whose CV AUC is within one standard error of the best."""
    best = comparison["roc_auc"].idxmax()
    se = comparison.loc[best, "roc_auc_sd"] / np.sqrt(C.CV_FOLDS)
    ok = comparison[comparison["roc_auc"] >= comparison.loc[best, "roc_auc"] - se].copy()
    ok["complexity"] = [COMPLEXITY[m.replace("_tuned", "")] for m in ok.index]
    return ok.sort_values(["complexity", "roc_auc"], ascending=[True, False]).index[0]


def choose_calibration(pipe, X, y) -> pd.DataFrame:
    """Out-of-fold Brier / ECE for raw, sigmoid (Platt) and isotonic calibration."""
    from .metrics import expected_calibration_error
    rows = []
    for method in ("none", "sigmoid", "isotonic"):
        est = pipe if method == "none" else CalibratedClassifierCV(pipe, method=method, cv=5)
        p = cross_val_predict(est, X, y, cv=cv_splitter(), method="predict_proba", n_jobs=-1)[:, 1]
        rows.append({"method": method, "brier": np.mean((p - y) ** 2),
                     "ece": expected_calibration_error(y, p)})
    return pd.DataFrame(rows).set_index("method")


def calibrated(pipe, method):
    return pipe if method == "none" else CalibratedClassifierCV(clone(pipe), method=method, cv=5)


def run(n_iter=25, verbose=True) -> dict:
    C.ensure_dirs()
    df = load_clean()
    X_tr, X_te, y_tr, y_te = train_test(df)

    # 1. Baseline zoo with default-ish hyperparameters.
    pipes = {k: make_pipeline(m) for k, m in base_models().items()}
    if verbose:
        print("== Base model zoo (5-fold CV on train) ==")
    comparison = cv_evaluate(pipes, X_tr, y_tr, verbose)

    # 2. Tune the ensemble members, then build ensembles from the tuned versions.
    if verbose:
        print("\n== Hyperparameter search ==")
    tuned, tuning_report = tune(X_tr, y_tr, n_iter=n_iter, verbose=verbose)
    tuned_pipes = {f"{k}_tuned": make_pipeline(m) for k, m in tuned.items()}
    ens_pipes = {k: make_pipeline(m) for k, m in ensembles(tuned).items()}
    if verbose:
        print("\n== Tuned models + ensembles ==")
    comparison = pd.concat([comparison, cv_evaluate({**tuned_pipes, **ens_pipes}, X_tr, y_tr, verbose)])
    comparison = comparison.sort_values("roc_auc", ascending=False)
    all_pipes = {**pipes, **tuned_pipes, **ens_pipes}

    # 3. Select with the one-standard-error rule, then pick a calibration method.
    final_name = one_se_choice(comparison)
    best_name = comparison["roc_auc"].idxmax()
    cal_report = choose_calibration(all_pipes[final_name], X_tr, y_tr)
    cal_method = cal_report["brier"].idxmin()
    if verbose:
        print(f"\nBest CV AUC: {best_name}; chosen by 1-SE rule: {final_name}; calibration: {cal_method}")

    # 4. Held-out test evaluation for every model (fit on full train).
    test_rows = []
    fitted, test_probs = {}, {}
    for name in comparison.index:
        if name == "dummy":
            continue
        est = clone(all_pipes[name]).fit(X_tr, y_tr)
        fitted[name] = est
        test_probs[name] = est.predict_proba(X_te)[:, 1]
        test_rows.append({"model": name, **prob_metrics(y_te, test_probs[name])})
    final_cal = calibrated(all_pipes[final_name], cal_method).fit(X_tr, y_tr)
    final_label = f"{final_name} + {cal_method}"
    test_probs[final_label] = final_cal.predict_proba(X_te)[:, 1]
    test_rows.append({"model": final_label, **prob_metrics(y_te, test_probs[final_label])})
    test_results = pd.DataFrame(test_rows).set_index("model").sort_values("roc_auc", ascending=False)


    # 5. Out-of-fold calibrated scores for every customer (unbiased input to the A/B design).
    final_est = calibrated(all_pipes[final_name], cal_method)
    oof = cross_val_predict(final_est, *split_xy(df), cv=cv_splitter(),
                            method="predict_proba", n_jobs=-1)[:, 1]
    scored = df[[C.ID_COL, C.TARGET, "contract", "tenure_months", "monthly_charges",
                 "internet_service", "payment_method", "cltv"]].copy()
    scored["churn_prob"] = oof
    scored.to_pickle(C.SCORED_PATH)

    # 6. Persist artefacts. The deployable model is refit on all labelled data.
    deploy = calibrated(all_pipes[final_name], cal_method).fit(*split_xy(df))
    joblib.dump(deploy, C.MODELS_DIR / "churn_model.joblib")
    joblib.dump({"final_name": final_name, "calibration": cal_method,
                 "calibrated": final_cal, "uncalibrated": fitted[final_name],
                 "lightgbm": fitted["lightgbm_tuned"], "test_probs": test_probs,
                 "X_test": X_te, "y_test": y_te, "X_train": X_tr, "y_train": y_tr},
                C.MODELS_DIR / "evaluation_artifacts.joblib")
    comparison.to_csv(C.METRICS_DIR / "cv_comparison.csv")
    test_results.to_csv(C.METRICS_DIR / "test_results.csv")
    cal_report.to_csv(C.METRICS_DIR / "calibration_choice.csv")
    tuning_report.to_csv(C.METRICS_DIR / "tuning_report.csv")
    summary = {"best_cv_model": best_name, "final_model": final_name,
               "calibration": cal_method,
               "final_test_metrics": test_results.loc[final_label].to_dict()}
    (C.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return {"comparison": comparison, "test_results": test_results, "summary": summary,
            "calibration_report": cal_report, "tuning_report": tuning_report}
