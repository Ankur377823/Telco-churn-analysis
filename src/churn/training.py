"""
training.py — the model-training engine
=======================================

WHAT THIS FILE DOES
    Runs the whole modelling process end to end. `run()` is the entry point and
    calls six clearly separated stages:

    Stage 1  Compare the 12 base models with 5-fold cross-validation (CV)
    Stage 2  Tune 4 of them with random search, then build 2 ensembles from them
    Stage 3  Pick the final model with the one-standard-error rule, then decide
             whether its probabilities need calibrating (none / Platt / isotonic)
    Stage 4  Score every model ONCE on the untouched 20 % test set (reporting only)
    Stage 5  Give every customer an out-of-fold churn probability
             (each score comes from a model that never saw that customer)
    Stage 6  Save the deployable model, evaluation artefacts and metric reports

THE THREE KINDS OF "FIT" AND WHY
    fit on the 80 % training split    -> honest test-set scores (stage 4)
    out-of-fold over all customers    -> honest per-customer scores for the A/B design (stage 5)
    fit on all 7,043 customers        -> the best possible model to deploy (stage 6)

    Model selection only ever uses CV on the training split. The test set is
    looked at once, at the end, so its score stays an unbiased estimate.

MAIN FUNCTIONS
    cv_splitter, train_test      how the data is split
    cv_evaluate                  CV scores for a dict of pipelines
    tune                         RandomizedSearchCV for the four ensemble members
    one_se_choice                the one-standard-error selection rule
    choose_calibration           compare none / sigmoid / isotonic calibration
    run                          all six stages
"""
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
from .metrics import CV_SCORING, expected_calibration_error, prob_metrics
from .model_zoo import COMPLEXITY, SEARCH_SPACES, base_models, ensembles
from .preprocessing import make_pipeline

# Metrics where lower is better; sklearn reports them as negatives, we flip them back.
LOWER_IS_BETTER = ("brier", "log_loss")


def _say(verbose, message):
    """Print progress only when verbose is on."""
    if verbose:
        print(message)


# ─────────────────────────────────────────────────────────────────────────────
# Splitting the data
# ─────────────────────────────────────────────────────────────────────────────
def cv_splitter():
    """5 stratified folds with a fixed seed.

    Stratified: every fold keeps the 26.5 % churn rate.
    Fixed seed: every model is scored on the *same* folds, so comparisons are fair.
    """
    return StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def train_test(df):
    """80 / 20 stratified split -> X_train, X_test, y_train, y_test."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated comparison
# ─────────────────────────────────────────────────────────────────────────────
def _summarise_cv(name, cv_result, seconds):
    """Turn sklearn's per-fold scores into one row: mean and sd for every metric."""
    row = {"model": name, "fit_seconds": seconds}
    for metric in CV_SCORING:
        fold_scores = cv_result[f"test_{metric}"]
        if metric in LOWER_IS_BETTER:
            fold_scores = -fold_scores
        row[metric] = fold_scores.mean()
        row[f"{metric}_sd"] = fold_scores.std(ddof=1)     # sample standard deviation
    return row


def cv_evaluate(pipes: dict, X, y, verbose=True) -> pd.DataFrame:
    """5-fold stratified CV of each pipeline; one row per model, best AUC first."""
    rows = []
    for name, pipe in pipes.items():
        start = time.time()
        result = cross_validate(pipe, X, y, cv=cv_splitter(), scoring=CV_SCORING, n_jobs=-1)
        row = _summarise_cv(name, result, time.time() - start)
        rows.append(row)
        _say(verbose, f"{name:<20} AUC {row['roc_auc']:.4f} ± {row['roc_auc_sd']:.4f}  "
                      f"Brier {row['brier']:.4f}  ({row['fit_seconds']:.0f}s)")
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune(X, y, n_iter=25, verbose=True):
    """Random search over the four models that later feed the ensembles.

    Random search instead of grid search: with several hyperparameters it
    covers each one more thoroughly for the same number of trials.

    Returns
        tuned   {name: best bare estimator}
        report  DataFrame of best CV AUC and best parameters per model
    """
    tuned, report = {}, []
    for name, space in SEARCH_SPACES.items():
        search = RandomizedSearchCV(make_pipeline(base_models()[name]), space, n_iter=n_iter,
                                    scoring="roc_auc", cv=cv_splitter(), n_jobs=-1,
                                    random_state=C.RANDOM_STATE)
        search.fit(X, y)

        tuned[name] = search.best_estimator_.named_steps["model"]
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        report.append({"model": name, "best_cv_auc": search.best_score_, "best_params": params})
        _say(verbose, f"tuned {name:<20} AUC {search.best_score_:.4f}  {params}")
    return tuned, pd.DataFrame(report).set_index("model")


# ─────────────────────────────────────────────────────────────────────────────
# Model selection and calibration
# ─────────────────────────────────────────────────────────────────────────────
def one_se_choice(comparison: pd.DataFrame) -> str:
    """Pick the simplest model whose CV AUC is within one standard error of the best.

    standard error = sd of the 5 fold AUCs / sqrt(5)

    Models inside that band are statistically tied with the best, so we prefer
    the simplest one (lowest COMPLEXITY), breaking ties by higher AUC. Simpler
    means easier to explain, cheaper to run and more stable.
    """
    best = comparison["roc_auc"].idxmax()
    standard_error = comparison.loc[best, "roc_auc_sd"] / np.sqrt(C.CV_FOLDS)
    cutoff = comparison.loc[best, "roc_auc"] - standard_error

    tied = comparison[comparison["roc_auc"] >= cutoff].copy()
    tied["complexity"] = [COMPLEXITY[m.replace("_tuned", "")] for m in tied.index]
    return tied.sort_values(["complexity", "roc_auc"], ascending=[True, False]).index[0]


def calibrated(pipe, method):
    """A fresh, unfitted copy of `pipe`, wrapped in a calibrator unless method is "none".

    sigmoid  = Platt scaling: fits a logistic curve on top of the scores
    isotonic = fits a free-form increasing step function (more flexible, can overfit)

    Always returning a copy matters: otherwise the train-only model from stage 4
    and the all-data model from stage 6 would be the same object, and the later
    fit would silently overwrite the earlier one.
    """
    if method == "none":
        return clone(pipe)
    return CalibratedClassifierCV(clone(pipe), method=method, cv=5)


def choose_calibration(pipe, X, y) -> pd.DataFrame:
    """Out-of-fold Brier score and ECE for raw, sigmoid and isotonic probabilities.

    Every probability is predicted by a model that did not see that customer,
    so the comparison is honest. The lowest Brier score wins.
    """
    rows = []
    for method in ("none", "sigmoid", "isotonic"):
        estimator = pipe if method == "none" else CalibratedClassifierCV(pipe, method=method, cv=5)
        p = cross_val_predict(estimator, X, y, cv=cv_splitter(),
                              method="predict_proba", n_jobs=-1)[:, 1]
        rows.append({"method": method,
                     "brier": np.mean((p - y) ** 2),
                     "ece": expected_calibration_error(y, p)})
    return pd.DataFrame(rows).set_index("method")


# ─────────────────────────────────────────────────────────────────────────────
# The six stages of run()
# ─────────────────────────────────────────────────────────────────────────────
def _stage1_compare_base_models(X_tr, y_tr, verbose):
    """Wrap each of the 12 base models in the shared pipeline and cross-validate it."""
    _say(verbose, "== Base model zoo (5-fold CV on train) ==")
    pipes = {name: make_pipeline(model) for name, model in base_models().items()}
    return pipes, cv_evaluate(pipes, X_tr, y_tr, verbose)


def _stage2_tune_and_ensemble(X_tr, y_tr, n_iter, verbose):
    """Tune LR / RF / XGBoost / LightGBM, build voting + stacking, cross-validate them all."""
    _say(verbose, "\n== Hyperparameter search ==")
    tuned, tuning_report = tune(X_tr, y_tr, n_iter=n_iter, verbose=verbose)

    tuned_pipes = {f"{name}_tuned": make_pipeline(model) for name, model in tuned.items()}
    ensemble_pipes = {name: make_pipeline(model) for name, model in ensembles(tuned).items()}
    new_pipes = {**tuned_pipes, **ensemble_pipes}

    _say(verbose, "\n== Tuned models + ensembles ==")
    return new_pipes, cv_evaluate(new_pipes, X_tr, y_tr, verbose), tuning_report


def _stage3_select(comparison, all_pipes, X_tr, y_tr, verbose):
    """One-SE rule picks the model; out-of-fold Brier picks the calibration method."""
    final_name = one_se_choice(comparison)
    best_name = comparison["roc_auc"].idxmax()
    calibration_report = choose_calibration(all_pipes[final_name], X_tr, y_tr)
    calibration_method = calibration_report["brier"].idxmin()
    _say(verbose, f"\nBest CV AUC: {best_name}; chosen by 1-SE rule: {final_name}; "
                  f"calibration: {calibration_method}")
    return final_name, best_name, calibration_method, calibration_report


def _stage4_test_set(comparison, all_pipes, final_name, calibration_method,
                     X_tr, X_te, y_tr, y_te):
    """Fit every model on the training split and score it once on the test split.

    The test set is used for reporting only — the choice was already made in stage 3.
    """
    rows, fitted, test_probs = [], {}, {}
    for name in comparison.index:
        if name == "dummy":
            continue
        model = clone(all_pipes[name]).fit(X_tr, y_tr)
        fitted[name] = model
        test_probs[name] = model.predict_proba(X_te)[:, 1]
        rows.append({"model": name, **prob_metrics(y_te, test_probs[name])})

    # The final model with its chosen calibration, reported as its own row.
    final_calibrated = calibrated(all_pipes[final_name], calibration_method).fit(X_tr, y_tr)
    final_label = f"{final_name} + {calibration_method}"
    test_probs[final_label] = final_calibrated.predict_proba(X_te)[:, 1]
    rows.append({"model": final_label, **prob_metrics(y_te, test_probs[final_label])})

    test_results = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    return test_results, fitted, test_probs, final_calibrated, final_label


def _stage5_out_of_fold_scores(df, final_pipe, calibration_method):
    """Churn probability for every customer from a model that never saw them.

    In-sample scores would be over-confident for customers the model trained on,
    which would distort the baseline used for the A/B power and ROI maths.
    """
    estimator = calibrated(final_pipe, calibration_method)
    oof = cross_val_predict(estimator, *split_xy(df), cv=cv_splitter(),
                            method="predict_proba", n_jobs=-1)[:, 1]
    scored = df[[C.ID_COL, C.TARGET, "contract", "tenure_months", "monthly_charges",
                 "internet_service", "payment_method", "cltv"]].copy()
    scored["churn_prob"] = oof
    scored.to_pickle(C.SCORED_PATH)


def _stage6_save(df, all_pipes, final_name, best_name, calibration_method, final_calibrated,
                 fitted, test_probs, final_label, splits, reports):
    """Persist the deployable model, the evaluation artefacts and every metric table."""
    X_tr, X_te, y_tr, y_te = splits
    comparison, test_results, calibration_report, tuning_report = reports

    # Deployable model: once evaluation is done, refit on ALL labelled customers.
    deploy = calibrated(all_pipes[final_name], calibration_method).fit(*split_xy(df))
    joblib.dump(deploy, C.MODELS_DIR / "churn_model.joblib")

    # Everything notebook 04 needs, so it never has to retrain.
    joblib.dump({"final_name": final_name, "calibration": calibration_method,
                 "calibrated": final_calibrated, "uncalibrated": fitted[final_name],
                 "lightgbm": fitted["lightgbm_tuned"], "test_probs": test_probs,
                 "X_test": X_te, "y_test": y_te, "X_train": X_tr, "y_train": y_tr},
                C.MODELS_DIR / "evaluation_artifacts.joblib")

    comparison.to_csv(C.METRICS_DIR / "cv_comparison.csv")
    test_results.to_csv(C.METRICS_DIR / "test_results.csv")
    calibration_report.to_csv(C.METRICS_DIR / "calibration_choice.csv")
    tuning_report.to_csv(C.METRICS_DIR / "tuning_report.csv")

    summary = {"best_cv_model": best_name, "final_model": final_name,
               "calibration": calibration_method,
               "final_test_metrics": test_results.loc[final_label].to_dict()}
    # default=float lets numpy numbers be written as JSON
    (C.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(n_iter=25, verbose=True) -> dict:
    """Run all six stages and return the main result tables.

    Returns {"comparison", "test_results", "summary", "calibration_report", "tuning_report"}.
    """
    C.ensure_dirs()
    df = load_clean()
    X_tr, X_te, y_tr, y_te = train_test(df)

    # Stage 1 + 2: 12 base models, then 4 tuned models + 2 ensembles = 18 rows.
    base_pipes, comparison = _stage1_compare_base_models(X_tr, y_tr, verbose)
    new_pipes, new_comparison, tuning_report = _stage2_tune_and_ensemble(X_tr, y_tr, n_iter, verbose)
    comparison = pd.concat([comparison, new_comparison]).sort_values("roc_auc", ascending=False)
    all_pipes = {**base_pipes, **new_pipes}

    # Stage 3: choose the model and its calibration.
    final_name, best_name, calibration_method, calibration_report = _stage3_select(
        comparison, all_pipes, X_tr, y_tr, verbose)

    # Stage 4: one look at the held-out test set.
    test_results, fitted, test_probs, final_calibrated, final_label = _stage4_test_set(
        comparison, all_pipes, final_name, calibration_method, X_tr, X_te, y_tr, y_te)

    # Stage 5: honest per-customer scores for notebook 05.
    _stage5_out_of_fold_scores(df, all_pipes[final_name], calibration_method)

    # Stage 6: save everything.
    summary = _stage6_save(df, all_pipes, final_name, best_name, calibration_method,
                           final_calibrated, fitted, test_probs, final_label,
                           splits=(X_tr, X_te, y_tr, y_te),
                           reports=(comparison, test_results, calibration_report, tuning_report))

    return {"comparison": comparison, "test_results": test_results, "summary": summary,
            "calibration_report": calibration_report, "tuning_report": tuning_report}
