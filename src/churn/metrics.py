"""Metrics for probabilistic classifiers."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}


def prob_metrics(y, p, threshold=0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ece": expected_calibration_error(y, p),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat),
    }


def expected_calibration_error(y, p, n_bins=10) -> float:
    """Weighted mean |observed - predicted| over equal-width probability bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y, p, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def threshold_sweep(y, p, thresholds=np.linspace(0.05, 0.95, 19)):
    rows = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        rows.append({"threshold": t, "flagged_share": yhat.mean(),
                     "precision": precision_score(y, yhat, zero_division=0),
                     "recall": recall_score(y, yhat), "f1": f1_score(y, yhat)})
    return pd.DataFrame(rows)


def lift_table(y, p, n_bins=10):
    """Churn rate by predicted-risk decile (decile 1 = highest risk)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), n_bins,
                          labels=range(1, n_bins + 1))
    t = d.groupby("decile", observed=True).agg(customers=("y", "size"), churn_rate=("y", "mean"),
                                               mean_predicted=("p", "mean"))
    t["lift"] = t["churn_rate"] / d["y"].mean()
    t["cum_capture"] = d.groupby("decile", observed=True)["y"].sum().cumsum() / d["y"].sum()
    return t
