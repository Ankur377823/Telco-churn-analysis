"""
metrics.py — how we judge a churn model
=======================================

WHAT THIS FILE DOES
    Measures a model's predicted churn probabilities from three angles:

    1. RANKING      does it put churners above non-churners?
                    -> ROC-AUC, PR-AUC
    2. PROBABILITY  when it says "30 %", do ~30 % really churn?  (calibration)
                    -> Brier score, log-loss, expected calibration error (ECE)
    3. DECISIONS    at a cut-off, how many churners do we catch, how many false alarms?
                    -> accuracy, precision, recall, F1, threshold sweep, lift table

    Calibration matters here because notebook 05 multiplies probabilities by
    revenue. A model can rank well (high AUC) and still give wrong
    probabilities — naive Bayes and AdaBoost in this project do exactly that.

MAIN FUNCTIONS
    CV_SCORING                    the metrics computed during cross-validation
    prob_metrics(y, p)            all headline metrics in one dict
    expected_calibration_error()  average gap between predicted and observed churn
    reliability_table()           data for a calibration (reliability) curve
    threshold_sweep()             precision / recall at many cut-offs
    lift_table()                  churn rate by risk decile
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)


# sklearn always MAXIMISES a score, so "lower is better" metrics come as negatives
# ("neg_brier_score"); training.cv_evaluate flips their sign back.
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


# ─────────────────────────────────────────────────────────────────────────────
# Headline metrics
# ─────────────────────────────────────────────────────────────────────────────
def prob_metrics(y, p, threshold=0.5) -> dict:
    """Every headline metric for true labels `y` and predicted probabilities `p`.

    Ranking      roc_auc   chance a random churner scores above a random stayer (0.5 = coin flip)
                 pr_auc    precision-recall area; its baseline is the churn rate (0.265), not 0.5
    Probability  brier     mean squared error of the probabilities (lower = better)
                 log_loss  punishes confident mistakes heavily (clipped so log(0) can't happen)
                 ece       average |predicted - observed| churn across probability bins
    Decisions    accuracy, precision, recall, f1 at the given threshold
    """
    y_hat = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ece": expected_calibration_error(y, p),
        "accuracy": accuracy_score(y, y_hat),
        "precision": precision_score(y, y_hat, zero_division=0),   # 0 if nothing is flagged
        "recall": recall_score(y, y_hat),
        "f1": f1_score(y, y_hat),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def expected_calibration_error(y, p, n_bins=10) -> float:
    """Weighted average of |observed churn rate - mean predicted| over 10 equal-width bins.

    Read it as: "on average, the predicted probability is off by this much".
    Each bin's gap is weighted by the share of customers in that bin.
    """
    # p = 1.0 would fall in bin 10; np.minimum keeps it in the last bin (9).
    bin_of = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_of == b
        if in_bin.any():
            gap = abs(y[in_bin].mean() - p[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def reliability_table(y, p, n_bins=10):
    """Mean predicted vs observed churn per bin, for a reliability curve.

    Uses quantile bins so every point on the curve has the same number of customers.
    """
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


# ─────────────────────────────────────────────────────────────────────────────
# Turning scores into actions
# ─────────────────────────────────────────────────────────────────────────────
def threshold_sweep(y, p, thresholds=np.linspace(0.05, 0.95, 19)):
    """Share flagged, precision, recall and F1 at each cut-off.

    Lets the business pick a threshold from its capacity ("we can call 20 % of
    customers") instead of using the default 0.5.
    """
    rows = []
    for t in thresholds:
        y_hat = (p >= t).astype(int)
        rows.append({"threshold": t,
                     "flagged_share": y_hat.mean(),
                     "precision": precision_score(y, y_hat, zero_division=0),
                     "recall": recall_score(y, y_hat),
                     "f1": f1_score(y, y_hat)})
    return pd.DataFrame(rows)


def lift_table(y, p, n_bins=10):
    """Churn rate by predicted-risk decile (decile 1 = the riskiest 10 %).

    Columns
        customers       customers in the decile
        churn_rate      observed churn in the decile
        mean_predicted  average predicted probability in the decile
        lift            churn_rate / overall churn rate  (2.5 = 2.5x the average)
        cum_capture     share of ALL churners found in deciles 1..k
                        ("targeting the top 30 % reaches X % of churners")
    """
    d = pd.DataFrame({"y": y, "p": p})
    # rank(method="first") breaks ties so the ten deciles are equal-sized.
    risk_rank = d["p"].rank(method="first", ascending=False)
    d["decile"] = pd.qcut(risk_rank, n_bins, labels=range(1, n_bins + 1))

    by_decile = d.groupby("decile", observed=True)
    table = by_decile.agg(customers=("y", "size"), churn_rate=("y", "mean"),
                          mean_predicted=("p", "mean"))
    table["lift"] = table["churn_rate"] / d["y"].mean()
    table["cum_capture"] = by_decile["y"].sum().cumsum() / d["y"].sum()
    return table
