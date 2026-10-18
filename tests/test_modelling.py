"""
test_modelling.py — preprocessing, models, selection rule, metrics and SHAP
===========================================================================

WHAT IS TESTED
    Preprocessing  output is all finite numbers with the expected 29 columns;
                   an unseen category does not crash it
    Models         every model in the zoo returns valid probabilities in [0, 1]
                   (SVM is skipped because it is slow)
    Selection      the one-SE rule picks the simplest tied model, but keeps a
                   clearly better model
    Metrics        perfect predictions score perfectly; ECE spots miscalibration
    SHAP           base value + SHAP values = the model's raw output (additivity)

Tests use an 800-customer sample of the real data to stay fast.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from churn import config as C
from churn.interpretation import base_feature, lgbm_shap, pretty
from churn.metrics import expected_calibration_error, prob_metrics
from churn.model_zoo import base_models
from churn.preprocessing import build_preprocessor, make_pipeline
from churn.training import one_se_choice


@pytest.fixture(scope="module")
def sample(real_clean):
    """800 random real customers as (X, y)."""
    df = real_clean[0].sample(800, random_state=0)
    return df[C.FEATURES], df[C.TARGET].to_numpy()


# ── Preprocessing ────────────────────────────────────────────────────────────
def test_preprocessor_outputs_finite_matrix(sample):
    X, _ = sample
    Xt = build_preprocessor().fit_transform(X)
    assert np.isfinite(Xt).all()
    n_onehot = sum(X[c].nunique() for c in C.CATEGORICAL_COLS)   # 3 internet + 3 contract + 4 payment
    assert Xt.shape == (len(X), len(C.NUMERIC_COLS) + len(C.BINARY_COLS) + n_onehot)


def test_preprocessor_ignores_unseen_category(sample):
    X, _ = sample
    prep = build_preprocessor().fit(X)
    odd = X.head(1).copy()
    odd["payment_method"] = odd["payment_method"].cat.add_categories("Crypto")
    odd.loc[odd.index[0], "payment_method"] = "Crypto"
    assert np.isfinite(prep.transform(odd)).all()


# ── Every model gives valid probabilities ────────────────────────────────────
@pytest.mark.parametrize("name", [m for m in base_models() if m != "svm_rbf"])
def test_every_model_returns_valid_probabilities(name, sample):
    X, y = sample
    p = make_pipeline(clone(base_models()[name])).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


# ── One-standard-error rule ──────────────────────────────────────────────────
def test_one_se_rule_prefers_simplest_tied_model():
    comp = pd.DataFrame({"roc_auc": [0.865, 0.863, 0.840], "roc_auc_sd": [0.012, 0.013, 0.010]},
                        index=["stacking", "logistic_regression_tuned", "knn"])
    assert one_se_choice(comp) == "logistic_regression_tuned"


def test_one_se_rule_keeps_best_when_clearly_ahead():
    comp = pd.DataFrame({"roc_auc": [0.90, 0.85], "roc_auc_sd": [0.005, 0.005]},
                        index=["xgboost_tuned", "logistic_regression"])
    assert one_se_choice(comp) == "xgboost_tuned"


# ── Metrics ──────────────────────────────────────────────────────────────────
def test_metrics_on_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = prob_metrics(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert m["roc_auc"] == 1 and m["brier"] == 0 and m["f1"] == 1


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)            # outcomes drawn from p itself
    assert expected_calibration_error(y, p) < 0.02            # calibrated
    assert expected_calibration_error(y, np.sqrt(p)) > 0.1     # over-predicts -> miscalibrated


# ── Interpretation ───────────────────────────────────────────────────────────
def test_shap_values_sum_to_model_output(sample):
    X, y = sample
    pipe = make_pipeline(clone(base_models()["lightgbm"])).fit(X, y)
    shap, Xt, base = lgbm_shap(pipe, X)
    raw = pipe.named_steps["model"].predict(Xt, raw_score=True)
    np.testing.assert_allclose(base + shap.sum(axis=1), raw, atol=1e-6)


def test_feature_name_helpers():
    assert pretty("contract_Two year") == "Contract: Two year"
    assert pretty("tech_support") == "Tech support"
    assert base_feature("payment_method_Electronic check") == "payment_method"
    assert base_feature("tenure_months") == "tenure_months"
