"""Model interpretation: logistic-regression odds ratios and LightGBM TreeSHAP values."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C

# Money columns are dropped from the inference model: monthly_charges is ~100% explained
# by the services bought (VIF ~ 860) and total_charges ~ tenure x monthly_charges.
LOGIT_COLS = (["tenure_years"] + [c for c in C.BINARY_COLS if c != "has_internet"]
              + C.CATEGORICAL_COLS)


def design(frame, cols):
    """One-hot design matrix with a constant (first level of each categorical is the reference)."""
    X = pd.get_dummies(frame[cols], columns=[c for c in C.CATEGORICAL_COLS if c in cols],
                       drop_first=True, dtype=float)
    return sm.add_constant(X.astype(float))


def vif(X) -> pd.Series:
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                     index=X.columns).drop("const", errors="ignore")


PREFIXES = {"internet_service_": "Internet: ", "contract_": "Contract: ", "payment_method_": "Pay: "}


def pretty(name: str) -> str:
    """'contract_Two year' -> 'Contract: Two year'; 'tech_support' -> 'Tech support'."""
    for prefix, label in PREFIXES.items():
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name.replace("_", " ").capitalize()


def fit_logit(df):
    """Fit the interpretable logit on all customers; return (model, odds-ratio table)."""
    d = df.assign(tenure_years=df.tenure_months / 12)
    model = sm.Logit(d[C.TARGET], design(d, LOGIT_COLS)).fit(disp=0)
    ci = model.conf_int()
    ors = pd.DataFrame({"odds_ratio": np.exp(model.params), "ci_low": np.exp(ci[0]),
                        "ci_high": np.exp(ci[1]), "p_value": model.pvalues}).drop("const")
    ors.index = [pretty(i) for i in ors.index]
    return model, ors.sort_values("odds_ratio")


def lgbm_shap(pipe, X):
    """Exact TreeSHAP values (log-odds) from LightGBM's native pred_contrib.

    Returns (shap_df, transformed_X, base_value). Rows of shap_df + base sum to the raw score.
    """
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    contrib = model.predict(Xt, pred_contrib=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=Xt.columns, index=Xt.index)
    return shap, Xt, float(contrib[0, -1])


def base_feature(col: str) -> str:
    """Map a one-hot column (e.g. 'contract_Two year') back to its source feature."""
    for c in C.CATEGORICAL_COLS:
        if col.startswith(c + "_"):
            return c
    return col
