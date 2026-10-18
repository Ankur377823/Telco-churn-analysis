"""
interpretation.py — explaining WHY the model predicts churn
===========================================================

WHAT THIS FILE DOES
    Two independent ways of explaining churn, so each can check the other:

    1. ODDS RATIOS from a statsmodels logistic regression
       "A two-year contract multiplies the odds of churn by 0.27, all else equal."
       Gives p-values and confidence intervals, so it is built for inference.

    2. SHAP VALUES from LightGBM
       How much each feature pushed ONE customer's prediction up or down.
       Computed exactly with LightGBM's built-in TreeSHAP (pred_contrib=True),
       because the `shap` package is blocked on this machine. Same numbers.

    In notebook 04 the two rankings agree well (Spearman ρ = 0.81).

WHY SOME COLUMNS ARE LEFT OUT OF THE LOGIT
    monthly_charges  ~100 % explained by the services bought (VIF ≈ 860)
    total_charges    ≈ tenure x monthly charges
    n_addons         the sum of the add-on columns
    has_internet     the same as internet_service = "No"
    Highly collinear inputs make coefficients unstable and meaningless. They
    are fine for prediction, so the ML pipeline keeps them.

MAIN FUNCTIONS
    design, vif          build the logit's input matrix and check collinearity
    fit_logit            fit the logit and return an odds-ratio table
    lgbm_shap            exact SHAP values from a fitted LightGBM pipeline
    pretty, base_feature name helpers for charts
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C

# Inputs of the inference model. Tenure is in YEARS so its odds ratio reads "per extra year".
LOGIT_COLS = (["tenure_years"]
              + [c for c in C.BINARY_COLS if c != "has_internet"]
              + C.CATEGORICAL_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression (inference)
# ─────────────────────────────────────────────────────────────────────────────
def design(frame, cols):
    """One-hot input matrix plus an intercept column ("const").

    drop_first=True drops the first level of each category, which becomes the
    reference group: DSL, Month-to-month and Bank transfer. Every odds ratio
    is "compared with that reference". Keeping all levels plus an intercept
    would be perfectly collinear (the "dummy-variable trap").
    """
    categorical = [c for c in C.CATEGORICAL_COLS if c in cols]
    X = pd.get_dummies(frame[cols], columns=categorical, drop_first=True, dtype=float)
    return sm.add_constant(X.astype(float))


def vif(X) -> pd.Series:
    """Variance inflation factor per column: 1 / (1 − R²) of that column on all others.

    Rule of thumb: above 5 is a worry, above 10 is serious.
    """
    values = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return pd.Series(values, index=X.columns).drop("const", errors="ignore")


def fit_logit(df):
    """Fit the interpretable logit on all customers.

    Fitting on everyone is fine here: the goal is to estimate effects
    (inference), not to predict unseen customers.

    Returns
        model  the fitted statsmodels result (summary(), prsquared, ...)
        ors    odds ratio, 95 % CI and p-value per feature, sorted by odds ratio
               odds ratio < 1 lowers churn, > 1 raises it
    """
    d = df.assign(tenure_years=df.tenure_months / 12)
    model = sm.Logit(d[C.TARGET], design(d, LOGIT_COLS)).fit(disp=0)

    ci = model.conf_int()                   # CI of the coefficients (log-odds scale)
    ors = pd.DataFrame({"odds_ratio": np.exp(model.params),
                        "ci_low": np.exp(ci[0]),
                        "ci_high": np.exp(ci[1]),
                        "p_value": model.pvalues}).drop("const")
    ors.index = [pretty(name) for name in ors.index]
    return model, ors.sort_values("odds_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP values (per-customer explanations)
# ─────────────────────────────────────────────────────────────────────────────
def lgbm_shap(pipe, X):
    """Exact TreeSHAP values (log-odds scale) from a fitted LightGBM pipeline.

    LightGBM returns one column per feature plus a final column holding the
    base value (the average log-odds). For every customer:
        base value + sum of their SHAP values = the model's raw log-odds
    (a test checks this "additivity" to 6 decimal places).

    Returns
        shap_df  SHAP value per customer x model column
        Xt       the preprocessed inputs the model actually saw
        base     the base value
    """
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)

    contributions = model.predict(Xt, pred_contrib=True)
    shap_df = pd.DataFrame(contributions[:, :-1], columns=Xt.columns, index=Xt.index)
    base_value = float(contributions[0, -1])
    return shap_df, Xt, base_value


# ─────────────────────────────────────────────────────────────────────────────
# Name helpers
# ─────────────────────────────────────────────────────────────────────────────
PREFIXES = {"internet_service_": "Internet: ", "contract_": "Contract: ", "payment_method_": "Pay: "}


def pretty(name: str) -> str:
    """Readable label: 'contract_Two year' -> 'Contract: Two year'; 'tech_support' -> 'Tech support'."""
    for prefix, label in PREFIXES.items():
        if name.startswith(prefix):
            return label + name[len(prefix):]
    return name.replace("_", " ").capitalize()


def base_feature(col: str) -> str:
    """Map a one-hot column back to its source: 'contract_Two year' -> 'contract'."""
    for source in C.CATEGORICAL_COLS:
        if col.startswith(source + "_"):
            return source
    return col
