"""Loading and cleaning the IBM Telco churn workbook.

Every cleaning decision is appended to a log so the notebooks can show
exactly what was changed and why.
"""
import re

import numpy as np
import pandas as pd

from . import config as C


def to_snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def load_raw(path=C.RAW_PATH) -> pd.DataFrame:
    # The workbook has an Excel autofilter that hides rows; pandas ignores
    # filters and reads every row, which we assert below.
    df = pd.read_excel(path)
    assert len(df) == 7043, f"expected 7043 rows, got {len(df)}"
    return df


def clean(raw: pd.DataFrame):
    """Return (clean_df, decision_log)."""
    log = []

    def note(step, issue, action, reason, rows=None):
        log.append({"step": step, "issue": issue, "action": action,
                    "reason": reason, "rows_affected": rows})

    df = raw.copy()
    df.columns = [to_snake(c) for c in df.columns]
    df = df.rename(columns={"churn_value": C.TARGET, "customerid": C.ID_COL})
    note("names", "Mixed-case column names with spaces", "Converted to snake_case",
         "Consistent attribute access in code")

    dup = int(df[C.ID_COL].duplicated().sum())
    note("duplicates", "Possible duplicate customers", "Checked customer_id",
         "One row per customer is the unit of analysis", dup)

    # Total Charges: mixed float/int/str, 11 blanks (' ').
    tc = pd.to_numeric(df["total_charges"], errors="coerce")
    blank = tc.isna()
    assert (df.loc[blank, "tenure_months"] == 0).all()
    df["total_charges"] = tc.fillna(0.0)
    note("total_charges", "Stored as text; 11 values are a single space",
         "Coerced to float; blanks set to 0",
         "All 11 blanks have tenure 0: brand-new customers who have not been billed yet. "
         "Dropping them would bias against new customers; imputing a mean would invent revenue.",
         int(blank.sum()))

    # "No internet service" / "No phone service" duplicate information already in
    # internet_service / phone_service. Collapse to "No" and keep an explicit flag.
    df["has_internet"] = (df["internet_service"] != "No").astype(int)
    n_noint = int((df["online_security"] == "No internet service").sum())
    for c in C.ADDON_COLS:
        df[c] = df[c].replace("No internet service", "No")
    note("add-ons", "'No internet service' is a third level of every add-on column",
         "Mapped to 'No' and added has_internet flag",
         "Otherwise the same fact is one-hot encoded 7 times, which creates perfect "
         "collinearity in the logistic regression", n_noint)

    n_nophone = int((df["multiple_lines"] == "No phone service").sum())
    df["multiple_lines"] = df["multiple_lines"].replace("No phone service", "No")
    note("multiple_lines", "'No phone service' level duplicates phone_service",
         "Mapped to 'No'", "phone_service already carries this", n_nophone)

    yes_no = ["senior_citizen", "partner", "dependents", "phone_service",
              "multiple_lines", *C.ADDON_COLS, "paperless_billing"]
    for c in yes_no:
        assert set(df[c].unique()) <= {"Yes", "No"}, c
        df[c] = (df[c] == "Yes").astype(int)
    df["is_male"] = (df["gender"] == "Male").astype(int)
    df = df.drop(columns="gender")
    note("binary", "Yes/No stored as strings", "Encoded as 0/1 integers",
         "Lets churn rates be computed as means and models consume them directly")

    df["n_addons"] = df[C.ADDON_COLS].sum(axis=1)
    df["tenure_bucket"] = pd.cut(df["tenure_months"], C.TENURE_BINS,
                                 labels=C.TENURE_LABELS)
    note("features", "Tenure is right-skewed with many brand-new customers",
         "Added tenure_bucket (0-6, 7-12, 13-24, 25-48, 49-72) and n_addons",
         "Buckets make the tenure effect readable in EDA; n_addons summarises service depth")

    for c in C.CATEGORICAL_COLS:
        df[c] = df[c].astype("category")

    note("leakage", "churn_score, cltv, churn_reason, churn_label",
         "Kept in the clean table but excluded from model features",
         "churn_score is IBM's own model output; churn_reason exists only for churners; "
         "churn_label duplicates the target; cltv is used for framing only")
    note("constant", "count, country, state each have one value; lat_long duplicates lat/lon",
         "Dropped", "No information")
    df = df.drop(columns=C.CONSTANT_COLS + C.REDUNDANT_COLS, errors="ignore")

    if "churn_reason" in df:  # absent when scoring new, unlabelled customers
        df["churn_reason"] = df["churn_reason"].fillna("Did not churn")
    return df, pd.DataFrame(log)


def build_clean(save=True):
    C.ensure_dirs()
    df, log = clean(load_raw())
    if save:
        df.to_pickle(C.CLEAN_PATH)
    return df, log


def load_clean() -> pd.DataFrame:
    if not C.CLEAN_PATH.exists():
        build_clean()
    return pd.read_pickle(C.CLEAN_PATH)


def split_xy(df):
    return df[C.FEATURES].copy(), df[C.TARGET].to_numpy()


def churn_rate_table(df, col):
    """Churn rate, count and Wilson 95% CI per level of `col`."""
    from statsmodels.stats.proportion import proportion_confint
    g = df.groupby(col, observed=True)[C.TARGET].agg(["sum", "count"])
    g["churn_rate"] = g["sum"] / g["count"]
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    g["ci_low"], g["ci_high"] = np.asarray(lo), np.asarray(hi)
    return g.rename(columns={"sum": "churned", "count": "customers"})
