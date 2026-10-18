"""
data_cleaning.py — load the raw IBM workbook and turn it into a clean table
===========================================================================

WHAT THIS FILE DOES
    1. Loads the raw Excel file (7,043 customers x 33 columns).
    2. Cleans it through a series of small, named steps (one function each).
    3. Records every decision — what was wrong, what was done, and why — in a
       "decision log" that notebook 02 displays.
    4. Provides small helpers used everywhere else: load the cached clean
       table, split it into X / y, and compute churn rates with confidence
       intervals.

WHY CLEANING LIVES HERE (AND NOT IN A NOTEBOOK)
    The notebooks, the training pipeline, the scoring script and the tests all
    call the very same `clean()` function. Training data and new customers are
    therefore always cleaned identically ("train/serve consistency").

THE CLEANING STEPS, IN ORDER
    rename columns -> check duplicates -> fix total_charges -> collapse
    "No internet service" -> collapse "No phone service" -> Yes/No to 0/1 ->
    add engineered features -> set categorical dtypes -> note leakage columns
    -> drop constant columns -> fill churn_reason

MAIN FUNCTIONS
    load_raw()          read the Excel file and check the row count
    clean(raw)          run every cleaning step; returns (clean_df, decision_log)
    build_clean()       load + clean + save to data/processed/telco_clean.pkl
    load_clean()        read the saved clean table (building it if missing)
    split_xy(df)        model inputs X and target y
    churn_rate_table()  churn rate per group with Wilson 95 % CIs
"""
import re

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Decision log: a tiny helper that records every cleaning decision
# ─────────────────────────────────────────────────────────────────────────────
class DecisionLog:
    """Collects one row per cleaning decision and turns them into a DataFrame."""

    def __init__(self):
        self.rows = []

    def note(self, step, issue, action, reason, rows=None):
        """Record a decision.

        step   short name of the cleaning step
        issue  what was wrong with the raw data
        action what we did about it
        reason why that was the right thing to do
        rows   how many rows were affected (optional)
        """
        self.rows.append({"step": step, "issue": issue, "action": action,
                          "reason": reason, "rows_affected": rows})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def to_snake(name: str) -> str:
    """'Tenure Months' -> 'tenure_months'  (any run of non-alphanumerics becomes '_')."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def load_raw(path=C.RAW_PATH) -> pd.DataFrame:
    """Read the raw IBM workbook and check it has the expected 7,043 rows.

    The workbook has an Excel autofilter that hides some rows inside Excel.
    pandas ignores filters and reads every row; the assert proves it and stops
    the program early if the file is ever truncated or replaced.
    """
    df = pd.read_excel(path)
    assert len(df) == 7043, f"expected 7043 rows, got {len(df)}"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning steps — each takes the table + the log and returns the new table
# ─────────────────────────────────────────────────────────────────────────────
def _rename_columns(df, log):
    """Use snake_case names, and friendlier names for the target and the ID."""
    df.columns = [to_snake(c) for c in df.columns]
    df = df.rename(columns={"churn_value": C.TARGET, "customerid": C.ID_COL})
    log.note("names", "Mixed-case column names with spaces", "Converted to snake_case",
             "Consistent attribute access in code")
    return df


def _check_duplicates(df, log):
    """Count duplicate customer IDs (there are none, but we check and record it)."""
    n_duplicates = int(df[C.ID_COL].duplicated().sum())
    log.note("duplicates", "Possible duplicate customers", "Checked customer_id",
             "One row per customer is the unit of analysis", n_duplicates)
    return df


def _fix_total_charges(df, log):
    """Turn total_charges into numbers; its 11 blank values become 0.

    The raw column mixes floats, ints and text, and 11 values are a single
    space ' '. All 11 belong to customers with tenure 0 — they joined this
    month and have not been billed yet — so 0 is the true value, not a guess.
    The assert proves that claim every time the code runs.
    """
    total = pd.to_numeric(df["total_charges"], errors="coerce")   # ' ' -> NaN
    is_blank = total.isna()
    assert (df.loc[is_blank, "tenure_months"] == 0).all()
    df["total_charges"] = total.fillna(0.0)
    log.note("total_charges", "Stored as text; 11 values are a single space",
             "Coerced to float; blanks set to 0",
             "All 11 blanks have tenure 0: brand-new customers who have not been billed yet. "
             "Dropping them would bias against new customers; imputing a mean would invent revenue.",
             int(is_blank.sum()))
    return df


def _collapse_no_internet(df, log):
    """Map 'No internet service' to 'No' in the six add-on columns and add has_internet.

    'No internet service' repeats a fact already stored in internet_service.
    Left alone, one-hot encoding would create seven columns that all say the
    same thing — perfect collinearity, which breaks logistic regression.
    """
    df["has_internet"] = (df["internet_service"] != "No").astype(int)
    n_no_internet = int((df["online_security"] == "No internet service").sum())
    for col in C.ADDON_COLS:
        df[col] = df[col].replace("No internet service", "No")
    log.note("add-ons", "'No internet service' is a third level of every add-on column",
             "Mapped to 'No' and added has_internet flag",
             "Otherwise the same fact is one-hot encoded 7 times, which creates perfect "
             "collinearity in the logistic regression", n_no_internet)
    return df


def _collapse_no_phone(df, log):
    """Map 'No phone service' to 'No' in multiple_lines (phone_service already says it)."""
    n_no_phone = int((df["multiple_lines"] == "No phone service").sum())
    df["multiple_lines"] = df["multiple_lines"].replace("No phone service", "No")
    log.note("multiple_lines", "'No phone service' level duplicates phone_service",
             "Mapped to 'No'", "phone_service already carries this", n_no_phone)
    return df


def _encode_yes_no(df, log):
    """Turn every Yes/No column into 0/1, and gender into is_male.

    With 0/1 columns, the mean of a column IS a rate (e.g. the churn rate),
    and models can use them directly. The assert guards against surprise
    values such as 'yes ' or 'N/A' slipping through.
    """
    yes_no_cols = ["senior_citizen", "partner", "dependents", "phone_service",
                   "multiple_lines", *C.ADDON_COLS, "paperless_billing"]
    for col in yes_no_cols:
        assert set(df[col].unique()) <= {"Yes", "No"}, col
        df[col] = (df[col] == "Yes").astype(int)

    df["is_male"] = (df["gender"] == "Male").astype(int)
    df = df.drop(columns="gender")
    log.note("binary", "Yes/No stored as strings", "Encoded as 0/1 integers",
             "Lets churn rates be computed as means and models consume them directly")
    return df


def _add_features(df, log):
    """Add n_addons (how many add-ons a customer has) and tenure_bucket (for EDA)."""
    df["n_addons"] = df[C.ADDON_COLS].sum(axis=1)
    df["tenure_bucket"] = pd.cut(df["tenure_months"], C.TENURE_BINS, labels=C.TENURE_LABELS)
    log.note("features", "Tenure is right-skewed with many brand-new customers",
             "Added tenure_bucket (0-6, 7-12, 13-24, 25-48, 49-72) and n_addons",
             "Buckets make the tenure effect readable in EDA; n_addons summarises service depth")
    return df


def _set_categorical_types(df, log):
    """Store the three multi-level text columns as pandas 'category' (compact, ordered levels)."""
    for col in C.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def _note_leakage(df, log):
    """Leakage columns stay in the table (EDA uses them) but are never model features."""
    log.note("leakage", "churn_score, cltv, churn_reason, churn_label",
             "Kept in the clean table but excluded from model features",
             "churn_score is IBM's own model output; churn_reason exists only for churners; "
             "churn_label duplicates the target; cltv is used for framing only")
    return df


def _drop_constant_columns(df, log):
    """Drop columns with a single value and the duplicated lat_long column."""
    log.note("constant", "count, country, state each have one value; lat_long duplicates lat/lon",
             "Dropped", "No information")
    return df.drop(columns=C.CONSTANT_COLS + C.REDUNDANT_COLS, errors="ignore")


def _fill_churn_reason(df, log):
    """Customers who stayed have no churn reason; label them explicitly.

    The column is missing when scoring new, unlabelled customers, so the step
    only runs if the column exists. This lets score_customers.py reuse clean().
    """
    if "churn_reason" in df:
        df["churn_reason"] = df["churn_reason"].fillna("Did not churn")
    return df


# The full recipe. clean() runs these one after another, in this order.
CLEANING_STEPS = [
    _rename_columns,
    _check_duplicates,
    _fix_total_charges,
    _collapse_no_internet,
    _collapse_no_phone,
    _encode_yes_no,
    _add_features,
    _set_categorical_types,
    _note_leakage,
    _drop_constant_columns,
    _fill_churn_reason,
]


def clean(raw: pd.DataFrame):
    """Run every cleaning step on a copy of `raw`.

    Returns
        clean_df      the cleaned table (7,043 x 32 for the full dataset)
        decision_log  DataFrame with one row per decision (step, issue, action, reason, rows)
    """
    df = raw.copy()          # never modify the caller's data
    log = DecisionLog()
    for step in CLEANING_STEPS:
        df = step(df, log)
    return df, log.to_frame()


# ─────────────────────────────────────────────────────────────────────────────
# Saving / loading the clean table
# ─────────────────────────────────────────────────────────────────────────────
def build_clean(save=True):
    """Load the raw file, clean it, and (by default) save it as a pickle.

    Pickle rather than parquet: pyarrow is blocked on this machine, and pickle
    keeps pandas 'category' columns exactly as they are.
    """
    C.ensure_dirs()
    df, log = clean(load_raw())
    if save:
        df.to_pickle(C.CLEAN_PATH)
    return df, log


def load_clean() -> pd.DataFrame:
    """Return the clean table, building and caching it on first use."""
    if not C.CLEAN_PATH.exists():
        build_clean()
    return pd.read_pickle(C.CLEAN_PATH)


def split_xy(df):
    """Model inputs X (the 22 FEATURES columns) and the target y as a numpy array."""
    return df[C.FEATURES].copy(), df[C.TARGET].to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Churn rates with confidence intervals (used in EDA charts and tests)
# ─────────────────────────────────────────────────────────────────────────────
def churn_rate_table(df, col):
    """Churn rate for each level of `col`, with a Wilson 95 % confidence interval.

    Why Wilson? The simpler normal ("Wald") interval misbehaves for small
    groups or rates near 0 % / 100 % — it can even go below zero. Wilson
    stays inside [0, 1] and has better coverage (e.g. two-year contracts, 3 %).

    Returns a DataFrame indexed by level with columns:
        churned, customers, churn_rate, ci_low, ci_high
    """
    table = df.groupby(col, observed=True)[C.TARGET].agg(["sum", "count"])
    table["churn_rate"] = table["sum"] / table["count"]
    low, high = proportion_confint(table["sum"], table["count"], method="wilson")
    table["ci_low"], table["ci_high"] = np.asarray(low), np.asarray(high)
    return table.rename(columns={"sum": "churned", "count": "customers"})
