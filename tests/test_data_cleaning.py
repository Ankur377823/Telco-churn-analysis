"""
test_data_cleaning.py — does clean() fix every problem, and only those?
=======================================================================

WHAT IS TESTED
    On the 4-customer `raw_small` table (exact answers known):
        blank total charges, 'No internet / phone service' levels, Yes/No -> 0/1,
        engineered features, dropped columns, and cleaning WITHOUT label columns
        (which is how score_customers.py sees new customers).
    On the real data:
        row count, churn rate, no missing features, the 11 total-charges fixes,
        and churn-rate confidence intervals.
    Plus one rule: leakage columns are never model features.
"""
import pytest

from churn import config as C
from churn.data_cleaning import churn_rate_table, clean


def _customer(df, cid, col):
    """Value of `col` for customer `cid`."""
    return df.loc[df.customer_id == cid, col].item()


# ── Hand-made table ──────────────────────────────────────────────────────────
def test_blank_total_charges_become_zero(raw_small):
    df, _ = clean(raw_small)
    assert df["total_charges"].dtype == float
    assert _customer(df, "A", "total_charges") == 0.0


def test_no_internet_service_level_is_collapsed(raw_small):
    df, _ = clean(raw_small)
    for col in C.ADDON_COLS + ["multiple_lines"]:
        assert set(df[col].unique()) <= {0, 1}, col
    assert _customer(df, "B", "has_internet") == 0
    assert _customer(df, "C", "has_internet") == 1


def test_yes_no_columns_are_binary(raw_small):
    df, _ = clean(raw_small)
    assert df.loc[df.customer_id == "C", ["online_security", "tech_support"]].sum(axis=1).item() == 2
    assert _customer(df, "D", "is_male") == 1


def test_n_addons_counts_services(raw_small):
    df, _ = clean(raw_small)
    assert _customer(df, "C", "n_addons") == 2


def test_tenure_zero_falls_in_first_bucket(raw_small):
    # The first bin edge is -1 precisely so that tenure 0 is included.
    df, _ = clean(raw_small)
    assert _customer(df, "A", "tenure_bucket") == "0-6"


def test_constant_and_redundant_columns_dropped(raw_small):
    df, _ = clean(raw_small)
    assert not set(C.CONSTANT_COLS + C.REDUNDANT_COLS) & set(df.columns)


def test_clean_works_without_label_columns(raw_small):
    """score_customers.py scores new customers that have no churn columns."""
    unlabelled = raw_small.drop(columns=["Churn Label", "Churn Score", "CLTV", "Churn Reason"])
    df, _ = clean(unlabelled)
    assert set(C.FEATURES) <= set(df.columns)


# ── Configuration rule ───────────────────────────────────────────────────────
def test_leakage_columns_never_used_as_features():
    assert not set(C.LEAKAGE_COLS) & set(C.FEATURES)
    assert C.TARGET not in C.FEATURES


# ── Real data ────────────────────────────────────────────────────────────────
def test_real_data_shape_and_churn_rate(real_clean):
    df, log = real_clean
    assert df.shape[0] == 7043
    assert df[C.TARGET].mean() == pytest.approx(0.2654, abs=1e-3)
    assert df[C.FEATURES].isna().sum().sum() == 0
    assert log.loc[log.step == "total_charges", "rows_affected"].item() == 11


def test_churn_rate_table_ci_brackets_rate(real_clean):
    t = churn_rate_table(real_clean[0], "contract")
    assert ((t.ci_low <= t.churn_rate) & (t.churn_rate <= t.ci_high)).all()
    assert t.loc["Month-to-month", "churn_rate"] > t.loc["Two year", "churn_rate"]
    assert t["customers"].sum() == 7043
