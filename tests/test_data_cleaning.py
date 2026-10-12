import pytest

from churn import config as C
from churn.data_cleaning import churn_rate_table, clean


def test_blank_total_charges_become_zero(raw_small):
    df, _ = clean(raw_small)
    assert df["total_charges"].dtype == float
    assert df.loc[df.customer_id == "A", "total_charges"].item() == 0.0


def test_no_internet_service_level_is_collapsed(raw_small):
    df, _ = clean(raw_small)
    for c in C.ADDON_COLS + ["multiple_lines"]:
        assert set(df[c].unique()) <= {0, 1}, c
    assert df.loc[df.customer_id == "B", "has_internet"].item() == 0
    assert df.loc[df.customer_id == "C", "has_internet"].item() == 1


def test_yes_no_columns_are_binary(raw_small):
    df, _ = clean(raw_small)
    assert df.loc[df.customer_id == "C", ["online_security", "tech_support"]].sum(axis=1).item() == 2
    assert df.loc[df.customer_id == "D", "is_male"].item() == 1


def test_n_addons_counts_services(raw_small):
    df, _ = clean(raw_small)
    assert df.set_index("customer_id").loc["C", "n_addons"] == 2


def test_tenure_zero_falls_in_first_bucket(raw_small):
    df, _ = clean(raw_small)
    assert df.loc[df.customer_id == "A", "tenure_bucket"].item() == "0-6"


def test_constant_and_redundant_columns_dropped(raw_small):
    df, _ = clean(raw_small)
    assert not set(C.CONSTANT_COLS + C.REDUNDANT_COLS) & set(df.columns)


def test_leakage_columns_never_used_as_features():
    assert not set(C.LEAKAGE_COLS) & set(C.FEATURES)
    assert C.TARGET not in C.FEATURES


def test_clean_works_without_label_columns(raw_small):
    """score_customers.py scores new customers that have no churn columns."""
    unlabelled = raw_small.drop(columns=["Churn Label", "Churn Score", "CLTV", "Churn Reason"])
    df, _ = clean(unlabelled)
    assert set(C.FEATURES) <= set(df.columns)


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
