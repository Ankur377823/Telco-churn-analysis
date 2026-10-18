"""
conftest.py — shared test data (pytest "fixtures")
==================================================

WHAT THIS FILE DOES
    pytest loads this file automatically and makes its fixtures available to
    every test file. A test asks for one simply by naming it as an argument:

        def test_something(raw_small): ...

FIXTURES
    raw_small   4 hand-made customers in the RAW IBM format. Each one carries
                one specific kind of mess the cleaner must fix, so tests can
                check exact answers on a tiny table.
    real_clean  the real 7,043-customer dataset, cleaned ONCE per test session
                and shared by every test that needs it (keeps the suite fast).
"""
import numpy as np
import pandas as pd
import pytest

from churn.data_cleaning import clean, load_raw


def _raw_row(cid, tenure=12, monthly=70.0, total=840.0, internet="Fiber optic", churn=0, **overrides):
    """One customer exactly as the raw workbook stores it (original names, Yes/No strings).

    Defaults describe an ordinary fiber customer; keyword overrides inject the
    one quirk a test row is meant to carry, e.g. **{"Contract": "Two year"}.
    """
    addon = "No internet service" if internet == "No" else "No"
    row = {
        # identity and location
        "CustomerID": cid, "Count": 1, "Country": "United States", "State": "California",
        "City": "Los Angeles", "Zip Code": 90001, "Lat Long": "33.9, -118.2",
        "Latitude": 33.9, "Longitude": -118.2,
        # demographics
        "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
        # services
        "Tenure Months": tenure, "Phone Service": "Yes", "Multiple Lines": "No",
        "Internet Service": internet, "Online Security": addon, "Online Backup": addon,
        "Device Protection": addon, "Tech Support": addon, "Streaming TV": addon,
        "Streaming Movies": addon,
        # account and billing
        "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Electronic check",
        "Monthly Charges": monthly, "Total Charges": total,
        # labels (and leakage columns)
        "Churn Label": "Yes" if churn else "No", "Churn Value": churn, "Churn Score": 50, "CLTV": 4000,
        "Churn Reason": "Competitor made better offer" if churn else np.nan,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_small():
    """Tiny raw table containing every kind of mess the cleaner must handle."""
    return pd.DataFrame([
        # A: brand-new customer, Total Charges is a blank ' '
        _raw_row("A", tenure=0, total=" ", monthly=20.0),
        # B: no internet -> 'No internet service' in all six add-on columns
        _raw_row("B", internet="No", monthly=20.0, total=240.0),
        # C: a churner with two add-ons
        _raw_row("C", churn=1, **{"Online Security": "Yes", "Tech Support": "Yes"}),
        # D: no phone service -> 'No phone service'; two-year contract; male
        _raw_row("D", tenure=60, total=4200, **{"Phone Service": "No",
                                                "Multiple Lines": "No phone service",
                                                "Contract": "Two year", "Gender": "Male"}),
    ])


@pytest.fixture(scope="session")
def real_clean():
    """(clean_df, decision_log) for the real dataset, built once and shared."""
    df, log = clean(load_raw())
    return df, log
