import numpy as np
import pandas as pd
import pytest

from churn.data_cleaning import clean, load_raw


def _raw_row(cid, tenure=12, monthly=70.0, total=840.0, internet="Fiber optic", churn=0, **overrides):
    """One row in the raw IBM workbook format (original column names and Yes/No strings)."""
    addon = "No internet service" if internet == "No" else "No"
    row = {
        "CustomerID": cid, "Count": 1, "Country": "United States", "State": "California",
        "City": "Los Angeles", "Zip Code": 90001, "Lat Long": "33.9, -118.2",
        "Latitude": 33.9, "Longitude": -118.2, "Gender": "Female", "Senior Citizen": "No",
        "Partner": "No", "Dependents": "No", "Tenure Months": tenure, "Phone Service": "Yes",
        "Multiple Lines": "No", "Internet Service": internet, "Online Security": addon,
        "Online Backup": addon, "Device Protection": addon, "Tech Support": addon,
        "Streaming TV": addon, "Streaming Movies": addon, "Contract": "Month-to-month",
        "Paperless Billing": "Yes", "Payment Method": "Electronic check",
        "Monthly Charges": monthly, "Total Charges": total, "Churn Label": "Yes" if churn else "No",
        "Churn Value": churn, "Churn Score": 50, "CLTV": 4000,
        "Churn Reason": "Competitor made better offer" if churn else np.nan,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_small():
    """Tiny raw frame containing every kind of mess the cleaner must handle."""
    return pd.DataFrame([
        _raw_row("A", tenure=0, total=" ", monthly=20.0),                      # blank total charges
        _raw_row("B", internet="No", monthly=20.0, total=240.0),               # no internet
        _raw_row("C", churn=1, **{"Online Security": "Yes", "Tech Support": "Yes"}),
        _raw_row("D", tenure=60, total=4200, **{"Phone Service": "No",
                                                "Multiple Lines": "No phone service",
                                                "Contract": "Two year", "Gender": "Male"}),
    ])


@pytest.fixture(scope="session")
def real_clean():
    df, log = clean(load_raw())
    return df, log
