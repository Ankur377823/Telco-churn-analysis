"""Paths, column groups and shared constants for the churn project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "telco_customer_churn.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "telco_clean.pkl"
SCORED_PATH = PROCESSED_DIR / "scored_customers.pkl"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "churn"
ID_COL = "customer_id"

# Columns removed before modelling, with the reason recorded in the cleaning log.
LEAKAGE_COLS = ["churn_label", "churn_score", "churn_reason", "cltv"]
CONSTANT_COLS = ["count", "country", "state"]
REDUNDANT_COLS = ["lat_long"]
GEO_COLS = ["city", "zip_code", "latitude", "longitude"]

ADDON_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]
BINARY_COLS = [
    "is_male", "senior_citizen", "partner", "dependents", "phone_service",
    "multiple_lines", *ADDON_COLS, "paperless_billing", "has_internet",
]
CATEGORICAL_COLS = ["internet_service", "contract", "payment_method"]
NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges", "n_addons"]
FEATURES = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def ensure_dirs():
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
