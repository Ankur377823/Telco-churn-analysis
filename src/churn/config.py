"""
config.py — the project's single source of truth
================================================

WHAT THIS FILE DOES
    Holds every setting the rest of the code needs, in one place:
      1. Folder and file paths       (where data, models and reports live)
      2. Experiment settings         (random seed, test-set size, number of CV folds)
      3. Column names and groups     (target, ID, leakage columns, feature lists)
      4. Tenure buckets used in EDA

WHY IT EXISTS
    If a path or a column list were typed out in several files, they could
    silently drift apart. Defining everything once here means a change is made
    in one place and every notebook, script and test picks it up.

HOW OTHER FILES USE IT
    from churn import config as C
    C.RAW_PATH, C.FEATURES, C.RANDOM_STATE, ...
"""
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────────────────────────────────────
# This file lives at <project>/src/churn/config.py, so going up two parent
# folders gives the project root. Paths therefore work no matter which folder
# the code is run from (project root, notebooks/, tests/ ...).
ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "telco_customer_churn.xlsx"   # original IBM workbook
PROCESSED_DIR = ROOT / "data" / "processed"                       # generated data
CLEAN_PATH = PROCESSED_DIR / "telco_clean.pkl"                    # cleaned table
SCORED_PATH = PROCESSED_DIR / "scored_customers.pkl"              # out-of-fold churn scores
MODELS_DIR = ROOT / "models"                                      # saved models (joblib)
FIGURES_DIR = ROOT / "reports" / "figures"                        # every chart as PNG
METRICS_DIR = ROOT / "reports" / "metrics"                        # CSV / JSON results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42   # one seed everywhere, so every run gives identical results
TEST_SIZE = 0.20    # 20 % of customers are held out for the final, untouched test
CV_FOLDS = 5        # 5-fold cross-validation for comparing and tuning models


# ─────────────────────────────────────────────────────────────────────────────
# 3. Columns
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "churn"          # 1 = customer left this quarter, 0 = stayed
ID_COL = "customer_id"

# Columns that must NEVER be used as model inputs (the reason is logged during
# cleaning and a test checks none of them is in FEATURES):
#   churn_label  -> the target again, as "Yes"/"No"
#   churn_score  -> IBM's own churn model output (would leak the answer)
#   churn_reason -> only filled in after a customer has already left
#   cltv         -> opaque lifetime-value estimate, used only for sizing
LEAKAGE_COLS = ["churn_label", "churn_score", "churn_reason", "cltv"]

CONSTANT_COLS = ["count", "country", "state"]   # one value for every row: no information
REDUNDANT_COLS = ["lat_long"]                    # duplicates latitude + longitude

# Location columns. Kept in the clean table but deliberately NOT used as
# features: city/zip have hundreds of levels (overfitting risk) and a
# retention team cannot act on "lives in zip 90001".
GEO_COLS = ["city", "zip_code", "latitude", "longitude"]

# The six optional internet add-on services.
ADDON_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

# Model input groups. Each group gets its own preprocessing (see preprocessing.py).
BINARY_COLS = [                                        # already 0/1 after cleaning
    "is_male", "senior_citizen", "partner", "dependents", "phone_service",
    "multiple_lines", *ADDON_COLS, "paperless_billing", "has_internet",
]
CATEGORICAL_COLS = ["internet_service", "contract", "payment_method"]   # one-hot encoded
NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges", "n_addons"]  # scaled

# The full list of 22 model inputs, in a fixed order.
FEATURES = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tenure buckets (for readable EDA charts)
# ─────────────────────────────────────────────────────────────────────────────
# pd.cut bins are right-inclusive: (-1, 6], (6, 12], ... Starting at -1 makes
# sure brand-new customers with tenure 0 land in the first bucket.
TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create every output folder if it does not exist yet (safe to call repeatedly)."""
    for folder in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
