"""
score_customers.py — give every customer a churn-risk score (command-line script)
=================================================================================

WHAT THIS SCRIPT DOES
    1. Reads a customer file (.xlsx or .csv) in the raw IBM column format.
       Label columns (Churn Value, Churn Reason, ...) are optional, so brand-new
       customers can be scored.
    2. Cleans it with exactly the same code used for training.
    3. Loads the trained model (models/churn_model.joblib — preprocessing + model).
    4. Writes one row per customer, riskiest first, with:
           churn_prob                probability of churning this quarter
           risk_decile               1 = riskiest 10 % of THIS file, 10 = safest
           expected_revenue_at_risk  churn_prob x monthly charges (rank by money, not just risk)

USAGE
    .venv\\Scripts\\python.exe score_customers.py data\\raw\\telco_customer_churn.xlsx
    .venv\\Scripts\\python.exe score_customers.py new_customers.csv --out my_scores.csv

    Run train_pipeline.py first so the model file exists.
"""
import argparse
import sys
from pathlib import Path

# Make the `churn` package in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from churn import config as C  # noqa: E402
from churn.data_cleaning import clean  # noqa: E402

DEFAULT_OUT = C.ROOT / "reports" / "customer_risk_scores.csv"
MODEL_PATH = C.MODELS_DIR / "churn_model.joblib"


def parse_args():
    parser = argparse.ArgumentParser(description="Score customers with the trained churn model.")
    parser.add_argument("input", help="customer file in the raw IBM format (.xlsx, .xls or .csv)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the scores (CSV)")
    return parser.parse_args()


def read_customers(path: Path) -> pd.DataFrame:
    """Read Excel or CSV depending on the file extension."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean new customers with the training code.

    clean() expects a 'Churn Value' column; unlabelled customers get a -1
    placeholder so the same function runs unchanged. It is never used as input.
    """
    if "Churn Value" not in raw:
        raw["Churn Value"] = -1
    df, _ = clean(raw)
    return df


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    """Churn probability, risk decile and revenue at risk for each customer."""
    out = df[[C.ID_COL, "contract", "tenure_months", "monthly_charges"]].copy()
    out["churn_prob"] = model.predict_proba(df[C.FEATURES])[:, 1]

    # Decile 1 = the riskiest 10 % of this file (relative ranking, not a fixed threshold).
    risk_rank = out["churn_prob"].rank(method="first", ascending=False)
    out["risk_decile"] = pd.qcut(risk_rank, 10, labels=range(1, 11))

    out["expected_revenue_at_risk"] = out["churn_prob"] * out["monthly_charges"]
    return out.sort_values("churn_prob", ascending=False)


def main():
    args = parse_args()
    customers = prepare(read_customers(Path(args.input)))
    model = joblib.load(MODEL_PATH)
    scores = score(customers, model)

    scores.to_csv(args.out, index=False)
    print(f"Scored {len(scores):,} customers -> {args.out}")
    print(scores.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
