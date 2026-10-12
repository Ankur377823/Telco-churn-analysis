"""Score customers with the trained churn model.

Usage:  .venv\\Scripts\\python.exe score_customers.py <input.xlsx|csv> [--out customer_risk_scores.csv]

The input must have the raw IBM Telco columns (label columns are optional).
Run train_pipeline.py first to create models/churn_model.joblib.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from churn import config as C  # noqa: E402
from churn.data_cleaning import clean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=str(C.ROOT / "reports" / "customer_risk_scores.csv"))
    args = ap.parse_args()

    path = Path(args.input)
    raw = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    if "Churn Value" not in raw:
        raw["Churn Value"] = -1  # placeholder so the shared cleaning code runs unchanged
    df, _ = clean(raw)

    model = joblib.load(C.MODELS_DIR / "churn_model.joblib")
    out = df[[C.ID_COL, "contract", "tenure_months", "monthly_charges"]].copy()
    out["churn_prob"] = model.predict_proba(df[C.FEATURES])[:, 1]
    out["risk_decile"] = pd.qcut(out["churn_prob"].rank(method="first", ascending=False), 10,
                                 labels=range(1, 11))
    out["expected_revenue_at_risk"] = out["churn_prob"] * out["monthly_charges"]
    out.sort_values("churn_prob", ascending=False).to_csv(args.out, index=False)
    print(f"Scored {len(out):,} customers -> {args.out}")
    print(out.sort_values("churn_prob", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
