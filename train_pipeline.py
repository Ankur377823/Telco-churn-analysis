"""End-to-end churn pipeline: clean -> compare models -> tune -> ensemble -> calibrate -> score.

Usage:  .venv\\Scripts\\python.exe train_pipeline.py [--n-iter 25]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
# The dummy baseline never predicts churn, so precision is undefined for it; silence
# that warning here and in the joblib worker processes (they inherit the env var).
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
import warnings  # noqa: E402
from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import pandas as pd  # noqa: E402

from churn import training  # noqa: E402
from churn.data_cleaning import build_clean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=25, help="random-search iterations per model")
    args = ap.parse_args()

    t0 = time.time()
    df, log = build_clean()
    print(f"Cleaned data: {df.shape}, churn rate {df['churn'].mean():.3f}\n")

    out = training.run(n_iter=args.n_iter)
    pd.set_option("display.width", 160)
    cols = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "f1", "recall", "precision"]
    print("\n== Held-out test set ==")
    print(out["test_results"][cols].round(4).to_string())
    print("\nSummary:", out["summary"]["final_model"], "with", out["summary"]["calibration"],
          "calibration")
    print(f"Done in {time.time() - t0:.0f}s. Artefacts in models/ and reports/metrics/.")


if __name__ == "__main__":
    main()
