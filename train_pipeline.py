"""
train_pipeline.py — train every model and save the results (command-line script)
================================================================================

WHAT THIS SCRIPT DOES
    Runs the full modelling pipeline in one go (about 8 minutes):
        1. clean the raw data          -> data/processed/telco_clean.pkl
        2. compare, tune and choose    -> see src/churn/training.py for the six stages
        3. save models and reports     -> models/, data/processed/, reports/metrics/
        4. print the held-out test results

    Notebooks 04 and 05 read what this script saves, so run it first.

USAGE
    .venv\\Scripts\\python.exe train_pipeline.py              # default: 25 search iterations
    .venv\\Scripts\\python.exe train_pipeline.py --n-iter 50  # a wider hyperparameter search
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# ── Make the `churn` package in src/ importable without installing it ────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Silence one expected warning ─────────────────────────────────────────────
# The dummy baseline never predicts churn, so its precision is undefined and
# sklearn warns about it. The environment variable is set BEFORE sklearn is
# imported so the parallel worker processes (which inherit environment
# variables, not in-process filters) are quieter too.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")

import pandas as pd  # noqa: E402  (imports below must come after the setup above)
from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402

from churn import training  # noqa: E402
from churn.data_cleaning import build_clean  # noqa: E402

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Columns shown in the final test-set table.
REPORT_COLUMNS = ["roc_auc", "pr_auc", "brier", "log_loss", "ece", "f1", "recall", "precision"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the churn model zoo.")
    parser.add_argument("--n-iter", type=int, default=25,
                        help="random-search iterations per tuned model")
    return parser.parse_args()


def print_results(out, seconds):
    """Show the held-out test table and the final choice."""
    pd.set_option("display.width", 160)
    print("\n== Held-out test set ==")
    print(out["test_results"][REPORT_COLUMNS].round(4).to_string())
    summary = out["summary"]
    print("\nSummary:", summary["final_model"], "with", summary["calibration"], "calibration")
    print(f"Done in {seconds:.0f}s. Artefacts in models/ and reports/metrics/.")


def main():
    args = parse_args()
    start = time.time()

    # Step 1 — clean the raw data and cache it.
    df, _ = build_clean()
    print(f"Cleaned data: {df.shape}, churn rate {df['churn'].mean():.3f}\n")

    # Steps 2 + 3 — compare, tune, select, calibrate, score and save.
    out = training.run(n_iter=args.n_iter)

    # Step 4 — report.
    print_results(out, time.time() - start)


if __name__ == "__main__":
    main()
