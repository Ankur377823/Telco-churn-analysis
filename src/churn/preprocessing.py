"""
preprocessing.py — turn the clean table into numbers a model can learn from
===========================================================================

WHAT THIS FILE DOES
    Builds ONE sklearn preprocessing step that every model shares, and wraps
    it together with a model into a single Pipeline.

    Column group            Transform                         Why
    ----------------------  --------------------------------  -------------------------------
    total_charges           log(1 + x), then standardise      strongly right-skewed money
    other numeric columns   standardise (mean 0, sd 1)        LR / KNN / SVM need equal scales
    binary 0/1 columns      passed through unchanged          already numbers
    categorical columns     one-hot encode                    models need numbers, not text

    Output: 4 numeric + 15 binary + 10 one-hot = 29 columns.

WHY A PIPELINE MATTERS (NO DATA LEAKAGE)
    The preprocessing sits *inside* the Pipeline. During cross-validation each
    fold re-fits the scaler / encoder on that fold's training rows only, so the
    validation rows never influence the transformation. Scaling the whole
    dataset first would leak information and make the scores look better than
    they really are.

    It also means the saved model file contains preprocessing + model together:
    scoring new customers applies exactly the same transforms as training.

MAIN FUNCTIONS
    build_preprocessor()   the shared ColumnTransformer
    make_pipeline(model)   preprocessor + model in one Pipeline
    feature_names(pipe)    names of the 29 columns the model sees
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C

# Numeric columns split by treatment: skewed ones are log-transformed first.
SKEWED = ["total_charges"]
SCALED = [c for c in C.NUMERIC_COLS if c not in SKEWED]


def _log_then_scale() -> Pipeline:
    """log1p handles zeros (log(1 + 0) = 0) and pulls in the long right tail."""
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    return Pipeline([("log", log1p), ("scale", StandardScaler())])


def build_preprocessor() -> ColumnTransformer:
    """The shared preprocessing step used by every model in the zoo.

    Scaling is required for distance / linear models (LR, KNN, SVM) and does no
    harm to tree models (a split on x behaves the same as a split on scaled x),
    so a single preprocessor serves all 12 model families.

    handle_unknown="ignore": a category never seen in training (say a new
    payment method) becomes all zeros instead of crashing the model.
    """
    return ColumnTransformer(
        [
            ("log_num", _log_then_scale(), SKEWED),
            ("num", StandardScaler(), SCALED),
            ("bin", "passthrough", C.BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_COLS),
        ],
        # keep names like "contract_Two year" instead of "cat__contract_Two year"
        verbose_feature_names_out=False,
    )


def make_pipeline(model) -> Pipeline:
    """Preprocessing + model as one object: fit, predict and save them together."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def feature_names(pipe: Pipeline):
    """The column names the model actually sees, after one-hot encoding (fitted pipe only)."""
    return list(pipe.named_steps["prep"].get_feature_names_out())
