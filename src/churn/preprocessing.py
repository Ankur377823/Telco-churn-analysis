"""sklearn preprocessing shared by every model, so train/serve transforms are identical."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C

SKEWED = ["total_charges"]
SCALED = [c for c in C.NUMERIC_COLS if c not in SKEWED]


def build_preprocessor() -> ColumnTransformer:
    """Log-scale the skewed money column, standardise numerics, one-hot categoricals.

    Scaling is harmless for tree models and required for LR / KNN / SVM, so one
    preprocessor serves the whole model zoo.
    """
    return ColumnTransformer(
        [
            ("log_num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                  ("scale", StandardScaler())]), SKEWED),
            ("num", StandardScaler(), SCALED),
            ("bin", "passthrough", C.BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def feature_names(pipe: Pipeline):
    return list(pipe.named_steps["prep"].get_feature_names_out())
