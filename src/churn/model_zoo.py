"""Model zoo: baselines, linear, distance/kernel, trees, boosting, and ensembles."""
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE as RS

# Rough complexity rank (lower = simpler / more interpretable). Used by the
# one-standard-error rule when choosing the final model.
COMPLEXITY = {
    "dummy": 0, "logistic_regression": 1, "naive_bayes": 1, "decision_tree": 2,
    "knn": 3, "svm_rbf": 4, "random_forest": 5, "extra_trees": 5, "adaboost": 5,
    "hist_gb": 6, "xgboost": 6, "lightgbm": 6, "soft_voting": 8, "stacking": 9,
}


def base_models() -> dict:
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=35, weights="distance"),
        "svm_rbf": SVC(C=1.0, gamma="scale", probability=True, random_state=RS),
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=RS),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                                max_features="sqrt", n_jobs=-1, random_state=RS),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10,
                                            n_jobs=-1, random_state=RS),
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, algorithm="SAMME",
                                       random_state=RS),
        "hist_gb": HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=RS),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 n_jobs=-1, random_state=RS),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=RS),
    }


# Search spaces for RandomizedSearchCV (keys are prefixed with the pipeline step).
SEARCH_SPACES = {
    "logistic_regression": {"model__C": loguniform(1e-3, 1e2)},
    "random_forest": {"model__n_estimators": randint(200, 800),
                      "model__min_samples_leaf": randint(2, 40),
                      "model__max_features": ["sqrt", 0.3, 0.5]},
    "xgboost": {"model__n_estimators": randint(100, 700),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(2, 6),
                "model__min_child_weight": randint(1, 20),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(0.1, 20)},
    "lightgbm": {"model__n_estimators": randint(100, 800),
                 "model__learning_rate": loguniform(0.01, 0.2),
                 "model__num_leaves": randint(4, 40),
                 "model__min_child_samples": randint(10, 120),
                 "model__colsample_bytree": uniform(0.5, 0.5),
                 "model__reg_lambda": loguniform(0.01, 20)},
}


def ensembles(tuned: dict) -> dict:
    """Build ensembles from tuned base estimators (bare estimators, not pipelines)."""
    members = [(k, tuned[k]) for k in ("logistic_regression", "random_forest", "xgboost", "lightgbm")]
    return {
        "soft_voting": VotingClassifier(members, voting="soft", n_jobs=1),
        "stacking": StackingClassifier(
            members + [("svm_rbf", base_models()["svm_rbf"])],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            stack_method="predict_proba", cv=5, n_jobs=1),
    }
