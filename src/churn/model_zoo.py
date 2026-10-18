"""
model_zoo.py — every model we compare, their search spaces, and the ensembles
=============================================================================

WHAT THIS FILE DOES
    Defines the "model zoo": 12 classifier families, from a dumb baseline to
    gradient boosting, so we can see how much each extra bit of complexity
    actually buys. It also holds:
      * COMPLEXITY     a simplicity ranking used by the one-standard-error rule
      * SEARCH_SPACES  hyperparameter ranges for RandomizedSearchCV
      * ensembles()    soft-voting and stacking built from the tuned models

    Every model here is a *bare* estimator. training.py wraps each one in the
    shared preprocessing Pipeline (see preprocessing.make_pipeline).

THE 12 FAMILIES
    baseline       dummy (always predicts the 26.5 % base rate -> AUC 0.5)
    linear/prob.   logistic regression, Gaussian naive Bayes
    distance       k-nearest neighbours, RBF support-vector machine
    single tree    decision tree
    bagging        random forest, extra trees
    boosting       AdaBoost, HistGradientBoosting, XGBoost, LightGBM
"""
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


# ─────────────────────────────────────────────────────────────────────────────
# Simplicity ranking (lower = simpler, more interpretable, cheaper to run)
# ─────────────────────────────────────────────────────────────────────────────
# Used by training.one_se_choice: among models statistically tied with the
# best, the one with the lowest number here wins. The ranking is a judgement
# call, not a formula.
COMPLEXITY = {
    "dummy": 0,
    "logistic_regression": 1, "naive_bayes": 1,
    "decision_tree": 2,
    "knn": 3,
    "svm_rbf": 4,
    "random_forest": 5, "extra_trees": 5, "adaboost": 5,
    "hist_gb": 6, "xgboost": 6, "lightgbm": 6,
    "soft_voting": 8,
    "stacking": 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# The base models, grouped by family
# ─────────────────────────────────────────────────────────────────────────────
def _baseline():
    # Predicts the training churn rate for everyone: the floor every model must beat.
    return {"dummy": DummyClassifier(strategy="prior")}


def _linear_and_probabilistic():
    return {
        # Linear model of the log-odds; interpretable and usually well calibrated.
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000),
        # Assumes features are independent given the class; fast but over-confident.
        "naive_bayes": GaussianNB(),
    }


def _distance_and_kernel():
    return {
        # Votes of the 35 most similar customers, closer ones count more.
        "knn": KNeighborsClassifier(n_neighbors=35, weights="distance"),
        # probability=True adds an internal Platt-scaling CV, which makes SVM slow.
        "svm_rbf": SVC(C=1.0, gamma="scale", probability=True, random_state=RS),
    }


def _trees():
    return {
        # One shallow tree; min_samples_leaf stops it memorising tiny groups.
        "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=RS),
        # Bagging: many trees on bootstrap samples + random feature subsets.
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                                max_features="sqrt", n_jobs=-1, random_state=RS),
        # Like a random forest but split thresholds are random too: lower variance.
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10,
                                            n_jobs=-1, random_state=RS),
    }


def _boosting():
    return {
        # Re-weights the rows earlier stumps got wrong; tends to squash scores toward 0.5.
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, algorithm="SAMME",
                                       random_state=RS),
        # sklearn's histogram-based gradient boosting (LightGBM-style).
        "hist_gb": HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=RS),
        # Gradient boosting with shallow trees and row / column subsampling.
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 n_jobs=-1, random_state=RS),
        # Leaf-wise gradient boosting; also the model we explain with SHAP.
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=RS),
    }


def base_models() -> dict:
    """All 12 base models with sensible default-ish settings, as {name: estimator}.

    A fresh set of unfitted estimators is returned on every call.
    """
    return {
        **_baseline(),
        **_linear_and_probabilistic(),
        **_distance_and_kernel(),
        **_trees(),
        **_boosting(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search spaces (for RandomizedSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
# Keys start with "model__" because the estimator being searched is a Pipeline
# whose model step is called "model".
#
# Reading the distributions:
#   loguniform(a, b)   equally likely on a log scale (good for C, learning rates)
#   randint(a, b)      whole numbers a ... b-1
#   uniform(loc, w)    a range from loc to loc + w   (so uniform(0.6, 0.4) = 0.6 ... 1.0)
SEARCH_SPACES = {
    "logistic_regression": {
        "model__C": loguniform(1e-3, 1e2),            # inverse regularisation strength
    },
    "random_forest": {
        "model__n_estimators": randint(200, 800),
        "model__min_samples_leaf": randint(2, 40),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(100, 700),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 6),
        "model__min_child_weight": randint(1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 20),
    },
    "lightgbm": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(4, 40),
        "model__min_child_samples": randint(10, 120),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.01, 20),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Ensembles
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_MEMBERS = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def ensembles(tuned: dict) -> dict:
    """Soft-voting and stacking ensembles built from the tuned base models.

    tuned  {name: fitted-or-unfitted bare estimator} from training.tune()

    soft_voting  averages the members' predicted probabilities
    stacking     a logistic regression learns how to weight the members, using
                 their *out-of-fold* probabilities (cv=5) so it never learns
                 from over-fitted in-sample predictions. SVM joins the stack
                 for extra diversity.
    """
    members = [(name, tuned[name]) for name in ENSEMBLE_MEMBERS]
    return {
        "soft_voting": VotingClassifier(members, voting="soft", n_jobs=1),
        "stacking": StackingClassifier(
            members + [("svm_rbf", base_models()["svm_rbf"])],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            stack_method="predict_proba", cv=5, n_jobs=1),
    }
