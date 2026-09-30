"""sklearn / LightGBM classifier factory with probability outputs aligned to a fixed
category set 0..C-1 (so attributes with categories unseen in training still get a
full-width, strictly positive distribution -- required by belief propagation)."""
from __future__ import annotations

import numpy as np

from graphrecon.attacks.base import normalize_rows

CLASSIFIERS = ("RandomForest", "LightGBM", "LogisticRegression", "MLP", "NaiveBayes", "KNN")

CLASSIFIER_DEFAULTS = {
    "RandomForest": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 1},
    "LightGBM": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 10},
    "LogisticRegression": {"C": 1.0, "max_iter": 1000},
    "MLP": {"hidden": [128, 64], "max_iter": 300, "alpha": 1e-4},
    "NaiveBayes": {"var_smoothing": 1e-9},
    "KNN": {"k": 10, "weights": "distance"},
}


def make_classifier(name: str, params: dict, seed: int):
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                      min_samples_leaf=params["min_samples_leaf"], n_jobs=8,
                                      random_state=seed)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                              num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
                              verbose=-1, n_jobs=8, random_state=seed)
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=params["max_iter"]))
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), MLPClassifier(tuple(params["hidden"]), max_iter=params["max_iter"],
                                                             alpha=params["alpha"], random_state=seed))
    if name == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=params["var_smoothing"])
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), KNeighborsClassifier(params["k"], weights=params["weights"]))
    raise ValueError(f"unknown classifier {name!r}; valid: {CLASSIFIERS}")


def fit_predict_proba(name: str, params: dict, X_train, y_train, X_test, n_classes: int,
                      seed: int, eps: float = 1e-4) -> np.ndarray:
    """Fit on (X_train, y_train) and return (len(X_test), n_classes) probabilities."""
    classes = np.unique(y_train)
    P = np.zeros((len(X_test), n_classes))
    if len(classes) == 1:
        P[:, classes[0]] = 1.0
    else:
        clf = make_classifier(name, params, seed).fit(X_train, y_train)
        P[:, clf.classes_] = clf.predict_proba(X_test)
    return normalize_rows(P, eps)
