"""Model comparison and RandomForest hyperparameter tuning.

``compare_models`` benchmarks six classifiers with 3-fold stratified CV on the
F1 of the lemon class (distance/iterative models run on a stratified subsample
for speed). ``tune_random_forest`` then GridSearch-tunes the RandomForest — the
model selected for production for its robustness, imbalance handling and
interpretable feature importances.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config

RS = config.RANDOM_STATE


def model_zoo():
    """Return the candidate models as ``(name, estimator, subsample_frac, note)``."""
    return [
        ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RS), None, ""),
        ("Decision Tree", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RS), None, ""),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=RS, n_jobs=-1), None, ""),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RS), None, ""),
        ("Neural Network", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=RS, early_stopping=True), config.SUBSAMPLE_FRAC, "30% subsample"),
        ("k-NN", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1), config.SUBSAMPLE_FRAC, "30% subsample"),
    ]


def _subsample(X, y, frac):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - frac, random_state=RS)
    idx = next(sss.split(X, y))[0]
    return X.iloc[idx], y.iloc[idx]


def compare_models(X_train_fe: pd.DataFrame, y_train: pd.Series, preprocessor):
    """3-fold stratified CV (F1) over the model zoo.

    Returns ``(results, winner_name)`` where ``results`` maps model name to its
    mean/std CV F1, wall-clock time and subsample fraction.
    """
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)
    results = {}
    for name, clf, frac, _note in model_zoo():
        Xs, ys = (_subsample(X_train_fe, y_train, frac) if frac else (X_train_fe, y_train))
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        t0 = time.time()
        scores = cross_val_score(pipe, Xs, ys, cv=skf, scoring="f1", n_jobs=-1)
        results[name] = {
            "cv_f1_mean": round(float(scores.mean()), 4),
            "cv_f1_std": round(float(scores.std()), 4),
            "time_s": round(time.time() - t0, 1),
            "subsample": f"{int(frac * 100)}%" if frac else "100%",
        }
    winner = max(results, key=lambda n: results[n]["cv_f1_mean"])
    return results, winner


def tune_random_forest(X_train_fe: pd.DataFrame, y_train: pd.Series, preprocessor):
    """GridSearch-tune the RandomForest (config.RF_PARAM_GRID, 3-fold F1).

    Returns the fitted ``GridSearchCV`` (``best_estimator_``, ``best_params_``,
    ``best_score_`` available on the result).
    """
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=-1)),
        ]
    )
    grid = GridSearchCV(pipe, config.RF_PARAM_GRID, scoring="f1", cv=skf, n_jobs=-1, refit=True)
    grid.fit(X_train_fe, y_train)
    return grid
