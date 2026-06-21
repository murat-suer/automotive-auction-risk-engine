"""Hold-out evaluation: metrics, ROC curve, PCA variance and feature importance.

All evaluation happens on the untouched test split. Permutation importance is
computed with F1 scoring so it reflects the imbalanced objective the model is
actually optimised for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def evaluate(model, X_test_fe: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score the fitted model on the test split.

    Returns the headline metrics, the confusion-matrix cells and the raw
    predictions/probabilities (reused by the ROC curve and the GEM backtest).
    """
    y_pred = model.predict(X_test_fe)
    y_proba = model.predict_proba(X_test_fe)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def roc_points(y_test: pd.Series, y_proba: np.ndarray, n_points: int = config.ROC_CURVE_POINTS) -> dict:
    """Down-sample the ROC curve to ``n_points`` for compact plotting."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    idx = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr)), dtype=int)
    return {
        "fpr": [round(float(x), 4) for x in fpr[idx]],
        "tpr": [round(float(x), 4) for x in tpr[idx]],
        "auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def pca_variance(model) -> tuple[list[float], float]:
    """Explained-variance ratio of the MMR PCA(2) branch and its cumulative sum."""
    pca = model.named_steps["preprocessor"].transformers_[2][1].named_steps["pca"]
    variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    return variance, round(float(sum(variance)), 4)


def permutation_importance_df(model, X_test_fe: pd.DataFrame, y_test: pd.Series,
                              n_repeats: int = config.PERMUTATION_REPEATS) -> pd.DataFrame:
    """Permutation importance (F1 scoring) sorted descending."""
    perm = permutation_importance(
        model, X_test_fe, y_test,
        n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=-1, scoring="f1",
    )
    return (
        pd.DataFrame(
            {"name": X_test_fe.columns.tolist(), "importance": perm.importances_mean, "std": perm.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
