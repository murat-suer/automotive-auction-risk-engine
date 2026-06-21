"""GEM opportunity engine — the self-initiated layer on top of the classifier.

The brief asked only for lemon classification. The GEM ("Good buy / undervalued"
opportunity Mining) engine sits on the model's ``predict_proba`` and surfaces
three tiers of low-risk buying opportunities. Its tier-1 probability threshold is
calibrated **out-of-fold** (5-fold ``cross_val_predict``) to a target precision
on the good-buy class, so no test data informs the threshold. Tier success rates
are reported with Wilson 95% confidence intervals on the hold-out set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (returns the ``(low, high)`` bounds as percentages)."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return round(max(0.0, center - margin) * 100, 1), round(min(1.0, center + margin) * 100, 1)


def calibrate_oof_threshold(model, X_train_fe: pd.DataFrame, y_train: pd.Series) -> dict:
    """Calibrate the tier-1 probability threshold out-of-fold.

    Uses 5-fold ``cross_val_predict`` probabilities to pick the smallest
    good-buy probability that still meets ``config.GEM_TARGET_PRECISION``; falls
    back to the maximum-precision operating point if the target is unreachable.
    Returns the threshold, its OOF precision/recall and the PR curve.
    """
    skf = StratifiedKFold(n_splits=config.OOF_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba_oof = cross_val_predict(model, X_train_fe, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    prob_good_oof = 1 - proba_oof

    y_good = (y_train == 0).astype(int).values
    precisions, recalls, thresholds = precision_recall_curve(y_good, prob_good_oof)
    pr_auc = sklearn_auc(recalls, precisions)

    valid = precisions[:-1] >= config.GEM_TARGET_PRECISION
    opt_idx = int(np.argmax(valid)) if valid.any() else int(np.argmax(precisions[:-1]))

    return {
        "threshold": float(thresholds[opt_idx]),
        "oof_precision": float(precisions[opt_idx]),
        "oof_recall": float(recalls[opt_idx]),
        "precisions": precisions,
        "recalls": recalls,
        "pr_auc": float(pr_auc),
    }


def gem_masks(X_test_fe: pd.DataFrame, y_proba: np.ndarray, threshold: float,
              price_q25: float, risky_cluster: str) -> dict[str, np.ndarray]:
    """Compute the three GEM tier boolean masks on the test set.

    - **GEM1 Bargain:** high-confidence, priced below the 25th percentile.
    - **GEM2 Undervalued:** priced >=10% below the current auction average.
    - **GEM3 Market Quality:** market-priced, very high confidence.

    All tiers exclude the statistically riskiest vehicle cluster.
    """
    prob_good = 1 - y_proba
    cluster = X_test_fe["Vehicle_Profile_Cluster"].values
    cost = X_test_fe["VehBCost"].values
    cvr = X_test_fe["CurrentValueRatio"].values
    not_risky = cluster != risky_cluster

    g1 = (prob_good >= threshold) & (cost <= price_q25) & not_risky
    g2 = (prob_good >= config.GEM2_MIN_CONFIDENCE) & (cvr < config.GEM2_CVR_MAX) & not_risky
    g3 = (
        (prob_good >= config.GEM3_MIN_CONFIDENCE)
        & (cvr >= config.GEM3_CVR_MIN)
        & (cvr <= config.GEM3_CVR_MAX)
        & (cost > price_q25)
        & not_risky
    )
    return {"gem1": g1, "gem2": g2, "gem3": g3}


def gem_backtest(mask: np.ndarray, y_test: pd.Series) -> dict:
    """Success rate (= share of genuine good buys) of a tier with Wilson 95% CI."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "success_rate": 0.0, "wilson_lo": 0.0, "wilson_hi": 0.0}
    ok = int((y_test.values[mask] == 0).sum())
    lo, hi = wilson_ci(ok, n)
    return {"n": n, "success_rate": round(ok / n * 100, 1), "wilson_lo": lo, "wilson_hi": hi}
