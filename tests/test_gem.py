"""Tests for the GEM engine's statistics and tier masking."""

import numpy as np
import pandas as pd

from auto_risk import config
from auto_risk.gem import gem_backtest, gem_masks, wilson_ci


def test_wilson_ci_known_values():
    # Perfect run: interval is below 100 on the low side, capped at 100 high.
    lo, hi = wilson_ci(10, 10)
    assert 65.0 < lo < 100.0
    assert hi == 100.0
    # Symmetric-ish around 50% for a balanced count.
    lo2, hi2 = wilson_ci(50, 100)
    assert lo2 < 50.0 < hi2
    # Degenerate input.
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_with_n():
    _, hi_small = wilson_ci(9, 10)
    lo_small, _ = wilson_ci(9, 10)
    lo_big, hi_big = wilson_ci(900, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_gem_backtest_counts_good_buys():
    mask = np.array([True, True, True, False])
    y = pd.Series([0, 0, 1, 0])  # among masked: 2 good (0), 1 lemon (1)
    out = gem_backtest(mask, y)
    assert out["n"] == 3
    assert out["success_rate"] == round(2 / 3 * 100, 1)


def test_gem_masks_respect_thresholds_and_risky_cluster():
    X = pd.DataFrame(
        {
            "Vehicle_Profile_Cluster": ["0", "0", "3", "0"],  # "3" is risky → excluded
            "VehBCost": [5_000.0, 5_000.0, 5_000.0, 20_000.0],
            "CurrentValueRatio": [0.8, 0.8, 0.8, 1.0],
        }
    )
    # prob_good = 1 - y_proba → use tiny y_proba so prob_good ~1 (passes floors).
    y_proba = np.array([0.02, 0.02, 0.02, 0.02])
    masks = gem_masks(X, y_proba, threshold=0.5, price_q25=6_000.0, risky_cluster="3")

    # Row 2 is in the risky cluster → excluded from every tier.
    assert not masks["gem1"][2] and not masks["gem2"][2] and not masks["gem3"][2]
    # Rows 0/1: cheap, confident, not risky → GEM1 bargain.
    assert masks["gem1"][0] and masks["gem1"][1]
    # Row 3: above the price floor but market-priced & confident → GEM3.
    assert masks["gem3"][3]
    assert config.GEM3_CVR_MIN <= 1.0 <= config.GEM3_CVR_MAX
