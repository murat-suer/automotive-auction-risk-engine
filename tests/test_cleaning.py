"""Tests for the leakage-safe cleaning layer."""

import numpy as np
import pandas as pd

from auto_risk.cleaning import DataCleaner, filter_outliers


def _frame(price, age, color):
    return pd.DataFrame({"VehBCost": price, "VehicleAge": age, "Color": color})


def test_medians_are_learned_on_train_only():
    """transform() must impute with the TRAINING median, not the test one."""
    train = _frame([10.0, 20.0, 30.0, np.nan], [1, 2, 3, 4], ["RED"] * 4)
    test = _frame([np.nan], [5], [None])

    cleaner = DataCleaner().fit(train)
    out = cleaner.transform(test)

    # train median of [10,20,30] is 20 — the test row inherits it.
    assert out["VehBCost"].iloc[0] == 20.0
    # categorical NaN becomes the explicit "Unknown" sentinel.
    assert out["Color"].iloc[0] == "Unknown"


def test_no_residual_nans_after_transform():
    train = _frame([10.0, np.nan, 30.0], [1, 2, 3], ["RED", None, "BLUE"])
    cleaner = DataCleaner().fit(train)
    out = cleaner.transform(train)
    assert out.isnull().sum().sum() == 0


def test_filter_outliers_drops_impossible_rows_and_aligns_target():
    X = _frame([100.0, 0.0, 200.0, 300.0], [2, 3, -1, 4], ["A", "B", "C", "D"])
    y = pd.Series([0, 1, 1, 0])
    X_f, y_f = filter_outliers(X, y)
    # rows with VehBCost==0 and VehicleAge==-1 are removed.
    assert len(X_f) == 2
    assert list(X_f["VehBCost"]) == [100.0, 300.0]
    assert list(y_f) == [0, 0]
