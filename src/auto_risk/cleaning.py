"""Leakage-safe data cleaning.

``DataCleaner`` is a scikit-learn transformer: it learns imputation statistics
(numeric medians) on the training split only and applies them unchanged to any
later split, so no test information leaks into preprocessing. ``filter_outliers``
removes physically impossible rows and is applied to the **training set only**
(we never drop rows we are asked to score).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """Median-impute numeric columns and fill categoricals with ``"Unknown"``.

    ``fit`` stores the per-column medians from the training data; ``transform``
    applies those stored medians to any dataset — the standard scikit-learn
    contract that keeps the test split unseen during fitting.
    """

    def fit(self, X: pd.DataFrame, y=None) -> DataCleaner:
        X_temp = X.copy()
        if "PurchDate" in X_temp.columns:
            X_temp["PurchDate"] = pd.to_datetime(X_temp["PurchDate"], errors="coerce")
        self.numeric_cols_ = X_temp.select_dtypes(include="number").columns.tolist()
        # Categorical = everything that is neither numeric nor datetime. Defining
        # it by exclusion keeps the selection identical across pandas 2 (object)
        # and pandas 3 (the new str dtype).
        self.categorical_cols_ = [
            c
            for c in X_temp.columns
            if c not in self.numeric_cols_ and not pd.api.types.is_datetime64_any_dtype(X_temp[c])
        ]
        self.num_medians_ = {
            col: float(X_temp[col].median())
            for col in self.numeric_cols_
            if X_temp[col].isnull().sum() > 0
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if "PurchDate" in X_out.columns:
            X_out["PurchDate"] = pd.to_datetime(X_out["PurchDate"], errors="coerce")
        for col in self.numeric_cols_:
            if col in self.num_medians_ and col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.num_medians_[col])
        for col in self.categorical_cols_:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("Unknown")
        return X_out


def filter_outliers(X: pd.DataFrame, y: pd.Series):
    """Drop physically impossible rows (``VehBCost <= 0`` or ``VehicleAge < 0``).

    Training-set only. Returns the filtered ``(X, y)`` aligned on the same mask.
    """
    mask = (X["VehBCost"] > 0) & (X["VehicleAge"] >= 0)
    return X[mask].copy(), y[mask].copy()
