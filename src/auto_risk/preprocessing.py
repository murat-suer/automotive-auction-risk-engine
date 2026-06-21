"""Preprocessing pipeline construction.

Numeric features are standardised, categoricals one-hot encoded, and the eight
correlated ``MMR*`` market-reference prices are compressed with PCA(2) to curb
multicollinearity. Column groups are derived from the engineered training frame.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def column_groups(X_fe: pd.DataFrame) -> dict[str, list[str]]:
    """Split engineered columns into numeric / categorical / MMR groups."""
    mmr_cols = [c for c in X_fe.columns if "MMR" in c]
    num_cols = [c for c in X_fe.select_dtypes(include="number").columns if c not in mmr_cols]
    cat_cols = [c for c in X_fe.columns if c not in num_cols and c not in mmr_cols]
    return {"num": num_cols, "cat": cat_cols, "mmr": mmr_cols}


def build_preprocessor(X_fe: pd.DataFrame) -> ColumnTransformer:
    """Build the ColumnTransformer: StandardScaler + OneHotEncoder + PCA(2) on MMR."""
    groups = column_groups(X_fe)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), groups["num"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), groups["cat"]),
            ("pca_mmr", Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2))]), groups["mmr"]),
        ]
    )
