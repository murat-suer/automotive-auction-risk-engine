"""Dataset loading and the stratified train/test split.

The raw file ships epoch-second purchase dates; they are parsed to datetime on
load so downstream feature engineering can derive calendar features. The split
is stratified on the (imbalanced) target and seeded for reproducibility.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(csv_path=config.TRAIN_CSV) -> pd.DataFrame:
    """Read the training CSV and parse ``PurchDate`` to datetime."""
    df = pd.read_csv(csv_path)
    if "PurchDate" in df.columns:
        df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit="s")
    return df


def make_split(df: pd.DataFrame):
    """Split into ``(X_train, X_test, y_train, y_test)``.

    90/10 stratified hold-out (``config.TEST_SIZE``) seeded with
    ``config.RANDOM_STATE``. The target column is removed from the features.
    """
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
