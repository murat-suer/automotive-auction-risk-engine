"""Domain feature engineering and unsupervised vehicle profiling.

``engineer_features`` derives seven business-meaningful ratios/time features and
drops six high-cardinality or identifier columns. ``add_vehicle_clusters`` fits
a K-Means vehicle-profile clustering on the training set and labels both splits
(fit on train, predict on test — no leakage). ``cluster_report`` validates the
clusters with one-way ANOVA and summarises their risk profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config


def engineer_features(df_in: pd.DataFrame, warranty_to_cost_median: float | None = None) -> pd.DataFrame:
    """Add 7 engineered features and drop 6 identifier/high-cardinality columns.

    For the test split, pass the **training** ``warranty_to_cost_median`` so the
    warranty imputation does not leak test statistics.
    """
    df = df_in.copy()

    # Calendar features from the purchase date.
    if "PurchDate" in df.columns:
        df["PurchMonth"] = df["PurchDate"].dt.month
        df["PurchWeekday"] = df["PurchDate"].dt.weekday
        df = df.drop(columns=["PurchDate"])

    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    # 1. PriceDeviation — paid price vs. acquisition auction value.
    if "VehBCost" in df.columns and "MMRAcquisitionAuctionAveragePrice" in df.columns:
        safe = df["MMRAcquisitionAuctionAveragePrice"].replace(0, np.nan)
        df["PriceDeviation"] = (df["VehBCost"] / safe).fillna(1.0)

    # 2. OdoPerYear — usage intensity.
    if "VehOdo" in df.columns and "VehicleAge" in df.columns:
        df["OdoPerYear"] = df["VehOdo"] / (df["VehicleAge"] + 1)

    # 3. RetailPotential — resale-margin signal.
    if "MMRCurrentRetailAveragePrice" in df.columns and "VehBCost" in df.columns:
        safe = df["VehBCost"].replace(0, np.nan)
        df["RetailPotential"] = (df["MMRCurrentRetailAveragePrice"] / safe).fillna(1.5).clip(0, 5)

    # 4. CurrentValueRatio — paid price vs. current market.
    if "VehBCost" in df.columns and "MMRCurrentAuctionAveragePrice" in df.columns:
        safe = df["MMRCurrentAuctionAveragePrice"].replace(0, np.nan)
        df["CurrentValueRatio"] = (df["VehBCost"] / safe).fillna(1.0).clip(0, 3)

    # 5. WarrantyToCostRatio — warranty burden relative to price.
    if "WarrantyCost" in df.columns and "VehBCost" in df.columns:
        safe = df["VehBCost"].replace(0, np.nan)
        df["WarrantyToCostRatio"] = df["WarrantyCost"] / safe
        if warranty_to_cost_median is None:
            warranty_to_cost_median = float(df["WarrantyToCostRatio"].median())
        df["WarrantyToCostRatio"] = df["WarrantyToCostRatio"].fillna(warranty_to_cost_median).clip(0, 2)

    return df


def add_vehicle_clusters(X_train_fe: pd.DataFrame, X_test_fe: pd.DataFrame):
    """Fit K-Means (k=``config.KMEANS_K``) on the train numeric features and add
    a ``Vehicle_Profile_Cluster`` label to both splits.

    Returns ``(X_train_fe, X_test_fe)`` with the new column. K-Means is fit on
    the training set only and merely predicts on the test set.
    """
    num_cols = X_train_fe.select_dtypes(include="number").columns.tolist()
    scaler = StandardScaler()
    X_km_train = scaler.fit_transform(X_train_fe[num_cols])

    km = KMeans(n_clusters=config.KMEANS_K, random_state=config.RANDOM_STATE, n_init=10)
    km.fit(X_km_train)

    X_train_fe = X_train_fe.copy()
    X_test_fe = X_test_fe.copy()
    X_train_fe["Vehicle_Profile_Cluster"] = km.predict(X_km_train).astype(str)
    X_test_fe["Vehicle_Profile_Cluster"] = km.predict(scaler.transform(X_test_fe[num_cols])).astype(str)
    return X_train_fe, X_test_fe


def cluster_report(X_train_fe: pd.DataFrame, y_train: pd.Series) -> dict:
    """ANOVA-validate the clusters and summarise their risk profile.

    Returns the F-statistic / p-value, the id of the riskiest cluster (highest
    lemon rate), the training 25th-percentile price, and a per-cluster profile
    list — the inputs the GEM engine and the cluster artifact need.
    """
    df = X_train_fe.copy()
    df["IsBadBuy"] = y_train.values
    groups = [grp["IsBadBuy"].values for _, grp in df.groupby("Vehicle_Profile_Cluster")]
    f_stat, p_val = stats.f_oneway(*groups)

    risk_rates = df.groupby("Vehicle_Profile_Cluster")["IsBadBuy"].mean()
    risky_cluster = str(risk_rates.idxmax())
    price_q25 = float(X_train_fe["VehBCost"].quantile(0.25))

    profiles = []
    for c in sorted(df["Vehicle_Profile_Cluster"].unique()):
        sub = df[df["Vehicle_Profile_Cluster"] == c]
        profiles.append(
            {
                "cluster": c,
                "n": int(len(sub)),
                "pct": round(len(sub) / len(df) * 100, 1),
                "lemon_rate": round(float(sub["IsBadBuy"].mean() * 100), 1),
                "avg_age": round(float(sub["VehicleAge"].mean()), 1),
                "avg_cost": round(float(sub["VehBCost"].mean()), 0),
                "is_risky": c == risky_cluster,
            }
        )

    return {
        "f_stat": float(f_stat),
        "p_val": float(p_val),
        "risky_cluster": risky_cluster,
        "price_q25": price_q25,
        "profiles": profiles,
    }
