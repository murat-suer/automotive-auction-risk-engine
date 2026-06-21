"""Tests for feature engineering and vehicle-profile clustering."""

import numpy as np
import pandas as pd

from auto_risk.features import add_vehicle_clusters, cluster_report, engineer_features


def _raw(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "PurchDate": pd.to_datetime(rng.integers(1_200_000_000, 1_300_000_000, n), unit="s"),
            "VehicleAge": rng.integers(1, 9, n),
            "VehOdo": rng.integers(20_000, 120_000, n),
            "VehBCost": rng.integers(3_000, 15_000, n).astype(float),
            "WarrantyCost": rng.integers(400, 2_000, n).astype(float),
            "MMRAcquisitionAuctionAveragePrice": rng.integers(3_000, 15_000, n).astype(float),
            "MMRCurrentAuctionAveragePrice": rng.integers(3_000, 15_000, n).astype(float),
            "MMRCurrentRetailAveragePrice": rng.integers(5_000, 20_000, n).astype(float),
            "Model": ["X"] * n,  # high-cardinality column that must be dropped
            "Make": rng.choice(["FORD", "GMC", "KIA"], n),
        }
    )


def test_engineered_and_dropped_columns():
    out = engineer_features(_raw())
    for col in ("PurchMonth", "PurchWeekday", "PriceDeviation", "OdoPerYear",
                "RetailPotential", "CurrentValueRatio", "WarrantyToCostRatio"):
        assert col in out.columns
    # identifier / high-cardinality columns are gone.
    assert "PurchDate" not in out.columns
    assert "Model" not in out.columns


def test_warranty_median_is_reused_for_test_split():
    """Passing the train median must drive the test imputation deterministically."""
    train = engineer_features(_raw(seed=1))
    train_median = float(train["WarrantyToCostRatio"].median())

    test_raw = _raw(n=10, seed=2)
    test_raw.loc[0, "VehBCost"] = 0  # forces a NaN ratio that gets imputed
    test = engineer_features(test_raw, warranty_to_cost_median=train_median)
    assert test["WarrantyToCostRatio"].notna().all()
    # the imputed row carries the supplied train median (clipped to <=2).
    assert test["WarrantyToCostRatio"].iloc[0] == min(train_median, 2)


def test_clusters_added_to_both_splits_without_leakage():
    train = engineer_features(_raw(seed=3))
    test = engineer_features(_raw(n=12, seed=4))
    y = pd.Series(np.r_[np.zeros(30), np.ones(10)].astype(int))

    train_c, test_c = add_vehicle_clusters(train, test)
    assert "Vehicle_Profile_Cluster" in train_c.columns
    assert "Vehicle_Profile_Cluster" in test_c.columns

    report = cluster_report(train_c, y)
    assert 0 <= len(report["profiles"]) <= 4
    assert report["risky_cluster"] in {p["cluster"] for p in report["profiles"]}
    assert report["price_q25"] > 0
