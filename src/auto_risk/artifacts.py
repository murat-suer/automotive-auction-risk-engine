"""JSON artifact writers.

The pipeline emits seven compact JSON files that the portfolio website renders
(metrics, model comparison, GEM candidates, feature importance, cluster
profiles, pipeline architecture and a scored prediction sample). Each builder is
a pure function returning a serialisable dict; ``save_json`` persists it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: dict, filename: str, out_dir: Path = config.ARTIFACTS_DIR) -> Path:
    """Write ``data`` to ``out_dir/filename`` (pretty-printed, UTF-8)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def build_metrics(*, df, X_train, X_train_sampled, X_test, X_train_fe, y,
                  ev, grid, pca_var, pca_cum, roc_data, minority_pct, majority_pct,
                  mmr_input_cols) -> dict:
    best_params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
    f1, rec = ev["f1"], ev["recall"]
    return {
        "model": "Random Forest (GridSearchCV optimized)",
        "dataset": {
            "total": int(df.shape[0]),
            "train": int(X_train_sampled.shape[0]),
            "test": int(X_test.shape[0]),
            "lemon_pct": minority_pct,
            "good_pct": majority_pct,
            "class_ratio": f"{int((y == 0).sum() // (y == 1).sum())}:1",
            "features_raw": int(X_train.shape[1]),
            "features_engineered": int(X_train_fe.shape[1]),
        },
        "performance": {
            "f1": round(f1, 4),
            "roc_auc": round(ev["roc_auc"], 4),
            "recall": round(rec, 4),
            "precision": round(ev["precision"], 4),
            "cv_f1_best": round(float(grid.best_score_), 4),
        },
        "best_params": best_params,
        "confusion_matrix": ev["confusion_matrix"],
        "baseline": {
            "label": "Naive baseline (no class weighting)",
            "f1": 0.00,
            "recall": 0.00,
            "note": "Ignores lemon class entirely — F1≈0",
        },
        "industry_standard": {
            "label": "Standard RF (class_weight balanced, no tuning)",
            "f1": config.REFERENCE_F1,
            "recall": config.REFERENCE_RECALL,
            "note": "Published benchmark for this dataset type",
        },
        "improvements": {
            "f1_vs_industry": round((f1 - config.REFERENCE_F1) / config.REFERENCE_F1 * 100, 1),
            "recall_pp_gain": round((rec - config.REFERENCE_RECALL) * 100, 1),
        },
        "pca": {"components": 2, "variance_explained": pca_var, "cumulative": pca_cum, "input_cols": mmr_input_cols},
        "roc_curve": roc_data,
        "generated_at": _now(),
    }


def build_model_comparison(results: dict, winner: str, selected: str = "Random Forest") -> dict:
    ordered = sorted(results.items(), key=lambda x: x[1]["cv_f1_mean"])
    return {
        "models": [
            {
                "name": name,
                "cv_f1": v["cv_f1_mean"],
                "cv_std": v["cv_f1_std"],
                "time_s": v["time_s"],
                "subsample": v["subsample"],
                "winner": name == winner,
                "selected": name == selected,
            }
            for name, v in ordered
        ],
        "winner_by_cv": winner,
        "selected_model": selected,
        "selection_rationale": [
            "Highest or near-highest CV F1 among full-dataset models",
            "Robust to outliers (tree-based, no distance assumption)",
            "Handles class imbalance via class_weight=balanced",
            "Provides feature importances for business interpretation",
            "Scales to production without architectural changes",
        ],
        "cv_folds": config.CV_FOLDS,
        "cv_metric": "F1-Score (Class 1 — lemon detection)",
        "generated_at": _now(),
    }


def build_gem_candidates(*, g1, g2, g3, calib, price_q25, baseline_sr) -> dict:
    precisions, recalls = calib["precisions"], calib["recalls"]
    pr_idx = np.linspace(0, len(recalls) - 2, min(50, len(recalls) - 1), dtype=int)
    thr = calib["threshold"]
    return {
        "gem1": {**g1, "strategy": "Bargain Buy",
                 "description": "High-confidence picks priced below 25th percentile — volume play",
                 "criteria": f"Model confidence ≥ {thr * 100:.1f}% | Price ≤ ${price_q25:,.0f} | Not risky cluster",
                 "target_buyer": "Volume Dealer"},
        "gem2": {**g2, "strategy": "Undervalued Find",
                 "description": "Priced ≥10% below current auction average — margin opportunity",
                 "criteria": f"Model confidence ≥ {config.GEM2_MIN_CONFIDENCE * 100:.0f}% | CurrentValueRatio < {config.GEM2_CVR_MAX} | Not risky cluster",
                 "target_buyer": "Value Investor"},
        "gem3": {**g3, "strategy": "Market Quality",
                 "description": "Market-priced, high-confidence vehicles — reliable resale",
                 "criteria": f"Model confidence ≥ {config.GEM3_MIN_CONFIDENCE * 100:.0f}% | Value ratio ∈ [{config.GEM3_CVR_MIN}, {config.GEM3_CVR_MAX}] | Not risky cluster",
                 "target_buyer": "Quality Dealer"},
        "baseline_success_rate": baseline_sr,
        "oof_threshold": round(thr, 4),
        "oof_precision": round(calib["oof_precision"], 4),
        "methodology": "Out-of-Fold precision calibration (5-Fold CV, zero leakage)",
        "pr_curve": {
            "recall": [round(float(x), 3) for x in recalls[pr_idx]],
            "precision": [round(float(x), 3) for x in precisions[pr_idx]],
            "auc": round(calib["pr_auc"], 4),
        },
        "generated_at": _now(),
    }


def build_feature_importance(feat_imp_df: pd.DataFrame) -> dict:
    pos = feat_imp_df[feat_imp_df["importance"] > 0].head(15)
    return {
        "method": "Permutation Importance (n_repeats=10, F1 scoring, test set)",
        "insight": "WheelType dominates: missing wheel data signals incomplete vehicle history — a strong lemon indicator",
        "features": [
            {"name": r["name"], "importance": round(float(r["importance"]), 4), "std": round(float(r["std"]), 4)}
            for _, r in pos.iterrows()
        ],
        "generated_at": _now(),
    }


def build_cluster_profiles(report: dict) -> dict:
    return {
        "k": config.KMEANS_K,
        "anova_f": round(report["f_stat"], 2),
        "anova_p": report["p_val"],
        "risky_cluster": report["risky_cluster"],
        "profiles": report["profiles"],
        "methodology": "K-Means (k=4, ANOVA-validated), risky cluster excluded from GEM",
        "generated_at": _now(),
    }


def build_pipeline_architecture(*, cleaner, groups, best_params) -> dict:
    return {
        "stages": [
            {"step": "DataCleaner", "type": "custom_transformer",
             "detail": f"Median imputation ({len(cleaner.num_medians_)} cols), Unknown fill; fit on train only"},
            {"step": "filter_outliers()", "type": "outlier_filter",
             "detail": "Remove VehBCost=0 and VehicleAge<0 (train only)"},
            {"step": "engineer_features()", "type": "feature_engineering",
             "detail": "7 new features, 6 dropped; no leakage via warranty_to_cost_median"},
            {"step": "add_vehicle_clusters()", "type": "unsupervised_layer",
             "detail": "K-Means k=4, StandardScaler, predict only on test"},
            {"step": "ColumnTransformer", "type": "preprocessor",
             "detail": f"StandardScaler({len(groups['num'])} num) + OHE({len(groups['cat'])} cat) + PCA(2) on {len(groups['mmr'])} MMR cols"},
            {"step": "RandomForest", "type": "classifier", "detail": f"GridSearchCV best: {best_params}"},
            {"step": "GEM Classifier", "type": "opportunity_layer",
             "detail": "OOF-calibrated 3-tier system on top of RF predict_proba"},
        ],
        "anti_leakage": [
            "DataCleaner.fit() only on train; .transform() on all sets",
            "engineer_features() uses training warranty median for test",
            "K-Means fitted on train, predict() on test",
            "OOF threshold via cross_val_predict (5-fold) — test set never seen",
            "GridSearchCV scoring only on train folds",
        ],
        "generated_at": _now(),
    }


def build_risk_sample(*, X_test, y_test, ev, masks, n: int = 500) -> dict:
    """Scored sample of test vehicles with risk score, prediction and GEM tier."""
    y_pred, y_proba = ev["y_pred"], ev["y_proba"]
    prob_good = 1 - y_proba
    disp = X_test.reset_index(drop=True)
    y_true = y_test.reset_index(drop=True).values
    sample_idx = disp.sample(frac=1, random_state=config.RANDOM_STATE).index[:n].tolist()

    def outcome(pred, actual):
        return {(0, 0): "TN", (1, 1): "TP", (0, 1): "FN", (1, 0): "FP"}[(pred, actual)]

    records = []
    for i in sample_idx:
        gem = "STANDARD"
        if masks["gem1"][i]:
            gem = "GEM1"
        elif masks["gem2"][i]:
            gem = "GEM2"
        elif masks["gem3"][i]:
            gem = "GEM3"
        records.append({
            "make": str(disp.at[i, "Make"]),
            "veh_year": int(disp.at[i, "VehYear"]),
            "vehicle_age": int(disp.at[i, "VehicleAge"]),
            "odo": int(disp.at[i, "VehOdo"]),
            "cost": int(disp.at[i, "VehBCost"]) if not pd.isna(disp.at[i, "VehBCost"]) else 0,
            "auction": str(disp.at[i, "Auction"]),
            "transmission": str(disp.at[i, "Transmission"]),
            "size": str(disp.at[i, "Size"]),
            "wheel_type": str(disp.at[i, "WheelType"]) if not pd.isna(disp.at[i, "WheelType"]) else "Unknown",
            "nationality": str(disp.at[i, "Nationality"]),
            "warranty_cost": int(disp.at[i, "WarrantyCost"]),
            "risk_score": round(float(y_proba[i] * 100), 1),
            "prob_good": round(float(prob_good[i] * 100), 1),
            "prediction": int(y_pred[i]),
            "actual": int(y_true[i]),
            "outcome": outcome(int(y_pred[i]), int(y_true[i])),
            "gem_type": gem,
        })
    return {"count": len(records), "vehicles": records}
