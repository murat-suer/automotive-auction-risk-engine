"""End-to-end pipeline orchestration.

Runs the full pipeline and writes the seven JSON artifacts to
``artifacts/``:

    python scripts/run_pipeline.py

Stages: load → split → clean → outlier filter → feature engineering →
K-Means profiling → preprocessing → model comparison → RF GridSearch →
hold-out evaluation → GEM calibration + backtest → permutation importance →
artifact export. Every stage reuses the modules in ``src/auto_risk`` so the
behaviour matches the unit-tested library exactly.
"""

from __future__ import annotations

import time
import warnings

from auto_risk import artifacts, evaluation, features, gem, modeling
from auto_risk.cleaning import DataCleaner, filter_outliers
from auto_risk.data import load_data, make_split
from auto_risk.preprocessing import build_preprocessor, column_groups

warnings.filterwarnings("ignore")


def main() -> None:
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}")  # noqa: E731

    # 1. Load + split ──────────────────────────────────────────────────────
    df = load_data()
    minority_pct = round(float((df["IsBadBuy"] == 1).mean() * 100), 2)
    majority_pct = round(float((df["IsBadBuy"] == 0).mean() * 100), 2)
    y = df["IsBadBuy"]
    X_train, X_test, y_train, y_test = make_split(df)
    log(f"Loaded {df.shape[0]:,} rows | lemon rate {minority_pct}% | "
        f"train {X_train.shape[0]:,} / test {X_test.shape[0]:,}")

    # 2. Clean + outlier filter (train only) ───────────────────────────────
    cleaner = DataCleaner().fit(X_train)
    X_train_clean = cleaner.transform(X_train)
    X_test_clean = cleaner.transform(X_test)
    X_train_sampled, y_train_sampled = filter_outliers(X_train_clean, y_train)
    log(f"Cleaned | train after outlier filter: {X_train_sampled.shape[0]:,}")

    # 3. Feature engineering + K-Means profiling ───────────────────────────
    X_train_fe = features.engineer_features(X_train_sampled)
    warranty_median = float(X_train_fe["WarrantyToCostRatio"].median())
    X_test_fe = features.engineer_features(X_test_clean, warranty_to_cost_median=warranty_median)
    X_train_fe, X_test_fe = features.add_vehicle_clusters(X_train_fe, X_test_fe)
    report = features.cluster_report(X_train_fe, y_train_sampled)
    log(f"Features engineered | ANOVA F={report['f_stat']:.2f} p={report['p_val']:.2e} | "
        f"risky cluster {report['risky_cluster']}")

    # 4. Preprocessing + model comparison ──────────────────────────────────
    groups = column_groups(X_train_fe)
    preprocessor = build_preprocessor(X_train_fe)
    results, winner = modeling.compare_models(X_train_fe, y_train_sampled, preprocessor)
    log(f"Model comparison done | CV winner: {winner} ({results[winner]['cv_f1_mean']:.4f})")

    # 5. Final RandomForest + GridSearch ───────────────────────────────────
    grid = modeling.tune_random_forest(X_train_fe, y_train_sampled, build_preprocessor(X_train_fe))
    best_model = grid.best_estimator_
    best_params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
    log(f"GridSearch best CV F1 {grid.best_score_:.4f} | params {best_params}")

    # 6. Hold-out evaluation ───────────────────────────────────────────────
    ev = evaluation.evaluate(best_model, X_test_fe, y_test)
    roc_data = evaluation.roc_points(y_test, ev["y_proba"])
    pca_var, pca_cum = evaluation.pca_variance(best_model)
    log(f"TEST F1={ev['f1']:.4f} AUC={ev['roc_auc']:.4f} "
        f"Recall={ev['recall']:.4f} Precision={ev['precision']:.4f}")

    # 7. GEM calibration + backtest ────────────────────────────────────────
    calib = gem.calibrate_oof_threshold(best_model, X_train_fe, y_train_sampled)
    masks = gem.gem_masks(X_test_fe, ev["y_proba"], calib["threshold"],
                          report["price_q25"], report["risky_cluster"])
    g1 = gem.gem_backtest(masks["gem1"], y_test)
    g2 = gem.gem_backtest(masks["gem2"], y_test)
    g3 = gem.gem_backtest(masks["gem3"], y_test)
    baseline_sr = round(float((y_test == 0).mean() * 100), 1)
    log(f"GEM1 {g1['success_rate']}% (n={g1['n']}) | "
        f"GEM2 {g2['success_rate']}% (n={g2['n']}) | GEM3 {g3['success_rate']}% (n={g3['n']})")

    # 8. Permutation importance ────────────────────────────────────────────
    feat_imp = evaluation.permutation_importance_df(best_model, X_test_fe, y_test)

    # 9. Export artifacts ──────────────────────────────────────────────────
    artifacts.save_json(artifacts.build_metrics(
        df=df, X_train=X_train, X_train_sampled=X_train_sampled, X_test=X_test,
        X_train_fe=X_train_fe, y=y, ev=ev, grid=grid, pca_var=pca_var, pca_cum=pca_cum,
        roc_data=roc_data, minority_pct=minority_pct, majority_pct=majority_pct,
        mmr_input_cols=len(groups["mmr"])), "metrics.json")
    artifacts.save_json(artifacts.build_model_comparison(results, winner), "model_comparison.json")
    artifacts.save_json(artifacts.build_gem_candidates(
        g1=g1, g2=g2, g3=g3, calib=calib, price_q25=report["price_q25"],
        baseline_sr=baseline_sr), "gem_candidates.json")
    artifacts.save_json(artifacts.build_feature_importance(feat_imp), "feature_importance.json")
    artifacts.save_json(artifacts.build_cluster_profiles(report), "cluster_profiles.json")
    artifacts.save_json(artifacts.build_pipeline_architecture(
        cleaner=cleaner, groups=groups, best_params=best_params), "pipeline_architecture.json")
    artifacts.save_json(artifacts.build_risk_sample(
        X_test=X_test, y_test=y_test, ev=ev, masks=masks), "risk_predictions_sample.json")

    log(f"Done — 7 artifacts written to {artifacts.config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
