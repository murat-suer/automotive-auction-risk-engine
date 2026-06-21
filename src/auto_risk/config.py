"""Central configuration — paths, reproducibility seed, hyperparameters and
GEM thresholds.

Every tunable constant the pipeline depends on lives here so a run is fully
described by this module plus the dataset. The values match the original
reference notebook exactly, so a re-run reproduces the committed artifacts.
"""

from __future__ import annotations

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TRAIN_CSV = DATA_DIR / "data_train.csv"

# ── Target / split ─────────────────────────────────────────────────────────
TARGET = "IsBadBuy"
TEST_SIZE = 0.10  # 90/10 stratified hold-out

# ── Cross-validation ───────────────────────────────────────────────────────
CV_FOLDS = 3   # model comparison + GridSearch
OOF_FOLDS = 5  # GEM threshold calibration (out-of-fold)

# ── Feature engineering ────────────────────────────────────────────────────
# High-cardinality / identifier columns dropped before modelling.
DROP_COLS = ["Model", "Trim", "SubModel", "BYRNO", "VNZIP1", "WheelTypeID"]
KMEANS_K = 4

# ── Model comparison: subsample fractions for distance/iterative models ─────
SUBSAMPLE_FRAC = 0.30

# ── RandomForest GridSearch grid (48 combinations × 3-fold = 144 fits) ──────
RF_PARAM_GRID = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [15, 20, 25, None],
    "classifier__min_samples_split": [5, 10, 20],
    "classifier__min_samples_leaf": [1, 2],
}

# ── GEM opportunity engine ─────────────────────────────────────────────────
# Tier 1 (Bargain) uses an OOF-calibrated probability threshold targeting this
# precision on the "good buy" class; tiers 2/3 use fixed confidence floors.
GEM_TARGET_PRECISION = 0.95
GEM2_MIN_CONFIDENCE = 0.75
GEM3_MIN_CONFIDENCE = 0.85
# CurrentValueRatio gates (paid price vs. current auction average).
GEM2_CVR_MAX = 0.90
GEM3_CVR_MIN = 0.90
GEM3_CVR_MAX = 1.10

# ── Published benchmark (standard balanced RandomForest, untuned) ──────────
REFERENCE_F1 = 0.34
REFERENCE_RECALL = 0.22
REFERENCE_PRECISION = 0.89

# ── Static artifact fields (not produced by the model itself) ──────────────
PERMUTATION_REPEATS = 10
ROC_CURVE_POINTS = 100
