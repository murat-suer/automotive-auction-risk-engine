# Methodology

This document records the modelling decisions behind the engine and the explicit
guarantees that keep it leakage-free, following a standard end-to-end
data-science workflow (framing → metric → split → cleaning → features →
modelling → evaluation).

## 1. Problem framing

A US used-car dealer buys vehicles at online auctions to resell them. Some are
"lemons" (German *Montagsauto*) — serious hidden defects that make resale a loss
once storage, repair and refund costs are added. The task is **binary
classification**: predict `IsBadBuy` (1 = lemon) from auction-time fields, as a
decision-support filter for buyers. Classes are imbalanced (~12 % lemons).

## 2. Success metric

Accuracy is useless here: always predicting "good buy" already scores ~88 %.
The objective is the **F1 of the lemon class**, with a deliberate lean toward
**recall** because the cost structure is asymmetric:

- A missed lemon (false negative) → purchase price + storage + repair + refund.
- A wrongly rejected good car (false positive) → only a missed opportunity.

So the model accepts lower precision to raise recall — a business decision, not a
metric regression. ROC-AUC is tracked as a threshold-independent sanity check.

## 3. Split

A seeded 90/10 stratified hold-out (`random_state=42`). The 10 % test split is
set aside immediately and touched exactly once, at final evaluation.

## 4. Cleaning (`auto_risk.cleaning`)

`DataCleaner` is a scikit-learn transformer:

- **Numeric:** median imputation, medians learned in `fit()` on the training
  split only and applied unchanged in `transform()` to any split.
- **Categorical:** missing values become an explicit `"Unknown"` category.

`filter_outliers()` removes physically impossible rows (`VehBCost <= 0`,
`VehicleAge < 0`) from the **training set only** — we never drop rows we are
asked to score.

## 5. Feature engineering (`auto_risk.features`)

Seven features encode domain knowledge; six identifier/high-cardinality columns
(`Model`, `Trim`, `SubModel`, `BYRNO`, `VNZIP1`, `WheelTypeID`) are dropped.

| Feature | Meaning |
|---|---|
| `PriceDeviation` | paid price ÷ acquisition auction average |
| `OdoPerYear` | odometer ÷ (age + 1) — usage intensity |
| `RetailPotential` | current retail average ÷ paid price — resale margin |
| `CurrentValueRatio` | paid price ÷ current auction average |
| `WarrantyToCostRatio` | warranty cost ÷ paid price — warranty burden |
| `PurchMonth`, `PurchWeekday` | calendar features from the purchase date |

The warranty-ratio imputation value is computed on training data and **passed
into** the test transform, so the test split contributes no statistic.

## 6. Vehicle profiling (K-Means)

K-Means (k=4, fit on train, predict on test) adds a `Vehicle_Profile_Cluster`
label. One-way **ANOVA** confirms the clusters differ significantly in lemon rate
(F ≈ 388, p ≪ 0.001). The highest-lemon-rate cluster is flagged "risky" and
excluded from every GEM tier.

## 7. Preprocessing (`auto_risk.preprocessing`)

A `ColumnTransformer`: `StandardScaler` on numeric features, `OneHotEncoder`
(`handle_unknown="ignore"`) on categoricals, and `StandardScaler → PCA(2)` on
the eight correlated `MMR*` market-reference prices (PCA captures ~96 % of their
variance while removing multicollinearity).

## 8. Model selection (`auto_risk.modeling`)

Six classifiers are benchmarked with 3-fold stratified CV on lemon-class F1
(distance/iterative models on a 30 % stratified subsample for speed):

| Model | CV F1 |
|---|---|
| k-NN | 0.09 |
| Decision Tree | 0.34 |
| AdaBoost | 0.34 |
| Logistic Regression | 0.37 |
| Neural Network | 0.40 |
| **Random Forest** | **0.42** |

RandomForest wins and is selected for its robustness, native imbalance handling
(`class_weight="balanced"`) and interpretable feature importances. It is then
GridSearch-tuned (48 combinations × 3 folds = 144 fits).

## 9. The GEM opportunity engine (`auto_risk.gem`)

Beyond classification, GEM mines low-risk opportunities from `prob_good = 1 - p`:

- **Threshold calibration is out-of-fold.** The tier-1 probability threshold is
  picked from 5-fold `cross_val_predict` probabilities targeting ~95 % precision
  on the good-buy class. The hold-out is never used for calibration.
- **Three tiers**, each excluding the risky cluster: **Bargain** (cheap +
  high-confidence), **Undervalued** (priced below current auction average),
  **Market Quality** (market-priced + very high confidence).
- **Wilson 95 % CIs.** Each tier's hold-out success rate is reported with a
  Wilson score interval, so claims carry their uncertainty.

## 10. Anti-leakage guarantees (summary)

- `DataCleaner.fit()` on train only; `.transform()` on all splits.
- `engineer_features()` uses the **training** warranty median for the test split.
- K-Means fit on train, `predict()` on test.
- GEM threshold via `cross_val_predict` (5-fold) — the test split is never seen.
- `GridSearchCV` scores only on train folds.
- The test split is evaluated exactly once, after everything above is frozen.
