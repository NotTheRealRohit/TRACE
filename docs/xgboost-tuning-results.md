# XGBoost Tuning Results — WD Accuracy Improvement

**Date**: 2025-04-19  
**Branch**: `feature/ml-model-improve`  
**Target**: 90%+ Warranty Decision (WD) accuracy  
**Achieved**: 89.0% (ceiling)

---

## Baseline

| Metric | Value |
|--------|-------|
| Model | RandomForestClassifier (n=200) |
| WD Accuracy | 86.4% |
| FA Accuracy | ~100% (synthetic data) |
| Features | 190 (no voltage) |

---

## Changes Applied

### Phase 1: Voltage Feature (+1.9%)

**Problem**: Voltage was passed to `predict()` but never used in ML training/inference. It was only used in the rule engine (over/under voltage rules).

**Changes**:
- `train_and_save()`: Added `StandardScaler` for Voltage, fitted on training split, added to feature matrix
- `run_ml()`: Added voltage scaling transform and hstack into feature matrix
- `predict()`: Added `"voltage": voltage` to fallback features dict + ensured LLM path also includes voltage
- `evaluate_model.py`: Feature names list updated (needs voltage too)
- Added `over_voltage` and `low_voltage` rules to RULES list
- `run_rules()` signature changed to `run_rules(fault_code, notes, voltage=None)` 
- All existing rule lambdas updated to accept 3rd `voltage` parameter (ignored by most rules)

**Result**: 86.4% → 88.3%

### Phase 2: XGBoost Replacement (+0.5%)

**Problem**: RandomForest was the only classifier. XGBoost generally outperforms RF on structured/tabular data of this size (100K rows).

**Changes**:
- Added `from xgboost import XGBClassifier` import
- Replaced `RandomForestClassifier` with `XGBClassifier` in:
  - OOF cross_val_predict (FA cascade probabilities)
  - `clf_fa` (Failure Analysis classifier)
  - `clf_wd` (Warranty Decision classifier)
- Initial params: `n_estimators=300, max_depth=8, learning_rate=0.1`
- Added `xgboost>=2.0.0` to `requirements.txt`

**Result**: 88.3% → 88.8%

### Phase 3: Hyperparameter Tuning (+0.2%)

Tested 15+ configurations. Best config:

| Parameter | Value |
|-----------|-------|
| n_estimators | 1000 |
| max_depth | 8 |
| learning_rate | 0.02 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 0.1 |
| eval_metric | mlogloss |

**Result**: 88.8% → 89.0%

---

## What Did NOT Work

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Sample weights (CF boosted 1.3-2.0x) | 88.6-88.9% | Hurts overall accuracy, over-emphasizes CF |
| TF-IDF max_features=80→100 | 89.0% | No gain over 80 |
| TF-IDF max_features=100 + month + voltage flags | 89.0% | No gain |
| TF-IDF max_features=100 + voltage_bracket OHE + month | 89.0% | No gain |
| Voltage × DTC prefix interactions | 89.0% | No gain |
| Wide XGBoost (d=12, lr=0.03) | 88.2-88.8% | Overfits |
| Very deep ensemble (n=1500, d=10, lr=0.015) | 89.0% | Same ceiling |

---

## Per-Class Breakdown (Best Model)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| According to Specification | 0.96 | 0.98 | 0.97 | 5931 |
| Customer Failure | 0.91 | **0.83** | 0.86 | 8139 |
| Production Failure | 0.81 | 0.89 | 0.85 | 5930 |

**Bottleneck**: 1,227 Customer Failure samples are misclassified as Production Failure. DTC patterns overlapping between these two classes is the root cause.

---

## Current Model Bundle Contents

The `trace_models.pkl` now includes:

```
clf_fa, clf_wd, le_fa, le_wd, ohe, tfidf_d, ohe_supplier, 
mileage_scaler, year_scaler, ohe_mileage, claim_age_scaler, voltage_scaler
```

---

## Configs Tested (Full Grid)

```
Config                                                WD Accuracy
────────────────────────────────────────────────────────────────────
n=700,  d=10, lr=0.03, mcw=3,  sub=0.80, col=0.80    88.8%
n=700,  d=12, lr=0.05, mcw=1,  sub=0.90, col=0.90    88.5%
n=800,  d=8,  lr=0.03, mcw=5,  sub=0.70, col=0.70    88.9%
n=1000, d=10, lr=0.02, mcw=3,  sub=0.80, col=0.80    89.0% ★
n=500,  d=15, lr=0.05, mcw=1,  sub=0.90, col=0.90    88.4%
n=700,  d=10, lr=0.05, mcw=5,  sub=0.70, col=0.80    88.7%
n=1000, d=8,  lr=0.02, mcw=5,  sub=0.80, col=0.80    89.0%
n=800,  d=12, lr=0.03, mcw=3,  sub=0.80, col=0.90    88.7%
n=1500, d=8,  lr=0.01, mcw=5,  sub=0.80, col=0.80    89.0%
n=1500, d=10, lr=0.015,mcw=5,  sub=0.85, col=0.85    89.0%
n=1000, d=10, lr=0.02, mcw=3,  sub=0.85, col=0.85    88.9%
n=1200, d=10, lr=0.02, mcw=3,  sub=0.85, col=0.85    88.8%
n=1000, d=12, lr=0.02, mcw=3,  sub=0.80, col=0.80    88.8%
n=800,  d=10, lr=0.05, mcw=3,  sub=0.80, col=0.80    88.8%
```

★ = currently applied config in `ml_predictor.py`