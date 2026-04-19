# Model Improvement Findings — 2026-04-19

**Date**: 2026-04-19  
**Branch**: `feature/improve-model`  
**Goal**: Push WD accuracy from 89% ceiling toward 95%+

---

## Executive Summary

The 89% accuracy ceiling is caused by **dataset noise**, not model or feature limitations. The synthetic dataset (v9) intentionally contains 4-7% noise (~93-96% correlation), making perfect classification mathematically impossible without improving the data generation logic.

### Implemented Changes

| Change | Impact |
|--------|--------|
| Tuned XGBoost params | Applied optimized config (n=1000, d=10, lr=0.02) |
| Voltage bracket feature | 7-level OHE capturing critical thresholds |
| DTC count bracket | 4-level OHE (none/single/few/many) |
| Interaction features | 4 boolean features for voltage×DTC combinations |

**Final WD Accuracy**: 88.9%

---

## Root Cause Analysis

### The Bottleneck: ASIC CJ327 Confusion

The primary accuracy ceiling comes from the **ASIC CJ327 failure due to EOS** category:

| Metric | Customer Failure | Production Failure |
|--------|------------------|-------------------|
| Count | 5,165 | 5,626 |
| Mean Voltage | 15.42V | 15.22V |
| Std Dev | 0.42V | 0.45V |
| Range | 14.04-16.50V | 13.80-16.50V |

These two classes have **statistically identical distributions** across ALL available features:
- Voltage difference: 0.2V (within noise)
- Customer distribution: Nearly identical
- Supplier distribution: Nearly identical
- Mileage distribution: Nearly identical
- DTC codes: Same codes, same patterns

### Secondary Confusion: Connector Damage

| Metric | Customer Failure | Production Failure |
|--------|------------------|-------------------|
| Count | 3,234 | 13,378 |
| Mean Voltage | 13.31V | 13.29V |

No distinguishing features for the 20% of cases that are CF vs PF.

### Clean Categories

These categories have near-perfect accuracy:
- **Track burnt due to EOS**: 95.7% CF (voltage ~17.8V)
- **NTF**: 96.6% According to Spec (voltage ~13.2V)
- **Sensor short due to moisture**: 94.9% CF (voltage ~12.7V)
- **Controller failure**: 95.4% PF (voltage ~10.4V)

---

## Dataset Analysis Findings

### Feature Distributions by Warranty Decision

| Feature | Customer Failure | Production Failure | According to Spec |
|---------|-----------------|-------------------|------------------|
| **DTC count (mean)** | 2.30 | 1.50 | 0.27 |
| **Voltage (mean)** | 15.39V | 12.86V | 13.22V |
| **Mileage (mean)** | 66,491 km | 54,261 km | 47,446 km |

### DTC Prefix Patterns

| Prefix | Customer Failure | Production Failure | According to Spec |
|--------|-----------------|-------------------|------------------|
| P | 70.7% | 16.2% | 13.1% |
| U | 37.9% | 56.5% | 5.6% |
| C | 19.6% | 80.4% | 0% |
| B | 19.3% | 80.7% | 0% |

---

## Feature Importance Analysis

### Top Features (XGBoost importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | has_P | 0.883 |
| 2 | fa_prob_NTF | 0.033 |
| 3 | has_multiple_prefixes | 0.008 |
| 4 | fa_prob_Track burnt | 0.007 |
| 5 | has_B | 0.004 |
| 6 | Voltage | 0.003 |
| 7 | voltage_bracket_very_low | 0.003 |
| 8 | fa_prob_Sensor short | 0.003 |

**Key insight**: The FA probability (cascade) is the primary signal for WD. The has_P flag dominates all other features.

---

## Implemented Improvements

### 1. Tuned XGBoost Parameters

**Before** (baseline):
```python
XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, ...)
```

**After** (tuned):
```python
_xgb_params = dict(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.02,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.1,
    eval_metric='mlogloss',
    verbosity=0,
    random_state=42
)
```

### 2. Voltage Bracket Feature

```python
def voltage_bracket(v):
    if v <= 11.0: return "very_low"
    elif v <= 13.5: return "low"
    elif v <= 14.5: return "normal"
    elif v <= 15.4: return "moderate_high"  # ASIC CJ327 threshold
    elif v <= 16.0: return "high"
    elif v <= 17.0: return "very_high"
    else: return "extreme"
```

**Rationale**: Captures the non-linear voltage thresholds that separate FA categories.

### 3. DTC Count Bracket Feature

```python
def dtc_count_bracket(c):
    if c == 0: return "none"
    elif c == 1: return "single"
    elif c <= 3: return "few"
    else: return "many"
```

**Rationale**: DTC count is highly predictive (CF avg 2.3, PF avg 1.5, AtS avg 0.27).

### 4. Interaction Features

```python
df["volt_high_and_P"] = ((df["Voltage"] > 15.4) & (dtc["has_P"] == 1)).astype(int)
df["volt_low_and_U"] = ((df["Voltage"] < 11.0) & (dtc["has_U"] == 1)).astype(int)
df["volt_normal_and_C"] = ((df["Voltage"] >= 11.0) & (df["Voltage"] <= 14.5) & (dtc["has_C"] == 1)).astype(int)
df["has_multiple_prefixes"] = ((dtc["has_P"] + dtc["has_U"] + dtc["has_C"] + dtc["has_B"]) > 1).astype(int)
```

**Rationale**: Domain-knowledge-based feature interactions.

---

## What Was Tested But Did NOT Work

| Approach | Result | Notes |
|----------|--------|-------|
| Voltage brackets alone | No improvement | has_P dominates |
| Sample weighting | 88.6-88.9% | Hurts overall accuracy |
| TF-IDF max_features=80-100 | No gain | Already saturated |
| Wide XGBoost (d=12, lr=0.03) | 88.2-88.8% | Overfits |
| Very deep ensemble | 89.0% | Same ceiling |

---

## Path to 95%+

To break through the 89% ceiling, options:

### Option 1: Improve Dataset Generation (Recommended)

The v9 dataset has artificial noise. Improve the data generation logic:
- Add more distinguishing features to ASIC CJ327 CF vs PF cases
- Add a "voltage_polarity" or "duty_cycle" feature that differentiates the 50/50 cases
- Reduce noise from 7% to 2-3%

### Option 2: Add Hard Rules

For ambiguous cases where FA = ASIC CJ327:
- If voltage > 15.4V → CF (customer over-charging)
- If voltage ≤ 15.4V → PF (production defect)
- This directly captures the domain logic

### Option 3: Ensemble Methods

- Train multiple models with different random seeds
- Use different feature subsets
- Aggregate predictions via voting

### Option 4: Improve FA→WD Cascade

The current cascade passes FA probabilities to WD. Consider:
- Multi-task learning
- Attention mechanisms between FA and WD

---

## Model Bundle Contents

The updated `trace_models.pkl` includes:

```
clf_fa, clf_wd
le_fa, le_wd
ohe, tfidf_d
ohe_supplier
mileage_scaler, year_scaler
ohe_mileage
claim_age_scaler
voltage_scaler
ohe_voltage_bracket       # NEW
ohe_dtc_count_bracket      # NEW
```

---

## Results Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| FA Accuracy | 100% | 100% | - |
| WD Accuracy | 88.8% | 88.9% | +0.1% |
| Features | ~72 | ~90 | +18 |
| Model | XGBoost baseline | XGBoost tuned | - |

The improvement is minimal because the model was already near the ceiling. The bottleneck is fundamentally in the data, not the model.

---

## References

- Dataset: `synthetic_warranty_claims_v9.csv` (100K rows)
- Tuning results: `docs/xgboost-tuning-results.md`
- Previous work: Git commit history on `feature/ml-model-improve`