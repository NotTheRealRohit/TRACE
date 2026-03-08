---
date: 2026-03-08T12:00:00+00:00
researcher: opencode
git_commit: 804f139f3b03b35052d110005b42c22fcd88e1ce
branch: feature/ml-model-improve
repository: capProj-2
topic: "ML Model Architecture and Input Tuning in TRACE"
tags: [research, ml, randomforest, feature-engineering, warranty-claims]
status: complete
last_updated: 2026-03-08
last_updated_by: opencode
---

# Research: ML Model Architecture and Input Tuning in TRACE

**Date**: 2026-03-08T12:00:00+00:00
**Researcher**: opencode
**Git Commit**: 804f139f3b03b35052d110005b42c22fcd88e1ce
**Branch**: feature/ml-model-improve
**Repository**: capProj-2

## Research Question

Which ML model are we currently using in `backend/ml_predictor.py`? How is its input tuned?

## Summary

The TRACE system uses **RandomForestClassifier** from scikit-learn as its ML engine. It's a hybrid system where two separate RandomForest models work in tandem—one for Failure Analysis (root cause classification) and one for Warranty Decision prediction. Input features are carefully engineered from raw claim data (DTC codes, technician notes, voltage readings) through a multi-stage transformation pipeline using OneHotEncoder, TfidfVectorizer, and StandardScaler. The final decision blends rule-based logic with ML predictions using weighted scoring.

## Detailed Findings

### ML Model Architecture

**Model Type**: `RandomForestClassifier` from scikit-learn (`backend/ml_predictor.py:24`)

**Two Parallel Models**:
- `clf_fa` - Predicts **Failure Analysis** (root cause classification)
- `clf_wd` - Predicts **Warranty Decision** (Production Failure / Customer Failure / According to Specification)

**Model Configuration** (`backend/ml_predictor.py:212-215`):
```python
clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_wd = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
```
- `n_estimators=200`: 200 decision trees in each forest
- `random_state=42`: Reproducible results
- `n_jobs=-1`: Use all CPU cores for parallel training

### Input Feature Engineering Pipeline

The ML input is transformed from raw claim data through 4 parallel feature extraction paths:

#### 1. Customer Complaint → OneHotEncoder
**Location**: `backend/ml_predictor.py:196, 200, 305`

```python
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_c = ohe.fit_transform(df[["Customer Complaint"]])
```

- Free-text technician notes are mapped to standardized complaint categories via `match_complaint()` function (`backend/ml_predictor.py:156-179`)
- Maps to 9 known complaint categories:
  - "Engine jerking during acceleration"
  - "Starting Problem"
  - "High fuel consumption"
  - "OBD Light ON"
  - "Vehicle not starting"
  - "Low pickup"
  - "Engine overheating"
  - "Rough idling"
  - "Brake warning light ON"

#### 2. DTC Code → TF-IDF + Binary Flags
**Location**: `backend/ml_predictor.py:141-153, 197, 201, 306`

```python
tfidf_d = TfidfVectorizer(max_features=40)
X_d = tfidf_d.fit_transform(dtc_feats["dtc_text"])
```

**Binary Features** (`backend/ml_predictor.py:146-152`):
- `dtc_count`: Number of DTC codes present
- `has_P`: Binary flag for Powertrain codes (P0xxx-PFxx)
- `has_U`: Binary flag for Network/communication codes (Uxxxx)
- `has_C`: Binary flag for Chassis codes (Cxxxx)
- `has_B`: Binary flag for Body codes (Bxxxx)

**Text Features**:
- DTC codes like "P0562, U0100" converted to 40-dimensional TF-IDF vectors

#### 3. Voltage → StandardScaler
**Location**: `backend/ml_predictor.py:198, 203, 308-310`

```python
scaler = StandardScaler()
X_v = scaler.fit_transform(df[["Voltage"]])
```

- Z-score normalization: `(value - mean) / std`
- Centers voltage around 12.5V (typical automotive system voltage)

#### 4. Feature Combination
**Location**: `backend/ml_predictor.py:206, 311`

```python
X = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v)])
```

All features concatenated into a sparse matrix for efficient computation.

### Confidence Calculation

**Location**: `backend/ml_predictor.py:320`

```python
ml_confidence = round(min(98.0, max(50.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)
```

- **Geometric mean** of both classifiers' probabilities
- Capped between 50-98% (minimum 50% confidence even for uncertain predictions, maximum 98% to leave room for rules)

### Hybrid Decision Blending

**Location**: `backend/ml_predictor.py:343-449`

The system blends rule-based and ML results:

| Scenario | Rule Weight | ML Weight | Bonus |
|----------|-------------|-----------|-------|
| Rule agrees with ML | 0.7 | 0.3 | +5.0 |
| Rule disagrees with ML | 0.6 | 0.1 | None |

**Confidence Thresholds**:
- `CONFIDENCE_THRESHOLD_FIRM = 85.0` - High confidence decisions
- `CONFIDENCE_THRESHOLD_MANUAL = 65.0` - Requires manual review below this

**Decision Engine Labels**:
- `"LLM+Rule+ML"` - All three components used
- `"Rule+ML"` - Rules fired without LLM
- `"ML"` - Only ML prediction used

## Code References

- `backend/ml_predictor.py:24` - RandomForestClassifier import
- `backend/ml_predictor.py:141-153` - `extract_dtc_features()` function
- `backend/ml_predictor.py:156-179` - `match_complaint()` function
- `backend/ml_predictor.py:182-227` - `train_and_save()` function
- `backend/ml_predictor.py:273-328` - `run_ml()` function
- `backend/ml_predictor.py:331-340` - Confidence thresholds and blending weights
- `backend/ml_predictor.py:343-449` - `combine_scores()` function

## Architecture Insights

1. **Hybrid Approach**: The system prioritizes rule-based decisions (domain knowledge) over ML when rules fire, only using ML as a soft signal when no rules match

2. **Feature Engineering Focus**: Significant effort goes into transforming raw inputs (especially DTC codes and free-text notes) into structured features

3. **Synthetic Dataset Note**: The training dataset (`synthetic_warranty_claims_v2.csv`) is synthetically balanced with near-random feature-target correlations by design (each class has equal representation across feature combinations)

4. **Fallback Strategy**: When LLM features are unavailable, the system falls back to rule-based feature extraction (`match_complaint()` + `extract_dtc_features()`)

## Related Research

- `thoughts/shared/research/2026-03-08-llm-integration-architecture.md` - LLM integration in the prediction pipeline

## Open Questions

- Could the model benefit from additional DTC code features (specific code patterns beyond prefix)?
- Is the 50% minimum confidence floor appropriate, or should it be lower for edge cases?
- How does the model perform on real-world data vs. the synthetic training set?
