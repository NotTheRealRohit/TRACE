# TRACE ML Model Improvement — Implementation Plan

**Last Updated**: 2025-04-19  
**Current WD Accuracy**: 89.0%  
**Target**: 90%+  
**Branch**: `feature/ml-model-improve`

---

## Completed Work

### Phase 1: Voltage Feature ✅
- Added `voltage_scaler = StandardScaler()` to `train_and_save()`
- Added voltage to feature matrix (train + test + inference)
- Added `over_voltage` and `low_voltage` rules with 93% and 95% confidence
- Changed `run_rules()` signature to accept `voltage` parameter
- All 7 original rule lambdas updated to 3-arg signature `(fc, notes, voltage)`
- `predict()` passes voltage through to both rules and ML features
- Result: 86.4% → 88.3% (+1.9%)

### Phase 2: XGBoost ✅
- Replaced `RandomForestClassifier` with `XGBClassifier` for both FA and WD
- Added `xgboost>=2.0.0` to `requirements.txt`
- Current params: `n_estimators=1000, max_depth=8, learning_rate=0.02, min_child_weight=5, subsample=0.8, colsample_bytree=0.8`
- Result: 88.3% → 89.0% (+0.7% with tuning)

### Phase 3: Hyperparameter Tuning ✅
- Tested 15+ XGBoost configurations (depth, learning rate, subsample, colsample, regularization)
- Tested sample weighting for Customer Failure class — no improvement
- Tested TF-IDF 40→100 — no improvement
- Tested voltage_bracket OHE, voltage flags, month features — no improvement
- Tested voltage × DTC interactions — no improvement
- **89.0% is the ceiling with current feature set and XGBoost**

---

## Remaining Gap: 89% → 90%+

The bottleneck is **Customer Failure recall (83%)** — 1,227 CF samples misclassified as Production Failure.

### Plan A: Target Encoding for Categorical Features (Expected: +0.5-1%)

Customer Complaint and Supplier have high-cardinality OHE that XGBoost struggles with. Target encoding (mean-encoding) captures the direct statistical relationship between each category and the WD target.

**Implementation**:
1. Add `category_encoders` to `requirements.txt` (`pip install category-encoders`)
2. In `train_and_save()`, replace `ohe` (Customer Complaint) and `ohe_supplier` (Supplier) with `LeaveOneOutEncoder` or `TargetEncoder`
3. Use `LeaveOneOutEncoder` to avoid data leakage — it computes leave-one-out mean of the target for each category
4. Save encoders in the pickle bundle
5. In `run_ml()`, apply the same target encoding transformations

**Files to modify**:
- `backend/ml_predictor.py`: `train_and_save()`, `run_ml()`, bundle dict
- `backend/requirements.txt`: add `category-encoders`

**Key concern**: Data leakage. Target encoding MUST use leave-one-out or K-fold strategy on the training set only. Never fit on the full dataset before splitting.

```python
from category_encoders import LeaveOneOutEncoder

loo_complaint = LeaveOneOutEncoder(cols=['Customer Complaint'])
X_c_tr = loo_complaint.fit_transform(df_tr[['Customer Complaint']], pd.Series(ywd_tr))

# At inference:
X_c = bundle['loo_complaint'].transform(pd.DataFrame([[features.get('customer_complaint')]], columns=['Customer Complaint']))
```

### Plan B: Ensemble Stacking (Expected: +0.5-1%)

Stack XGBoost + LightGBM + CatBoost predictions with a logistic regression meta-learner.

**Implementation**:
1. `pip install lightgbm catboost`
2. Train 3 diverse models on the same features:
   - XGBoost (already done)
   - LightGBM (different tree growing strategy, often complementary)
   - CatBoost (native categorical handling, ordered boosting)
3. Use out-of-fold predictions as meta-features
4. Train a `LogisticRegression` meta-learner on the stacked predictions
5. Save all models and meta-learner in the bundle

**Files to modify**:
- `backend/ml_predictor.py`: Add LGBM/CatBoost imports, training, inference
- `backend/requirements.txt`: add `lightgbm`, `catboost`

```python
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Train each model, get OOF predictions, stack
meta_features_tr = np.column_stack([xgb_oof, lgbm_oof, cat_oof])
meta_clf = LogisticRegression(max_iter=1000)
meta_clf.fit(meta_features_tr, ywd_tr)
```

### Plan C: Feature Engineering — Supplier Warranty Rate (Expected: +0.5-1%)

Compute historical warranty decision rates per supplier from training data. This directly encodes which suppliers have more PF vs CF claims.

**Implementation**:
1. Compute supplier-level PF/CF/AtS rates from training split
2. Join to df_tr and df_te as new numeric features: `supplier_pf_rate`, `supplier_cf_rate`, `supplier_ats_rate`
3. Default to global mean for unseen suppliers
4. Save the rate lookup table in the bundle
5. In `run_ml()`, look up supplier rates or default to global mean

**Files to modify**:
- `backend/ml_predictor.py`: Add supplier rate computation in `train_and_save()`, add lookup in `run_ml()`

```python
# In train_and_save():
supplier_rates = df_tr.groupby('Supplier')['Warranty Decision'].value_counts(normalize=True).unstack(fill_value=0)
df_tr = df_tr.merge(supplier_rates, left_on='Supplier', right_index=True, how='left')
```

### Plan D: Additional New Features (Expected: +0.3-0.5%)

Features in the dataset that are NOT yet used in ML:

1. **Voltage bracket OHE** (already tested, no gain in isolation but may help in ensemble)
2. **Month/season** from Date — seasonal warranty patterns (already tested, minimal gain)
3. **DTC count × voltage interaction** — high DTC count + abnormal voltage = stronger signal
4. **Customer Complaint TF-IDF** — currently only DTC text gets TF-IDF, expand to complaint text
5. **Number of DTC codes × specific prefix** — e.g., `dtc_count * has_U`

### Plan E: Increase Training Data Quality (Expected: +0.5-2%)

The v9 dataset has ~93-96% pattern correlation (synthetic noise). Consider:
1. Training on v10 dataset (`synthetic_warranty_claims_v10.csv` — already exists in backend/)
2. Verify v10 has better signal-to-noise ratio
3. Check if v10 has additional columns that could be features

**Quick check**:
```bash
cd backend && python3 -c "
import pandas as pd
df = pd.read_csv('synthetic_warranty_claims_v10.csv')
print(df.columns.tolist())
print(len(df))
"
```

---

## Recommended Execution Order

| Step | Plan | Expected Gain | Cumulative |
|------|------|---------------|------------|
| 1 | Target Encoding (Plan A) | +0.5-1% | 89.5-90% |
| 2 | Ensemble Stacking (Plan B) | +0.5-1% | 90-91% |
| 3 | Supplier Rates (Plan C) | +0.5-1% | Alternative to A |
| 4 | Check v10 dataset (Plan E) | Unknown | — |

**Recommendation**: Start with Plan A (target encoding) as it's the simplest change. If 90% is reached, commit and stop. If not, proceed to Plan B.

---

## Key Files Modified (Current State)

| File | Changes |
|------|---------|
| `backend/ml_predictor.py` | +Voltage feature, +XGBoost, +voltage rules, `run_rules()` 3-arg signature |
| `backend/requirements.txt` | Added `xgboost>=2.0.0` |
| `backend/trace_models.pkl` | Retrained with XGBoost + voltage |
| `backend/tests/test_ml_predictor.py` | Expanded `matched_complaint` assertion to include all 14 known complaints |

## Key Files NOT Yet Modified (Needed for Next Steps)

| File | What to Change |
|------|----------------|
| `backend/ml_predictor.py` | Target encoding, ensemble stacking, supplier rates |
| `backend/evaluate_model.py` | Update feature names to include `Voltage`, update for XGBoost |
| `backend/requirements.txt` | Add `category-encoders`, `lightgbm`, `catboost` as needed |

---

## Important Notes for Next Session

1. **The model in `trace_models.pkl` is trained with XGBoost + voltage**. Delete it and retrain if you change the feature set (`rm backend/trace_models.pkl` then `python3 -c "from ml_predictor import train_and_save; train_and_save()"`).

2. **`run_rules()` now takes 3 arguments**: `run_rules(fault_code, notes, voltage)`. The `voltage` parameter has a default of `None` so existing code that passes 2 args won't break, but tests should be updated to pass voltage.

3. **The 89% ceiling** is caused by Customer Failure / Production Failure overlap. Feature engineering (target encoding, supplier rates) is more likely to break through than more hyperparameter tuning.

4. **All 7 original rules now take `(fc, notes, voltage)` lambda signatures**. The 2 new voltage rules use the 3rd arg; the original 7 rules ignore it.

5. **`predict()` passes voltage to both rules and ML features**, including when LLM provides features (the `voltage` key is injected after LLM feature extraction if missing).