---
date: 2026-03-08T12:00:00+00:00
researcher: opencode
git_commit: 804f139f3b03b35052d110005b42c22fcd88e1ce
branch: feature/ml-model-improve
repository: capProj-2
topic: "ML Model Performance Evaluation"
tags: [research, ml, performance-evaluation, randomforest, improvements]
status: complete
last_updated: 2026-03-08
last_updated_by: opencode
---

# TRACE ML Model Performance Evaluation

## Executive Summary

The TRACE ML models demonstrate strong baseline performance on the synthetic dataset:

- **Failure Analysis Classifier (6 classes)**: 88.0% accuracy
- **Warranty Decision Classifier (3 classes)**: 93.4% accuracy

The Warranty Decision classifier outperforms the Failure Analysis classifier significantly. Key issues identified include:
- **Sensor short due to moisture** is the weakest class (75% precision, 67% recall)
- **According to Specification** has the lowest performance among warranty decisions (85% precision)
- Some significant class confusion exists between similar failure modes

## Evaluation Methodology

### Dataset
- Total samples: 12,000
- Train/Test split: 80/20 (9,600 train, 2,400 test)
- Random state: 42

### Metrics Computed
- Accuracy (overall)
- Precision (weighted, macro)
- Recall (weighted, macro)
- F1 Score (weighted, macro)
- Confusion Matrix
- Per-class TP/FP/FN/Support

## Results

### Failure Analysis Classifier (6 classes)

| Metric | Score |
|--------|-------|
| Accuracy | 0.8800 |
| Precision (weighted) | 0.8792 |
| Precision (macro) | 0.8799 |
| Recall (weighted) | 0.8800 |
| Recall (macro) | 0.8721 |
| F1 (weighted) | 0.8790 |
| F1 (macro) | 0.8753 |

**Confusion Matrix:**
```
              Predicted
              ASIC  Con  NTF  Sen  Tra  Con
              CJ327 Dam  .    sor  ck   tlr
Actual  ASIC CJ327  397    1   3    0   46    3
        Con Dam     6   326   0    0    2    5
        NTF          2     0 509   57    4    0
        Sensor      0     0  87  173    0    0
        Track      63     3   2    0  371    0
        Controller   2     2   0    0    0  336
```

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ASIC CJ327 failure due to EOS | 0.84 | 0.88 | 0.86 | 450 |
| Connector damage | 0.98 | 0.96 | 0.97 | 339 |
| NTF | 0.85 | 0.89 | 0.87 | 572 |
| Sensor short due to moisture | 0.75 | 0.67 | 0.71 | 260 |
| Track burnt due to EOS | 0.88 | 0.85 | 0.86 | 439 |
| controller failure due to supplier production failure | 0.98 | 0.99 | 0.98 | 340 |

### Warranty Decision Classifier (3 classes)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9342 |
| Precision (weighted) | 0.9350 |
| Precision (macro) | 0.9224 |
| Recall (weighted) | 0.9342 |
| Recall (macro) | 0.9256 |
| F1 (weighted) | 0.9344 |
| F1 (macro) | 0.9238 |

**Confusion Matrix:**
```
                    Predicted
                    Acc  Cus  Pro
                    .    tom  .
                    S    ial  d
Actual According to Spec  506   57    9
      Customer Failure     87  761    0
      Production Failure    5    0  975
```

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| According to Specification | 0.85 | 0.88 | 0.86 | 572 |
| Customer Failure | 0.93 | 0.90 | 0.91 | 848 |
| Production Failure | 0.99 | 0.99 | 0.99 | 980 |

## Issues Identified

### Critical Issues
1. **Sensor short due to moisture** has lowest performance (F1=0.71, 75% precision, 67% recall) - often confused with NTF (57 misclassifications)
2. **According to Specification** class shows confusion with Customer Failure (87 misclassifications)

### Moderate Issues
1. **ASIC CJ327 failure due to EOS** confused with Track burnt due to EOS (46 misclassifications) - both are EOS-related failures
2. **NTF (No Trouble Found)** has moderate performance - inherently difficult to predict

### Minor Issues
1. Class imbalance in test set: Production Failure has 980 samples vs Sensor short has 260 samples

## Improvement Recommendations

### 1. Input Tuning Options

#### A. Feature Engineering Improvements
- **Add DTC code pattern features**: Extract more granular DTC patterns beyond just P/U/C/B prefixes (e.g., specific code families)
- **Voltage interaction features**: Create interaction terms between voltage and other features
- **Temporal features**: If date information is available, add time-based patterns

#### B. Hyperparameter Tuning
- Current: n_estimators=200, default other params
- Options to try:
  - Increase n_estimators to 300-500
  - Add max_depth constraints (currently unlimited)
  - Try min_samples_split=5, min_samples_leaf=2
  - Add class_weight='balanced' to address class imbalance

#### C. Class Imbalance Handling
- Apply class_weight='balanced' to give more weight to minority classes (Sensor short, NTF)
- Consider SMOTE for oversampling minority classes

### 2. Alternative Models to Consider

#### A. Gradient Boosting Methods (Recommended Next Step)
- **XGBoost**: Typically outperforms RandomForest on structured data, handles class imbalance well
- **LightGBM**: Faster training, good for large datasets, excellent with categorical features
- **CatBoost**: Handles categorical features natively, robust out-of-box

#### B. Neural Network Options
- **MLPClassifier**: Simple feedforward neural network, may capture complex interactions
- **TabNet**: Deep learning for tabular data, attention-based feature selection

#### C. Ensemble Methods
- Stacking: Combine RandomForest with Gradient Boosting
- Voting: Hard/soft voting across multiple model types

### 3. Deep Learning Considerations

When to consider deep learning:
- Large dataset (50K+ samples) - currently only 12K
- Complex feature interactions - current features may be sufficient
- Need for end-to-end learning - current pipeline is already effective
- Availability of GPU - not currently needed

#### Recommended Approach
1. **Immediate**: Apply class_weight='balanced' to RandomForest
2. **Short-term**: Try XGBoost/LightGBM with default parameters
3. **If needed**: Add more DTC features, then try neural networks

## Action Plan

1. **Immediate** (1-2 days):
   - Retrain with class_weight='balanced'
   - Add max_depth constraint to prevent overfitting
   - Re-evaluate to see if Sensor short class improves

2. **Short-term** (1 week):
   - Implement XGBoost alternative
   - Compare performance across models
   - Add more DTC pattern features

3. **Long-term** (1 month):
   - Explore ensemble methods
   - Consider adding real-world data for validation
   - Implement A/B testing framework for model comparison

## Appendix: Full Metrics Output

### Feature Importance Analysis

**Top 20 Features for Failure Analysis:**
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Voltage | 0.1364 |
| 2 | Customer Complaint_Engine overheating | 0.1249 |
| 3 | Customer Complaint_OBD Light ON | 0.0948 |
| 4 | dtc_count | 0.0891 |
| 5 | none | 0.0873 |
| 6 | has_P | 0.0756 |
| 7 | Customer Complaint_Brake warning light ON | 0.0752 |
| 8 | has_U | 0.0723 |
| 9 | Customer Complaint_Vehicle not starting | 0.0262 |
| 10 | Customer Complaint_Starting Problem | 0.0257 |

**Top 20 Features for Warranty Decision:**
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Voltage | 0.3680 |
| 2 | Customer Complaint_OBD Light ON | 0.1507 |
| 3 | none | 0.1200 |
| 4 | dtc_count | 0.1123 |
| 5 | has_P | 0.0390 |
| 6 | has_U | 0.0370 |
| 7 | Customer Complaint_Brake warning light ON | 0.0328 |
| 8 | has_C | 0.0163 |
| 9 | Customer Complaint_Starting Problem | 0.0156 |
| 10 | Customer Complaint_Vehicle not starting | 0.0154 |

**Key Insights:**
- **Voltage is the most important feature** for both classifiers (13.6% for FA, 36.8% for WD)
- Customer complaint categories are highly predictive, especially "Engine overheating" and "OBD Light ON"
- DTC count and presence of P/U codes are important secondary features
- The "none" DTC category (no fault code) is surprisingly important

### Cross-Validation Analysis (3-fold on 30% sample)

| Classifier | CV Accuracy | Std Dev |
|------------|-------------|---------|
| Failure Analysis | 0.8842 | +/- 0.0062 |
| Warranty Decision | 0.9264 | +/- 0.0016 |

**Key Insights:**
- CV accuracy closely matches test set accuracy, indicating stable model performance
- Low standard deviation suggests consistent predictions across folds
- No significant overfitting detected

```
============================================================
TRACE MODEL PERFORMANCE EVALUATION
============================================================

============================================================
FAILURE ANALYSIS CLASSIFIER (6 classes)
============================================================
Accuracy:           0.8800
Precision (weighted): 0.8792
Precision (macro):   0.8799
Recall (weighted):    0.8800
Recall (macro):      0.8721
F1 (weighted):        0.8790
F1 (macro):           0.8753

Confusion Matrix:
[[397   1   3   0  46   3]
 [  6 326   0   0   2   5]
 [  2   0 509  57   4   0]
 [  0   0  87 173   0   0]
 [ 63   3 0 371   0]
   2   [  2   2   0   0   0 336]]

Classification Report:
                                                        precision    recall  f1-score   support

                        ASIC CJ327 failure due to EOS       0.84      0.88      0.86       450
                                     Connector damage       0.98      0.96      0.97       339
                                                  NTF       0.85      0.89      0.87       572
                         Sensor short due to moisture       0.75      0.67      0.71       260
                               Track burnt due to EOS       0.88      0.85      0.86       439
controller failure due to supplier production failure       0.98      0.99      0.98       340

                                             accuracy                           0.88      2400
                                            macro avg       0.88      0.87      0.88      2400
                                         weighted avg       0.88      0.88      0.88      2400


============================================================
WARRANTY DECISION CLASSIFIER (3 classes)
============================================================
Accuracy:           0.9342
Precision (weighted): 0.9350
Precision (macro):   0.9224
Recall (weighted):    0.9342
Recall (macro):      0.9256
F1 (weighted):        0.9344
F1 (macro):           0.9238

Confusion Matrix:
[[506  57   9]
 [ 87 761   0]
 [  5   0 975]]

Classification Report:
                             precision    recall  f1-score   support

             According to Specification       0.85      0.88      0.86       572
                   Customer Failure       0.93      0.90      0.91       848
                 Production Failure       0.99      0.99      0.99       980

                          accuracy                           0.93      2400
                         macro avg       0.92      0.93      0.92      2400
                      weighted avg       0.93      0.93      0.93      2400


============================================================
PER-CLASS ANALYSIS
============================================================

--- Failure Analysis Classes ---
ASIC CJ327 failure due to EOS: TP=397, FP=73, FN=53, Support=450
Connector damage: TP=326, FP=6, FN=13, Support=339
NTF: TP=509, FP=92, FN=63, Support=572
Sensor short due to moisture: TP=173, FP=57, FN=87, Support=260
Track burnt due to EOS: TP=371, FP=52, FN=68, Support=439
controller failure due to supplier production failure: TP=336, FP=8, FN=4, Support=340

--- Warranty Decision Classes ---
According to Specification: TP=506, FP=92, FN=66, Support=572
Customer Failure: TP=761, FP=57, FN=87, Support=848
Production Failure: TP=975, FP=9, FN=5, Support=980
```

## References

- Original research: `thoughts/shared/research/2026-03-08-ml-model-architecture-input-tuning.md`
- ML predictor: `backend/ml_predictor.py:182-227` (train_and_save)
- ML inference: `backend/ml_predictor.py:273-328` (run_ml)
- Training dataset: `backend/synthetic_warranty_claims_v2.csv`
- Evaluation script: `backend/evaluate_model.py`
