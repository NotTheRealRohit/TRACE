# TRACE ML Model Performance Evaluation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive ML model performance evaluation that tests accuracy, precision, recall, F1, confusion matrices, and per-class metrics. Document results in a markdown file with improvement recommendations.

**Architecture:** Build an evaluation script that loads the trained models, runs predictions on test data, computes comprehensive metrics, and outputs a detailed performance report.

**Tech Stack:** Python, scikit-learn, pandas, numpy

---

## Context

Based on research (`thoughts/shared/research/2026-03-08-ml-model-architecture-input-tuning.md`):

- **Current Model**: RandomForestClassifier (n_estimators=200)
- **Two Classifiers**: 
  - `clf_fa` for Failure Analysis (6 classes)
  - `clf_wd` for Warranty Decision (3 classes)
- **Training Data**: 12,000 rows, 9 customer complaint types
- **Currently Computed**: Only accuracy during training (lines 217-220 in ml_predictor.py)
- **Missing Metrics**: Precision, Recall, F1, Confusion Matrix, Classification Report, per-class analysis

---

## Phase 1: Create Comprehensive Evaluation Script

### Overview
Create a standalone evaluation script that computes all relevant metrics for both classifiers.

### Files
- Create: `backend/evaluate_model.py`

### Step 1: Write the evaluation script structure

Create `backend/evaluate_model.py` with:

```python
"""
TRACE Model Performance Evaluation Script
==========================================
Computes comprehensive metrics for both Failure Analysis and Warranty Decision classifiers.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictor import (
    extract_dtc_features, match_complaint,
    DATA_PATH, MODEL_PATH
)

def load_data():
    """Load and preprocess the training dataset."""
    df = pd.read_csv(DATA_PATH)
    
    # Extract DTC features
    dtc_feats = extract_dtc_features(df["DTC"].fillna(""))
    
    # Match complaints to categories
    df["matched_complaint"] = df["Customer Complaint"].apply(match_complaint)
    
    return df, dtc_feats

def evaluate_classifier(clf, X, y_true, le, label):
    """Compute comprehensive metrics for a classifier."""
    y_pred = clf.predict(X)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_names = le.classes_
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    return metrics, cm, report, y_pred

def main():
    print("=" * 60)
    print("TRACE MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Load models
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    
    clf_fa = bundle["clf_fa"]
    clf_wd = bundle["clf_wd"]
    le_fa = bundle["le_fa"]
    le_wd = bundle["le_wd"]
    ohe = bundle["ohe"]
    tfidf_d = bundle["tfidf_d"]
    scaler = bundle["scaler"]
    
    # Load data
    df, dtc_feats = load_data()
    
    # Prepare features (same as ml_predictor.py)
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import hstack, csr_matrix
    
    X_c = ohe.transform(df[["matched_complaint"]])
    X_d = tfidf_d.transform(dtc_feats["dtc_text"])
    X_n = dtc_feats[["dtc_count", "has_P", "has_U", "has_C", "has_B"]].values
    X_v = scaler.transform(df[["Voltage"]])
    X = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v)])
    
    # Encode labels
    y_fa = le_fa.transform(df["Failure Analysis"])
    y_wd = le_wd.transform(df["Warranty Decision"])
    
    # Use same split as training for fair comparison
    X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
        X, y_fa, y_wd, test_size=0.2, random_state=42
    )
    
    # Evaluate Failure Analysis
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS CLASSIFIER (6 classes)")
    print("=" * 60)
    
    fa_metrics, fa_cm, fa_report, fa_pred = evaluate_classifier(
        clf_fa, X_te, yfa_te, le_fa, "Failure Analysis"
    )
    
    print(f"Accuracy:           {fa_metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {fa_metrics['precision_weighted']:.4f}")
    print(f"Precision (macro):   {fa_metrics['precision_macro']:.4f}")
    print(f"Recall (weighted):    {fa_metrics['recall_weighted']:.4f}")
    print(f"Recall (macro):      {fa_metrics['recall_macro']:.4f}")
    print(f"F1 (weighted):        {fa_metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):           {fa_metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    print(fa_cm)
    print("\nClassification Report:")
    print(fa_report)
    
    # Evaluate Warranty Decision
    print("\n" + "=" * 60)
    print("WARRANTY DECISION CLASSIFIER (3 classes)")
    print("=" * 60)
    
    wd_metrics, wd_cm, wd_report, wd_pred = evaluate_classifier(
        clf_wd, X_te, ywd_te, le_wd, "Warranty Decision"
    )
    
    print(f"Accuracy:           {wd_metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {wd_metrics['precision_weighted']:.4f}")
    print(f"Precision (macro):   {wd_metrics['precision_macro']:.4f}")
    print(f"Recall (weighted):    {wd_metrics['recall_weighted']:.4f}")
    print(f"Recall (macro):      {wd_metrics['recall_macro']:.4f}")
    print(f"F1 (weighted):        {wd_metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):           {wd_metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    print(wd_cm)
    print("\nClassification Report:")
    print(wd_report)
    
    # Per-class analysis
    print("\n" + "=" * 60)
    print("PER-CLASS ANALYSIS")
    print("=" * 60)
    
    print("\n--- Failure Analysis Classes ---")
    for i, cls in enumerate(le_fa.classes_):
        tp = fa_cm[i, i]
        fn = fa_cm[i, :].sum() - tp
        fp = fa_cm[:, i].sum() - tp
        support = fa_cm[i, :].sum()
        print(f"{cls}: TP={tp}, FP={fp}, FN={fn}, Support={support}")
    
    print("\n--- Warranty Decision Classes ---")
    for i, cls in enumerate(le_wd.classes_):
        tp = wd_cm[i, i]
        fn = wd_cm[i, :].sum() - tp
        fp = wd_cm[:, i].sum() - tp
        support = wd_cm[i, :].sum()
        print(f"{cls}: TP={tp}, FP={fp}, FN={fn}, Support={support}")
    
    # Return metrics for documentation
    return {
        "failure_analysis": fa_metrics,
        "warranty_decision": wd_metrics,
        "fa_cm": fa_cm,
        "wd_cm": wd_cm,
        "fa_classes": le_fa.classes_.tolist(),
        "wd_classes": le_wd.classes_.tolist()
    }

if __name__ == "__main__":
    main()
```

### Step 2: Run evaluation script to verify it works

```bash
cd backend
python3 evaluate_model.py
```

Expected: Console output with all metrics

---

## Phase 2: Run Full Evaluation and Capture Results

### Overview
Run the evaluation script and capture all output for the documentation.

### Step 1: Run evaluation

```bash
cd backend
python3 evaluate_model.py > ../thoughts/shared/evaluation_output.txt 2>&1
```

### Step 2: Read and analyze the output

```bash
cat thoughts/shared/evaluation_output.txt
```

---

## Phase 3: Create Comprehensive Documentation

### Overview
Create a detailed markdown document with all findings and improvement recommendations.

### Files
- Create: `thoughts/shared/research/2026-03-08-ml-model-performance-evaluation.md`

### Step 1: Write the documentation structure

Create a comprehensive markdown file with:

```markdown
---
date: 2026-03-08T12:00:00+00:00
researcher: opencode
git_commit: [current commit]
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

[Overview of findings - overall accuracy, key issues, recommendations]

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
| Accuracy | X.XX |
| Precision (weighted) | X.XX |
| Recall (weighted) | X.XX |
| F1 (weighted) | X.XX |

### Warranty Decision Classifier (3 classes)

| Metric | Score |
|--------|-------|
| Accuracy | X.XX |
| Precision (weighted) | X.XX |
| Recall (weighted) | X.XX |
| F1 (weighted) | X.XX |

### Confusion Matrix Analysis

[Include confusion matrices and identify confused class pairs]

### Per-Class Performance

[Detailed per-class metrics and identify weakest classes]

## Issues Identified

### Critical Issues
1. [Issue description with evidence]

### Moderate Issues
1. [Issue description]

### Minor Issues
1. [Issue description]

## Improvement Recommendations

### 1. Input Tuning Options

#### A. Feature Engineering Improvements
- [Specific recommendation with expected impact]

#### B. Hyperparameter Tuning
- Current: n_estimators=200
- Options to try:
  - Increase n_estimators to 300-500
  - Add max_depth constraints
  - Try min_samples_split/min_samples_leaf tuning

#### C. Class Imbalance Handling
- [If applicable - class weights, SMOTE, etc.]

### 2. Alternative Models to Consider

#### A. Gradient Boosting Methods
- **XGBoost**: Typically outperforms RandomForest on structured data
- **LightGBM**: Faster training, good for large datasets
- **CatBoost**: Handles categorical features natively

#### B. Neural Network Options
- **MLPClassifier**: Simple feedforward neural network
- **TabNet**: Deep learning for tabular data
- **1D CNN**: For sequential DTC patterns

#### C. Ensemble Methods
- Stacking multiple models
- Voting classifiers

### 3. Deep Learning Considerations

When to consider deep learning:
- Large dataset (50K+ samples)
- Complex feature interactions
- Need for end-to-end learning
- Availability of GPU

#### Recommended Approach
1. Start with simpler improvements (input tuning)
2. Try gradient boosting (XGBoost/LightGBM)
3. Consider deep learning only if simpler methods plateau

## Action Plan

1. **Immediate** (1-2 days):
   - [Specific action]

2. **Short-term** (1 week):
   - [Specific action]

3. **Long-term** (1 month):
   - [Specific action]

## Appendix: Full Metrics Output

[Include the full evaluation output]
```

---

## Phase 4: Run Additional Analysis

### Overview
Run additional analysis to provide more insights for improvements.

### Step 1: Analyze feature importance

Add to evaluation script:

```python
# Feature importance analysis
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature names
feature_names = (
    list(ohe.get_feature_names_out(["matched_complaint"])) +
    list(tfidf_d.get_feature_names_out()) +
    ["dtc_count", "has_P", "has_U", "has_C", "has_B", "Voltage"]
)

# Failure Analysis importance
fa_importance = clf_fa.feature_importances_
fa_top = sorted(zip(feature_names, fa_importance), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 Features for Failure Analysis:")
for name, imp in fa_top:
    print(f"  {name}: {imp:.4f}")

# Warranty Decision importance
wd_importance = clf_wd.feature_importances_
wd_top = sorted(zip(feature_names, wd_importance), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 Features for Warranty Decision:")
for name, imp in wd_top:
    print(f"  {name}: {imp:.4f}")
```

### Step 2: Cross-validation analysis

Add to evaluate robustness:

```python
from sklearn.model_selection import cross_val_score

print("\n" + "=" * 60)
print("CROSS-VALIDATION ANALYSIS (5-fold)")
print("=" * 60)

# Use smaller sample for CV due to computational cost
X_sample, _, y_fa_sample, _ = train_test_split(X, y_fa, test_size=0.7, random_state=42)

fa_cv = cross_val_score(clf_fa, X_sample, y_fa_sample, cv=5, scoring='accuracy')
print(f"Failure Analysis CV Accuracy: {fa_cv.mean():.4f} (+/- {fa_cv.std()*2:.4f})")

_, _, y_wd_sample, _ = train_test_split(X, y_wd, test_size=0.7, random_state=42)
wd_cv = cross_val_score(clf_wd, X_sample, y_wd_sample, cv=5, scoring='accuracy')
print(f"Warranty Decision CV Accuracy: {wd_cv.mean():.4f} (+/- {wd_cv.std()*2:.4f})")
```

---

## Success Criteria

### Automated Verification:
- [ ] Phase 1: `python3 evaluate_model.py` runs without errors
- [ ] Phase 2: All metrics computed and captured
- [ ] Phase 3: Documentation file created at `thoughts/shared/research/2026-03-08-ml-model-performance-evaluation.md`
- [ ] Phase 4: Feature importance and cross-validation analysis completed

### Manual Verification:
- [ ] Metrics make sense (accuracy in reasonable range 0.3-0.9)
- [ ] Confusion matrices show interpretable patterns
- [ ] Recommendations are actionable

---

## Testing Strategy

### Unit Tests
- Test evaluation script runs without errors
- Test metrics are in valid ranges (0-1)
- Test confusion matrices sum to test set size

### Integration Tests
- Test evaluation script works with existing trained models
- Test output can be parsed for documentation

---

## References

- Original research: `thoughts/shared/research/2026-03-08-ml-model-architecture-input-tuning.md`
- ML predictor: `backend/ml_predictor.py:182-227` (train_and_save)
- ML inference: `backend/ml_predictor.py:273-328` (run_ml)
- Training dataset: `backend/synthetic_warranty_claims_v2.csv`
