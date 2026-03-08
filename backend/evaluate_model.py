"""
TRACE Model Performance Evaluation Script
=========================================
Computes comprehensive metrics for both Failure Analysis and Warranty Decision classifiers.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictor import (
    extract_dtc_features, match_complaint,
    DATA_PATH, MODEL_PATH, HIGH_VALUE_DTCS
)

FEATURE_NAMES = None

def load_data(ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler):
    """Load and preprocess the training dataset."""
    global FEATURE_NAMES
    
    df = pd.read_csv(DATA_PATH)
    
    df["DTC"] = df["DTC"].fillna("").replace("none", "")
    df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
    df["Failure Analysis"] = df["Failure Analysis"].fillna("NTF")
    df["Warranty Decision"] = df["Warranty Decision"].fillna("According to Specification")
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)
    df["Supplier"] = df["Supplier"].fillna("Unknown")
    df["Mileage_km"] = pd.to_numeric(df["Mileage_km"], errors="coerce").fillna(0)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(2020)
    
    dtc_feats = pd.DataFrame(list(df["DTC"].apply(extract_dtc_features)))
    
    dtc_flag_cols = (
        ["dtc_count","has_P","has_U","has_C","has_B"] +
        [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
    )
    
    FEATURE_NAMES = (
        list(ohe.get_feature_names_out(["Customer Complaint"])) +
        list(tfidf_d.get_feature_names_out()) +
        dtc_flag_cols +
        ["Voltage"] +
        list(ohe_supplier.get_feature_names_out(["Supplier"])) +
        ["Mileage_km", "Year"]
    )
    
    return df, dtc_feats, dtc_flag_cols

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
    
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = le.classes_
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    return metrics, cm, report, y_pred

def main():
    print("=" * 60)
    print("TRACE MODEL PERFORMANCE EVALUATION")
    print("=" * 60)
    
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    
    clf_fa = bundle["clf_fa"]
    clf_wd = bundle["clf_wd"]
    le_fa = bundle["le_fa"]
    le_wd = bundle["le_wd"]
    ohe = bundle["ohe"]
    tfidf_d = bundle["tfidf_d"]
    scaler = bundle["scaler"]
    ohe_supplier = bundle["ohe_supplier"]
    mileage_scaler = bundle["mileage_scaler"]
    year_scaler = bundle["year_scaler"]
    
    df, dtc_feats, dtc_flag_cols = load_data(ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler)
    
    from scipy.sparse import hstack, csr_matrix
    
    X_c = ohe.transform(df[["Customer Complaint"]])
    X_d = tfidf_d.transform(dtc_feats["dtc_text"])
    X_n = dtc_feats[dtc_flag_cols].values
    X_v = scaler.transform(df[["Voltage"]])
    X_s = ohe_supplier.transform(df[["Supplier"]])
    X_m = mileage_scaler.transform(df[["Mileage_km"]])
    X_y = year_scaler.transform(df[["Year"]])
    X = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v),
                X_s, csr_matrix(X_m), csr_matrix(X_y)])
    
    y_fa = le_fa.transform(df["Failure Analysis"])
    y_wd = le_wd.transform(df["Warranty Decision"])
    
    X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
        X, y_fa, y_wd, test_size=0.2, random_state=42
    )
    
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
    
    fa_probs_te = clf_fa.predict_proba(X_te)
    X_wd_te = hstack([X_te, csr_matrix(fa_probs_te)])
    
    print("\n" + "=" * 60)
    print("WARRANTY DECISION CLASSIFIER (3 classes)")
    print("=" * 60)
    
    wd_metrics, wd_cm, wd_report, wd_pred = evaluate_classifier(
        clf_wd, X_wd_te, ywd_te, le_wd, "Warranty Decision"
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
    
    # Feature Importance Analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    feature_names = FEATURE_NAMES
    
    fa_importance = clf_fa.feature_importances_
    fa_top = sorted(zip(feature_names, fa_importance), key=lambda x: x[1], reverse=True)[:20]
    
    print("\nTop 20 Features for Failure Analysis:")
    for name, imp in fa_top:
        print(f"  {name}: {imp:.4f}")
    
    wd_importance = clf_wd.feature_importances_
    wd_top = sorted(zip(feature_names, wd_importance), key=lambda x: x[1], reverse=True)[:20]
    
    print("\nTop 20 Features for Warranty Decision:")
    for name, imp in wd_top:
        print(f"  {name}: {imp:.4f}")
    
    # Cross-Validation Analysis (using smaller sample for speed)
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION ANALYSIS (3-fold)")
    print("=" * 60)
    
    X_sample, _, y_fa_sample, _ = train_test_split(X, y_fa, test_size=0.7, random_state=42)
    
    print("\nRunning 3-fold CV on 30% sample (3600 samples)...")
    fa_cv = cross_val_score(clf_fa, X_sample, y_fa_sample, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"Failure Analysis CV Accuracy: {fa_cv.mean():.4f} (+/- {fa_cv.std()*2:.4f})")
    
    _, _, y_wd_sample, _ = train_test_split(X, y_wd, test_size=0.7, random_state=42)
    wd_cv = cross_val_score(clf_wd, X_sample, y_wd_sample, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"Warranty Decision CV Accuracy: {wd_cv.mean():.4f} (+/- {wd_cv.std()*2:.4f})")
    
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
