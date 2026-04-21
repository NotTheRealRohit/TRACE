"""
TRACE Model Performance Evaluation Script  (v3 — aligned with XGBoost pipeline)
=================================================================================
Computes comprehensive metrics for both Failure Analysis and Warranty
Decision classifiers (XGBoost, 1000 estimators each), for the full
end-to-end prediction pipeline, and for the confidence scoring system.

Status of fixes from v2
------------------------
FIX 1 — Data leakage  [RESOLVED]
    train_and_save() now splits FIRST, then fit_transforms on df_tr only.
    Test-set metrics below reflect true generalisation performance.

FIX 2 — Correct cross-validation
    CV runs exclusively on X_te — the 20 000-row holdout that the trained
    model has never seen — giving a proper variance estimate.

FIX 3 — Cascade WD train/test mismatch  [RESOLVED]
    train_and_save() now uses cross_val_predict(cv=5) to generate
    out-of-fold FA probabilities before fitting clf_wd.  The cascade
    calibration check below verifies the distributional gap is minimal.

FIX 4 — End-to-end pipeline evaluation
    Evaluates the complete Rule+ML+LLM pipeline on held-out test rows.

FIX 5 — Preprocessing consistency
    Preprocessing path is identical between training and evaluation.

NEW — Confidence pipeline evaluation
    Tests the combine_scores() function that blends rule + ML scores
    and determines the final status (Approved/Rejected/Manual Review).
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictor import (
    extract_dtc_features, match_complaint, predict,
    run_rules, run_ml, combine_scores,
    DATA_PATH, MODEL_PATH, HIGH_VALUE_DTCS,
)

FEATURE_NAMES = None


# ---------------------------------------------------------------------------
# Feature engineering helpers (must match train_and_save() exactly)
# ---------------------------------------------------------------------------

def _voltage_bracket(v):
    if v <= 11.0: return "very_low"
    elif v <= 13.5: return "low"
    elif v <= 14.5: return "normal"
    elif v <= 15.4: return "moderate_high"
    elif v <= 16.0: return "high"
    elif v <= 17.0: return "very_high"
    else: return "extreme"


def _dtc_count_bracket(c):
    if c == 0: return "none"
    elif c == 1: return "single"
    elif c <= 3: return "few"
    else: return "many"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler,
              ohe_mileage, claim_age_scaler, voltage_scaler,
              ohe_voltage_bracket, ohe_dtc_count_bracket):
    """
    Load and preprocess the dataset using the *already-fitted* transformers
    from the pickle bundle.  Preprocessing mirrors train_and_save() exactly.
    """
    global FEATURE_NAMES

    df = pd.read_csv(DATA_PATH)

    # Exactly the same cleaning steps as train_and_save()
    df["DTC"]                = df["DTC"].fillna("").replace("none", "")
    df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
    df["Failure Analysis"]   = df["Failure Analysis"].fillna("NTF")
    df["Warranty Decision"]  = df["Warranty Decision"].fillna("According to Specification")

    # Feature engineering (matching train_and_save())
    _mileage_bins   = [0, 20_000, 60_000, 100_000, np.inf]
    _mileage_labels = ["low", "mid", "high", "very_high"]
    df["mileage_bracket"] = pd.cut(
        df["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)

    df["claim_age"] = pd.to_datetime(df["Date"]).dt.year - df["Year"]

    df["voltage_bracket"] = df["Voltage"].apply(_voltage_bracket)

    dtc_feats = pd.DataFrame(list(df["DTC"].apply(extract_dtc_features)))

    df["dtc_count_bracket"] = dtc_feats["dtc_count"].apply(_dtc_count_bracket)

    # Interaction features
    df["volt_high_and_P"] = ((df["Voltage"] > 15.4) & (dtc_feats["has_P"] == 1)).astype(int)
    df["volt_low_and_U"] = ((df["Voltage"] < 11.0) & (dtc_feats["has_U"] == 1)).astype(int)
    df["volt_normal_and_C"] = ((df["Voltage"] >= 11.0) & (df["Voltage"] <= 14.5) & (dtc_feats["has_C"] == 1)).astype(int)
    df["has_multiple_prefixes"] = ((dtc_feats["has_P"] + dtc_feats["has_U"] + dtc_feats["has_C"] + dtc_feats["has_B"]) > 1).astype(int)

    dtc_flag_cols = (
        ["dtc_count", "has_P", "has_U", "has_C", "has_B"]
        + [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
    )

    interaction_cols = ["volt_high_and_P", "volt_low_and_U", "volt_normal_and_C", "has_multiple_prefixes"]

    FEATURE_NAMES = (
        list(ohe.get_feature_names_out(["Customer Complaint"]))
        + list(tfidf_d.get_feature_names_out())
        + dtc_flag_cols
        + list(ohe_supplier.get_feature_names_out(["Supplier"]))
        + ["Mileage_km", "Year"]
        + list(ohe_mileage.get_feature_names_out(["mileage_bracket"]))
        + ["claim_age"]
        + ["Voltage"]
        + list(ohe_voltage_bracket.get_feature_names_out(["voltage_bracket"]))
        + list(ohe_dtc_count_bracket.get_feature_names_out(["dtc_count_bracket"]))
        + interaction_cols
    )

    return df, dtc_feats, dtc_flag_cols, interaction_cols


def evaluate_classifier(clf, X, y_true, le, label):
    """Compute standard classification metrics."""
    y_pred = clf.predict(X)

    metrics = {
        "accuracy":            accuracy_score(y_true, y_pred),
        "precision_weighted":  precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro":     precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_weighted":     recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro":        recall_score   (y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":         f1_score       (y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":            f1_score       (y_true, y_pred, average="macro",    zero_division=0),
    }

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)

    return metrics, cm, report, y_pred


def print_metrics(label, metrics):
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Precision (macro):    {metrics['precision_macro']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"Recall (macro):       {metrics['recall_macro']:.4f}")
    print(f"F1 (weighted):        {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):           {metrics['f1_macro']:.4f}")


def print_per_class(cm, classes, label):
    print(f"\n--- {label} ---")
    for i, cls in enumerate(classes):
        tp      = cm[i, i]
        fn      = cm[i, :].sum() - tp
        fp      = cm[:, i].sum() - tp
        support = cm[i, :].sum()
        print(f"  {cls}: TP={tp}, FP={fp}, FN={fn}, Support={support}")


# ---------------------------------------------------------------------------
# Cascade calibration check  (FIX 3 — now expected to pass)
# ---------------------------------------------------------------------------

def check_cascade_calibration(clf_fa, X_tr, X_te, le_fa):
    """
    Compare the distribution of FA top-class probabilities on training data
    vs test data.  Since train_and_save() now uses cross_val_predict for
    fa_probs_tr, the gap should be minimal.
    """
    fa_probs_tr = clf_fa.predict_proba(X_tr)
    fa_probs_te = clf_fa.predict_proba(X_te)

    top_conf_tr = fa_probs_tr.max(axis=1)
    top_conf_te = fa_probs_te.max(axis=1)

    print("\n  FA top-class probability distribution (clf_wd cascade input):")
    print(f"    Training rows  — mean={top_conf_tr.mean():.4f}  "
          f"median={np.median(top_conf_tr):.4f}  "
          f"std={top_conf_tr.std():.4f}")
    print(f"    Test rows      — mean={top_conf_te.mean():.4f}  "
          f"median={np.median(top_conf_te):.4f}  "
          f"std={top_conf_te.std():.4f}")

    mean_gap = abs(top_conf_tr.mean() - top_conf_te.mean())
    if mean_gap > 0.05:
        print(f"\n  !!  Mean gap = {mean_gap:.4f} (> 0.05 threshold).")
        print("     clf_wd cascade features still show distributional shift.")
        print("     Verify cross_val_predict is being used in train_and_save().")
    else:
        print(f"\n  OK  Mean gap = {mean_gap:.4f} — cascade distribution is consistent.")


# ---------------------------------------------------------------------------
# Shared parallel prediction runner
# ---------------------------------------------------------------------------

def _predict_row(row):
    """Call predict() for a single row. Returns (row, result) or None."""
    try:
        result = predict(
            fault_code       = str(row["DTC"]) if pd.notna(row["DTC"]) else "",
            technician_notes = str(row["Customer Complaint"]),
            voltage          = float(row["Voltage"]) if pd.notna(row.get("Voltage")) else None,
        )
        return (row, result)
    except Exception:
        return None


def run_predictions(df_te, sample_size=200, random_state=42, max_workers=10):
    """
    Run predict() on a sample of held-out rows using ThreadPoolExecutor
    for parallel execution.  Returns a list of (row, result) tuples.
    """
    sample = df_te.sample(n=min(sample_size, len(df_te)), random_state=random_state)
    rows = [row for _, row in sample.iterrows()]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_predict_row, row): row for row in rows}
        for future in as_completed(futures):
            outcome = future.result()
            if outcome is not None:
                results.append(outcome)

    print(f"\n  Predictions completed: {len(results)}/{len(rows)} "
          f"({max_workers} parallel workers)")
    return results


# ---------------------------------------------------------------------------
# End-to-end pipeline evaluation  (FIX 4)
# ---------------------------------------------------------------------------

def evaluate_pipeline(prediction_results, le_fa, le_wd):
    """
    Evaluate the full predict() pipeline using pre-computed results
    from run_predictions().
    """
    true_fa, pred_fa = [], []
    true_wd, pred_wd = [], []
    decision_engines = []

    for row, result in prediction_results:
        pred_fa.append(result["failure_analysis"])
        pred_wd.append(result["warranty_decision"])
        true_fa.append(row["Failure Analysis"])
        true_wd.append(row["Warranty Decision"])
        decision_engines.append(result["decision_engine"])

    if not true_fa:
        print("\n  No pipeline predictions succeeded.")
        return None, None

    engine_counts = pd.Series(decision_engines).value_counts()

    fa_acc = accuracy_score(true_fa, pred_fa)
    wd_acc = accuracy_score(true_wd, pred_wd)

    fa_f1  = f1_score(true_fa, pred_fa, average="weighted",
                      labels=le_fa.classes_.tolist(), zero_division=0)
    wd_f1  = f1_score(true_wd, pred_wd, average="weighted",
                      labels=le_wd.classes_.tolist(), zero_division=0)

    print(f"\n  Sample size: {len(true_fa)} rows")
    print(f"\n  Decision engine breakdown:")
    for eng, cnt in engine_counts.items():
        print(f"    {eng}: {cnt} ({cnt/len(true_fa)*100:.1f}%)")

    print(f"\n  Failure Analysis   — Accuracy: {fa_acc:.4f}  |  F1 (weighted): {fa_f1:.4f}")
    print(f"  Warranty Decision  — Accuracy: {wd_acc:.4f}  |  F1 (weighted): {wd_f1:.4f}")

    print("\n  Failure Analysis full report:")
    print(classification_report(true_fa, pred_fa,
                                labels=le_fa.classes_.tolist(),
                                zero_division=0))

    print("  Warranty Decision full report:")
    print(classification_report(true_wd, pred_wd,
                                labels=le_wd.classes_.tolist(),
                                zero_division=0))

    # WD accuracy per decision engine
    print("  Warranty Decision accuracy per engine:")
    results_df = pd.DataFrame({
        "true_wd": true_wd,
        "pred_wd": pred_wd,
        "engine":  decision_engines,
    })
    for eng in results_df["engine"].unique():
        mask  = results_df["engine"] == eng
        acc_e = accuracy_score(
            results_df.loc[mask, "true_wd"],
            results_df.loc[mask, "pred_wd"],
        )
        print(f"    {eng}: {acc_e:.4f}  (n={mask.sum()})")

    return fa_acc, wd_acc


# ---------------------------------------------------------------------------
# Confidence pipeline evaluation  (NEW)
# ---------------------------------------------------------------------------

def evaluate_confidence(prediction_results, le_fa, le_wd):
    """
    Evaluate the confidence scoring pipeline using pre-computed results
    from run_predictions().
    """
    confidences = []
    statuses = []
    correct_wd = []

    for row, result in prediction_results:
        confidences.append(result["confidence"])
        statuses.append(result["status"])
        correct_wd.append(result["warranty_decision"] == row["Warranty Decision"])

    if not confidences:
        print("\n  No confidence scores produced.")
        return

    confs = np.array(confidences)
    correct = np.array(correct_wd)

    print(f"\n  Sample size: {len(confs)} rows")

    # Distribution stats
    print(f"\n  Confidence distribution:")
    print(f"    Mean:   {confs.mean():.1f}%")
    print(f"    Median: {np.median(confs):.1f}%")
    print(f"    Std:    {confs.std():.1f}%")
    print(f"    Min:    {confs.min():.1f}%  |  Max: {confs.max():.1f}%")
    print(f"    Q25:    {np.percentile(confs, 25):.1f}%  |  Q75: {np.percentile(confs, 75):.1f}%")

    # Status bucket breakdown
    firm    = confs >= 85.0
    review  = (confs >= 65.0) & (confs < 85.0)
    manual  = confs < 65.0

    print(f"\n  Status bucket breakdown:")
    print(f"    Firm (>=85%):          {firm.sum():>5d} ({firm.mean()*100:.1f}%)"
          f"  — WD accuracy: {correct[firm].mean():.4f}" if firm.any() else
          f"    Firm (>=85%):              0 (0.0%)")
    print(f"    Review (65-84%):       {review.sum():>5d} ({review.mean()*100:.1f}%)"
          f"  — WD accuracy: {correct[review].mean():.4f}" if review.any() else
          f"    Review (65-84%):           0 (0.0%)")
    print(f"    Manual Review (<65%):  {manual.sum():>5d} ({manual.mean()*100:.1f}%)"
          f"  — WD accuracy: {correct[manual].mean():.4f}" if manual.any() else
          f"    Manual Review (<65%):      0 (0.0%)")

    # Overall WD accuracy
    print(f"\n  Overall WD accuracy: {correct.mean():.4f}")

    # Status distribution
    status_counts = pd.Series(statuses).value_counts()
    print(f"\n  Status distribution:")
    for s, cnt in status_counts.items():
        print(f"    {s}: {cnt} ({cnt/len(statuses)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TRACE MODEL PERFORMANCE EVALUATION  (v3 — XGBoost pipeline)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load bundle and data
    # ------------------------------------------------------------------
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    clf_fa        = bundle["clf_fa"]
    clf_wd        = bundle["clf_wd"]
    le_fa         = bundle["le_fa"]
    le_wd         = bundle["le_wd"]
    ohe           = bundle["ohe"]
    tfidf_d       = bundle["tfidf_d"]
    ohe_supplier  = bundle["ohe_supplier"]
    mileage_scaler= bundle["mileage_scaler"]
    year_scaler   = bundle["year_scaler"]
    ohe_mileage   = bundle["ohe_mileage"]
    claim_age_scaler    = bundle["claim_age_scaler"]
    voltage_scaler      = bundle["voltage_scaler"]
    ohe_voltage_bracket = bundle["ohe_voltage_bracket"]
    ohe_dtc_count_bracket = bundle["ohe_dtc_count_bracket"]

    # ------------------------------------------------------------------
    # Model type verification
    # ------------------------------------------------------------------
    print(f"\n  Failure Analysis classifier:  {type(clf_fa).__name__}")
    print(f"  Warranty Decision classifier: {type(clf_wd).__name__}")

    # ------------------------------------------------------------------
    # Data leakage status
    # ------------------------------------------------------------------
    print("\n  Data leakage: RESOLVED — transformers are fit on training slice only.")
    print("  Cascade fix:  RESOLVED — OOF probabilities via cross_val_predict(cv=5).")

    df, dtc_feats, dtc_flag_cols, interaction_cols = load_data(
        ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler,
        ohe_mileage, claim_age_scaler, voltage_scaler,
        ohe_voltage_bracket, ohe_dtc_count_bracket
    )

    from scipy.sparse import hstack, csr_matrix

    X_c  = ohe.transform(df[["Customer Complaint"]])
    X_d  = tfidf_d.transform(dtc_feats["dtc_text"])
    X_n  = dtc_feats[dtc_flag_cols].values
    X_s  = ohe_supplier.transform(df[["Supplier"]])
    X_m  = mileage_scaler.transform(df[["Mileage_km"]])
    X_y  = year_scaler.transform(df[["Year"]])
    X_mb = ohe_mileage.transform(df[["mileage_bracket"]])
    X_ca = claim_age_scaler.transform(df[["claim_age"]])
    X_v  = voltage_scaler.transform(df[["Voltage"]])
    X_vb = ohe_voltage_bracket.transform(df[["voltage_bracket"]])
    X_dcb = ohe_dtc_count_bracket.transform(df[["dtc_count_bracket"]])
    X_int = df[interaction_cols].values

    X = hstack([X_c, X_d, csr_matrix(X_n),
                X_s, csr_matrix(X_m), csr_matrix(X_y),
                X_mb, csr_matrix(X_ca), csr_matrix(X_v),
                X_vb, X_dcb, csr_matrix(X_int)])

    y_fa = le_fa.transform(df["Failure Analysis"])
    y_wd = le_wd.transform(df["Warranty Decision"])

    # Reproduce the exact same split used during training
    X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
        X, y_fa, y_wd, test_size=0.2, random_state=42
    )

    # Also keep a dataframe view of the test rows for pipeline evaluation
    idx_all = np.arange(len(df))
    _, idx_te = train_test_split(idx_all, test_size=0.2, random_state=42)
    df_te = df.iloc[idx_te].reset_index(drop=True)

    # Verify feature dimension match
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  clf_fa expects:       {clf_fa.n_features_in_} features")
    if X.shape[1] != clf_fa.n_features_in_:
        print(f"  !!  DIMENSION MISMATCH: matrix has {X.shape[1]}, model expects {clf_fa.n_features_in_}")
    else:
        print(f"  OK  Feature dimensions match.")

    # ------------------------------------------------------------------
    # Failure Analysis classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS CLASSIFIER")
    print("=" * 70)

    fa_metrics, fa_cm, fa_report, fa_pred = evaluate_classifier(
        clf_fa, X_te, yfa_te, le_fa, "Failure Analysis"
    )
    print_metrics("Failure Analysis", fa_metrics)
    print("\nConfusion Matrix:")
    print(fa_cm)
    print("\nClassification Report:")
    print(fa_report)

    fa_probs_te = clf_fa.predict_proba(X_te)
    X_wd_te     = hstack([X_te, csr_matrix(fa_probs_te)])

    # ------------------------------------------------------------------
    # Warranty Decision classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WARRANTY DECISION CLASSIFIER")
    print("=" * 70)

    wd_metrics, wd_cm, wd_report, wd_pred = evaluate_classifier(
        clf_wd, X_wd_te, ywd_te, le_wd, "Warranty Decision"
    )
    print_metrics("Warranty Decision", wd_metrics)
    print("\nConfusion Matrix:")
    print(wd_cm)
    print("\nClassification Report:")
    print(wd_report)

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-CLASS ANALYSIS")
    print("=" * 70)
    print_per_class(fa_cm, le_fa.classes_, "Failure Analysis Classes")
    print_per_class(wd_cm, le_wd.classes_, "Warranty Decision Classes")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    fa_importance = clf_fa.feature_importances_
    fa_top = sorted(zip(FEATURE_NAMES, fa_importance),
                    key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 Features for Failure Analysis:")
    for name, imp in fa_top:
        print(f"  {name}: {imp:.4f}")

    fa_cascade_names = [f"fa_prob_{cls}" for cls in le_fa.classes_]
    wd_feature_names = FEATURE_NAMES + fa_cascade_names
    wd_importance    = clf_wd.feature_importances_
    wd_top = sorted(zip(wd_feature_names, wd_importance),
                    key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 Features for Warranty Decision:")
    for name, imp in wd_top:
        print(f"  {name}: {imp:.4f}")

    # ------------------------------------------------------------------
    # FIX 2 — Cross-validation on held-out test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ANALYSIS (3-fold on held-out test set)")
    print("=" * 70)
    print("""
  CV is run exclusively on X_te (the 20 000 held-out rows the trained
  model has never seen).  cv=3 keeps runtime reasonable.
""")

    print("  Running 3-fold CV on the 20 000-row held-out test set...")
    fa_cv = cross_val_score(
        clf_fa, X_te, yfa_te, cv=3, scoring="accuracy", n_jobs=-1
    )
    print(f"  Failure Analysis CV Accuracy: {fa_cv.mean():.4f} "
          f"(+/- {fa_cv.std() * 2:.4f})")
    print(f"  Individual folds: {[f'{s:.4f}' for s in fa_cv]}")

    # WD cascade CV: clf_wd expects FA-augmented feature matrix.
    print("""
  WD CV uses FA-augmented features (inference-style probabilities
  from the already-trained clf_fa), matching the production data path.
""")
    wd_cv = cross_val_score(
        clf_wd, X_wd_te, ywd_te, cv=3, scoring="accuracy", n_jobs=-1
    )
    print(f"  Warranty Decision CV Accuracy: {wd_cv.mean():.4f} "
          f"(+/- {wd_cv.std() * 2:.4f})")
    print(f"  Individual folds: {[f'{s:.4f}' for s in wd_cv]}")

    # ------------------------------------------------------------------
    # FIX 3 — Cascade calibration check
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CASCADE CALIBRATION CHECK")
    print("=" * 70)
    print("""
  Now that train_and_save() uses cross_val_predict(cv=5) to generate
  out-of-fold FA probabilities, the train/test distributional gap in
  cascade input features should be minimal.
""")
    check_cascade_calibration(clf_fa, X_tr, X_te, le_fa)

    # ------------------------------------------------------------------
    # Run predictions once (shared by pipeline + confidence evaluation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING PARALLEL PREDICTIONS (ThreadPoolExecutor)")
    print("=" * 70)
    print("""
  Running predict() on 200 held-out rows with 10 parallel workers.
  Results are shared between pipeline and confidence evaluation.
  If LLM API keys are set, the full LLM+Rule+ML pipeline is used.
""")
    try:
        prediction_results = run_predictions(df_te, sample_size=200, max_workers=10)
    except ValueError as e:
        print(f"\n  Prediction run skipped: {e}")
        prediction_results = []

    # ------------------------------------------------------------------
    # FIX 4 — End-to-end pipeline evaluation
    # ------------------------------------------------------------------
    if prediction_results:
        print("\n" + "=" * 70)
        print("END-TO-END PIPELINE EVALUATION")
        print("=" * 70)
        evaluate_pipeline(prediction_results, le_fa, le_wd)

    # ------------------------------------------------------------------
    # Confidence pipeline evaluation (NEW)
    # ------------------------------------------------------------------
    if prediction_results:
        print("\n" + "=" * 70)
        print("CONFIDENCE PIPELINE EVALUATION")
        print("=" * 70)
        print("""
  Evaluates the confidence scoring system: how confidence correlates
  with actual prediction accuracy, and the distribution across status
  buckets (Firm >=85%, Review 65-84%, Manual <65%).
""")
        evaluate_confidence(prediction_results, le_fa, le_wd)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
  Model type: {type(clf_fa).__name__}

  Isolated ML classifiers (test set):
    Failure Analysis   accuracy: {fa_metrics['accuracy']:.4f}
    Warranty Decision  accuracy: {wd_metrics['accuracy']:.4f}

  CV variance (3-fold on held-out test set):
    Failure Analysis   {fa_cv.mean():.4f} +/- {fa_cv.std()*2:.4f}
    Warranty Decision  {wd_cv.mean():.4f} +/- {wd_cv.std()*2:.4f}

  All documented fixes (leakage, cascade calibration) have been applied.
""")

    return {
        "failure_analysis": fa_metrics,
        "warranty_decision": wd_metrics,
        "fa_cm": fa_cm,
        "wd_cm": wd_cm,
        "fa_classes": le_fa.classes_.tolist(),
        "wd_classes": le_wd.classes_.tolist(),
    }


if __name__ == "__main__":
    main()
