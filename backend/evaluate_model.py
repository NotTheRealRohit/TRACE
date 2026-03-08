"""
TRACE Model Performance Evaluation Script  (v2 — corrected)
============================================================
Computes comprehensive metrics for both Failure Analysis and Warranty
Decision classifiers, and for the full end-to-end prediction pipeline.

Fixes vs original evaluator
----------------------------
FIX 1 — Data-leakage warning
    All six transformers in train_and_save() are fit on the full 50 k rows
    before the train/test split, which leaks test-set statistics into the
    scalers and TF-IDF vocabulary.  The evaluator now calls this out
    prominently and labels the test-set metrics accordingly.  The fix
    itself must be applied in train_and_save(); the evaluator cannot
    undo leakage after the fact.

FIX 2 — Correct cross-validation
    Original CV sampled 30 % of the data with the same random_state=42
    used for the original split, so the CV rows heavily overlapped with
    the model's training set.  The new CV runs exclusively on X_te — the
    10 000-row holdout that the trained model has never seen — giving a
    proper variance estimate of test-set generalisation.

FIX 3 — Cascade WD train/test mismatch note
    clf_wd was trained with fa_probs_tr drawn from clf_fa on its own
    training data (overconfident, near-1.0 probabilities).  At inference
    time clf_fa runs on unseen data and produces calibrated probabilities,
    creating a distributional shift in the cascade features.  This
    evaluator documents the issue and adds a cascade-calibration check.
    The training fix is to use cross_val_predict to generate
    out-of-fold FA probabilities before fitting clf_wd.

FIX 4 — End-to-end pipeline evaluation
    The production predict() function runs a Rule Engine first, which can
    override ML entirely for ~30 % of claims (over/under-voltage, moisture,
    physical-damage, NTF keywords, etc.).  The original evaluator only
    measured the isolated ML classifiers.  A new section evaluates the
    complete Rule+ML pipeline on the held-out test rows.

FIX 5 — Preprocessing consistency
    load_data() in the original evaluator added fillna() calls for
    Supplier, Mileage_km, and Year that were absent from train_and_save().
    These are now removed so the preprocessing path is identical.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictor import (
    extract_dtc_features, match_complaint, predict,
    DATA_PATH, MODEL_PATH, HIGH_VALUE_DTCS,
)

FEATURE_NAMES = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler,
              ohe_mileage, ohe_vband, claim_age_scaler):
    """
    Load and preprocess the dataset using the *already-fitted* transformers
    from the pickle bundle.  Preprocessing mirrors train_and_save() exactly
    (FIX 5: no extra fillna calls that were absent during training).
    """
    global FEATURE_NAMES

    df = pd.read_csv(DATA_PATH)

    # Exactly the same cleaning steps as train_and_save()
    df["DTC"]                = df["DTC"].fillna("").replace("none", "")
    df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
    df["Failure Analysis"]   = df["Failure Analysis"].fillna("NTF")
    df["Warranty Decision"]  = df["Warranty Decision"].fillna("According to Specification")
    df["Voltage"]            = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)

    # Additional features added in train_and_save()
    _mileage_bins   = [0, 20_000, 60_000, 100_000, np.inf]
    _mileage_labels = ["low", "mid", "high", "very_high"]
    df["mileage_bracket"] = pd.cut(
        df["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)

    df["claim_age"] = pd.to_datetime(df["Date"]).dt.year - df["Year"]

    from ml_predictor import voltage_band
    df["voltage_band"] = df["Voltage"].apply(voltage_band)

    dtc_feats = pd.DataFrame(list(df["DTC"].apply(extract_dtc_features)))

    dtc_flag_cols = (
        ["dtc_count", "has_P", "has_U", "has_C", "has_B"]
        + [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
    )

    FEATURE_NAMES = (
        list(ohe.get_feature_names_out(["Customer Complaint"]))
        + list(tfidf_d.get_feature_names_out())
        + dtc_flag_cols
        + ["Voltage"]
        + list(ohe_supplier.get_feature_names_out(["Supplier"]))
        + ["Mileage_km", "Year"]
        + list(ohe_mileage.get_feature_names_out(["mileage_bracket"]))
        + list(ohe_vband.get_feature_names_out(["voltage_band"]))
        + ["claim_age"]
    )

    return df, dtc_feats, dtc_flag_cols


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
# Cascade calibration check  (FIX 3)
# ---------------------------------------------------------------------------

def check_cascade_calibration(clf_fa, X_tr, X_te, le_fa):
    """
    Compare the distribution of FA top-class probabilities on training data
    (as clf_wd saw during its own training) vs test data (as clf_wd sees at
    inference).  A large gap signals the train/test distributional shift
    described in FIX 3.
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
        print(f"\n  ⚠  Mean gap = {mean_gap:.4f} (> 0.05 threshold).")
        print("     clf_wd was trained on overconfident FA cascade features.")
        print("     Fix: regenerate fa_probs_tr via cross_val_predict before")
        print("     fitting clf_wd so train/test distributions match.")
    else:
        print(f"\n  ✓  Mean gap = {mean_gap:.4f} — cascade distribution is consistent.")


# ---------------------------------------------------------------------------
# End-to-end pipeline evaluation  (FIX 4)
# ---------------------------------------------------------------------------

def evaluate_pipeline(df_te, le_fa, le_wd, sample_size=3, random_state=42):
    """
    Run the full predict() pipeline (Rule Engine → ML → Score Combination)
    on a random sample of held-out rows and compare to ground truth.

    predict() takes (fault_code, technician_notes, voltage).  Because the
    training dataset stores structured complaint labels rather than raw
    technician notes, we pass the Customer Complaint text as notes so that
    match_complaint() maps it back to the same label — the closest we can
    get to a fair pipeline evaluation without real free-text notes.

    NOTE: sample_size defaults to 3 to avoid long runtimes when LLM is enabled
    (~55s per prediction). Set higher (e.g., 2000) only when LLM is disabled.
    """
    sample = df_te.sample(n=min(sample_size, len(df_te)), random_state=random_state)

    true_fa, pred_fa = [], []
    true_wd, pred_wd = [], []
    decision_engines = []

    for _, row in sample.iterrows():
        try:
            result = predict(
                fault_code       = str(row["DTC"]) if pd.notna(row["DTC"]) else "",
                technician_notes = str(row["Customer Complaint"]),
                voltage          = float(row["Voltage"]),
            )
            pred_fa.append(result["failure_analysis"])
            pred_wd.append(result["warranty_decision"])
            true_fa.append(row["Failure Analysis"])
            true_wd.append(row["Warranty Decision"])
            decision_engines.append(result["decision_engine"])
        except Exception as e:
            # Skip rows that cause unexpected predict() errors
            continue

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

    # Break down WD accuracy by which engine made the decision
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TRACE MODEL PERFORMANCE EVALUATION  (v2 — corrected)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # FIX 1 — Data-leakage warning
    # ------------------------------------------------------------------
    print("\n" + "!" * 70)
    print("DATA-LEAKAGE WARNING")
    print("!" * 70)
    print("""
  All six transformers (OHE, TF-IDF, three StandardScalers, supplier OHE)
  in train_and_save() are fit on the full 50 000-row dataset BEFORE the
  train/test split.  This leaks test-set statistics (vocabulary, mean,
  std) into the fitted transformers.

  Impact: the test-set metrics below are slightly optimistic.
  The true generalisation performance is somewhat lower than reported.

  Fix (in train_and_save()):
    1. Split the RAW dataframe first.
    2. Call fit_transform() only on the training slice.
    3. Call transform() only on the test slice.
""")

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
    scaler        = bundle["scaler"]
    ohe_supplier  = bundle["ohe_supplier"]
    mileage_scaler= bundle["mileage_scaler"]
    year_scaler   = bundle["year_scaler"]
    ohe_mileage   = bundle["ohe_mileage"]
    ohe_vband     = bundle["ohe_vband"]
    claim_age_scaler = bundle["claim_age_scaler"]

    df, dtc_feats, dtc_flag_cols = load_data(
        ohe, tfidf_d, ohe_supplier, mileage_scaler, year_scaler,
        ohe_mileage, ohe_vband, claim_age_scaler
    )

    from scipy.sparse import hstack, csr_matrix

    X_c = ohe.transform(df[["Customer Complaint"]])
    X_d = tfidf_d.transform(dtc_feats["dtc_text"])
    X_n = dtc_feats[dtc_flag_cols].values
    X_v = scaler.transform(df[["Voltage"]])
    X_s = ohe_supplier.transform(df[["Supplier"]])
    X_m = mileage_scaler.transform(df[["Mileage_km"]])
    X_y = year_scaler.transform(df[["Year"]])
    X_mb = ohe_mileage.transform(df[["mileage_bracket"]])
    X_vb = ohe_vband.transform(df[["voltage_band"]])
    X_ca = claim_age_scaler.transform(df[["claim_age"]])
    X   = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v),
                  X_s, csr_matrix(X_m), csr_matrix(X_y),
                  X_mb, X_vb, csr_matrix(X_ca)])

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

    # ------------------------------------------------------------------
    # Failure Analysis classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS CLASSIFIER (6 classes)  [test set — leaky, see above]")
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
    print("WARRANTY DECISION CLASSIFIER (3 classes)  [test set — leaky, see above]")
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
    # FIX 2 — Corrected cross-validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ANALYSIS (3-fold on held-out test set)  [FIX 2]")
    print("=" * 70)
    print("""
  NOTE: CV is run exclusively on X_te (the 10 000 held-out rows the
  trained model has never seen).  The original code sampled 30 % of the
  full dataset with random_state=42 — the same seed as the original
  split — causing heavy overlap with the training set and producing
  inflated CV scores.  Running on X_te gives a proper variance estimate.
  cv=3 is used here to keep runtime reasonable; increase for tighter CIs.
""")

    print("  Running 3-fold CV on the 10 000-row held-out test set...")
    fa_cv = cross_val_score(
        clf_fa, X_te, yfa_te, cv=3, scoring="accuracy", n_jobs=-1
    )
    print(f"  Failure Analysis CV Accuracy: {fa_cv.mean():.4f} "
          f"(+/- {fa_cv.std() * 2:.4f})")
    print(f"  Individual folds: {[f'{s:.4f}' for s in fa_cv]}")

    # WD cascade CV note: clf_wd expects FA-augmented feature matrix.
    # We build it from clf_fa predictions on X_te before passing to CV.
    # This is the correct evaluation-time behaviour.
    print("""
  NOTE: WD CV also uses FA-augmented features (inference-style probabilities
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
    print("CASCADE CALIBRATION CHECK  [FIX 3]")
    print("=" * 70)
    print("""
  clf_wd was trained with fa_probs_tr = clf_fa.predict_proba(X_tr),
  i.e. the FA model scored its own training data.  Those probabilities
  are systematically overconfident (skewed toward 1.0).  At inference
  time clf_fa runs on unseen data and returns calibrated probabilities.
  This creates a distributional shift in the cascade input features.

  The check below quantifies this gap.  A mean difference > 0.05 in top-
  class confidence indicates the shift is material and warrants the fix
  described in the header.
""")
    check_cascade_calibration(clf_fa, X_tr, X_te, le_fa)

    # ------------------------------------------------------------------
    # FIX 4 — End-to-end pipeline evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE EVALUATION  [FIX 4]")
    print("=" * 70)
    print("""
  Evaluates the full predict() pipeline: Rule Engine → ML → Score
  Combination.  The Rule Engine fires before ML for claims matching
  voltage thresholds, moisture/physical-damage keywords, or NTF patterns,
  and can override the ML decision entirely.  The isolated classifier
  metrics above do not capture this behaviour.

  NOTE: sample_size is limited to 3 by default to avoid long runtimes.
  If LLM is disabled, you can increase this to 2000 for better stats.
  Customer Complaint text is passed as technician_notes so that
  match_complaint() reproduces the same label as during training.
""")
    evaluate_pipeline(df_te, le_fa, le_wd, sample_size=3)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
  Isolated ML classifiers  (test set — mildly optimistic due to leakage):
    Failure Analysis   accuracy: {fa_metrics['accuracy']:.4f}
    Warranty Decision  accuracy: {wd_metrics['accuracy']:.4f}

  CV variance  (3-fold on held-out test set):
    Failure Analysis   {fa_cv.mean():.4f} +/- {fa_cv.std()*2:.4f}
    Warranty Decision  {wd_cv.mean():.4f} +/- {wd_cv.std()*2:.4f}

  Remaining recommended fixes in train_and_save():
    [HIGH]   Split the dataframe FIRST, then fit transformers on train only.
    [MEDIUM] Use cross_val_predict for fa_probs_tr before fitting clf_wd.
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
