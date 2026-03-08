# -*- coding: utf-8 -*-
"""
TRACE ML Predictor  —  Hybrid LLM + Rule + ML Engine
------------------------------------------------------
Dataset
  synthetic_warranty_claims_v2.csv (50 000 rows, 2019-2024).
  Synthetically generated with strong domain-consistent feature-target correlations across voltage, 
  DTC prefix, and warranty decision — designed so that rule-based and ML features carry real predictive weight.

Six-Stage Prediction Pipeline  (predict())
  Stage 1 — LLM Claim Understanding (optional)
      If OPENROUTER_API_KEY is set and notes are non-trivial, calls
      llm_client.understand_claim_with_retry() to categorise the claim and
      extract a structured failure analysis before any rule or ML logic runs.

  Stage 2 — Rule Engine  (run_rules())
      Nine deterministic automotive rules evaluated in priority order:
        • over_voltage    (V > 16 V  → Customer Failure / Rejected,   94 %)
        • low_voltage     (V < 11 V  → Production Failure / Approved, 83 %)
        • moisture        (keyword match in notes → Customer Failure,  91 %)
        • physical_damage (keyword match          → Customer Failure,  88.5 %)
        • ntf             (No-Trouble-Found keywords → Acc. to Spec,  82 %)
        • u_code          (U-series DTC → Production Failure,          85 %)
        • p_code_engine   (P0-series + symptom keyword → Prod. Failure, 80.5 %)
        • c_code          (C-series DTC → Production Failure,          78 %)
        • b_code          (B-series DTC → Production Failure,          76 %)
      First matching rule wins; returns rule_id, status, warranty_decision,
      confidence, failure_analysis, and a human-readable reason string.

  Stage 3 — Feature Extraction
      If LLM is available: llm_client.translate_to_ml_features() maps the
      raw claim to structured ML features.
      Fallback: extract_dtc_features() parses DTC codes into prefix flags
      (has_P/U/C/B), count, high-value DTC one-hot flags, and TF-IDF text;
      match_complaint() fuzzy-maps free-text notes to a known complaint label.

  Stage 4 — Cascaded RandomForest Scoring  (run_ml())
      Two RF classifiers (200 estimators each) trained on:
        Customer Complaint (OHE) · DTC text (TF-IDF 40) · DTC flags ·
        Voltage (scaled) · Supplier (OHE) · Mileage_km (scaled) · Year (scaled)
      Classifier 1 — Failure Analysis (root cause).
      Classifier 2 — Warranty Decision, whose feature matrix is augmented
                      with the FA probability vector (cascade architecture).
      ML confidence = geometric mean of FA and WD top-class probabilities,
      clamped to [0, 98] %.

  Stage 5 — Score Combination  (combine_scores())
      Weighted blend of rule confidence and ML confidence:
        Agreement    → 0.70 × rule + 0.30 × ML  + 5 % agreement bonus
        Disagreement → 0.55 × rule + 0.35 × ML
        No rule      → ML confidence only  (× 0.85 weak-input penalty if LLM
                        flagged the input category as "other")
      Status thresholds: ≥ 85 % → firm decision, 65–85 % → rule/ML status,
      < 65 % → "Needs Manual Review".
      Decision engine tag: "LLM+Rule+ML" | "Rule+ML" | "ML".

  Stage 6 — Output Formatting
      If LLM available: llm_client.format_output() produces a polished
      natural-language reason string.
      Fallback: assemble_output_from_fields() builds the reason from the
      structured fields returned by the earlier stages.

Public API
  predict(fault_code, technician_notes, voltage) -> dict
    Keys: status, failure_analysis, warranty_decision, confidence,
          reason, matched_complaint, decision_engine
"""

import os, re, pickle, warnings
import numpy  as np
import pandas as pd
from   difflib import get_close_matches
from   scipy.sparse import hstack
from   sklearn.ensemble import RandomForestClassifier
from   sklearn.preprocessing import LabelEncoder
from   sklearn.model_selection import train_test_split, cross_val_predict
from   sklearn.metrics import accuracy_score
from   sklearn.feature_extraction.text import TfidfVectorizer   # kept for DTC text
from   sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_config import get_logger, DecisionLogger

logger = get_logger("trace.ml_predictor")
decision_logger = DecisionLogger(logger)

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trace_models.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "synthetic_warranty_claims_v2.csv")

KNOWN_COMPLAINTS = [
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON",
    "ABS warning light ON", "Battery warning light ON",
    "Engine stalling", "Multiple warning lights ON", "Transmission jerking",
]

HIGH_VALUE_DTCS = [
    "P0300", "P0615", "P0481", "P1682", "P0301",
    "P0480", "P0073", "P0304", "P0482"
]

RULES = [
    {
        "id": "over_voltage",
        "match": lambda fc, notes, v: v is not None and v > 16.0,
        "failure_analysis":  "Track burnt due to EOS",
        "warranty_decision": "Customer Failure",
        "status":            "Rejected",
        "confidence":        94.0,
        "reason":            "Over-voltage detected ({v:.1f} V > 16 V). EOS is a customer-side fault.",
    },
    {
        "id": "low_voltage",
        "match": lambda fc, notes, v: v is not None and v < 11.0,
        "failure_analysis":  "controller failure due to supplier production failure",
        "warranty_decision": "Production Failure",
        "status":            "Approved",
        "confidence":        83.0,
        "reason":            "Low supply voltage ({v:.1f} V < 11 V). Under-voltage points to faulty voltage regulator — production defect.",
    },
    {
        "id": "moisture",
        "match": lambda fc, notes, v: any(k in notes.lower() for k in
                    ("water", "moisture", "wet", "flood", "rain", "humid", "corrosion", "corroded")),
        "failure_analysis":  "Sensor short due to moisture",
        "warranty_decision": "Customer Failure",
        "status":            "Rejected",
        "confidence":        91.0,
        "reason":            "Moisture / environmental contamination observed in technician notes. Warranty voided.",
    },
    {
        "id": "physical_damage",
        "match": lambda fc, notes, v: any(k in notes.lower() for k in
                    ("crack", "broken", "impact", "collision", "bent", "misuse", "dropped", "physical damage")),
        "failure_analysis":  "Connector damage",
        "warranty_decision": "Customer Failure",
        "status":            "Rejected",
        "confidence":        88.5,
        "reason":            "Physical or mechanical damage detected in technician notes. Not covered under warranty.",
    },
    {
        "id": "ntf",
        "match": lambda fc, notes, v: any(k in notes.lower() for k in
                    ("no fault", "ntf", "no trouble", "no issue", "no defect", "intermittent", "cannot reproduce")),
        "failure_analysis":  "NTF",
        "warranty_decision": "According to Specification",
        "status":            "Approved",
        "confidence":        82.0,
        "reason":            "No Trouble Found (NTF) — vehicle operating within specification limits.",
    },
    {
        "id": "u_code",
        "match": lambda fc, notes, v: bool(re.search(r'\bU[0-9A-Fa-f]{4}\b', fc)),
        "failure_analysis":  "controller failure due to supplier production failure",
        "warranty_decision": "Production Failure",
        "status":            "Approved",
        "confidence":        85.0,
        "reason":            "U-series DTC (CAN/LIN communication fault) indicates ECU/controller internal failure — likely production defect.",
    },
    {
        "id": "p_code_engine",
        "match": lambda fc, notes, v: (
            bool(re.search(r'\bP0[0-9]{3}\b', fc.upper())) and
            any(k in notes.lower() for k in ("jerk", "pickup", "acceleration", "overheat", "fuel", "idle", "rough"))
        ),
        "failure_analysis":  "ASIC CJ327 failure due to EOS",
        "warranty_decision": "Production Failure",
        "status":            "Approved",
        "confidence":        80.5,
        "reason":            "Standard P0-series powertrain DTC with matching symptom. ECU-level fault covered under production warranty.",
    },
    {
        "id": "c_code",
        "match": lambda fc, notes, v: bool(re.search(r'\bC[0-9A-Fa-f]{4}\b', fc)),
        "failure_analysis":  "Connector damage",
        "warranty_decision": "Production Failure",
        "status":            "Approved",
        "confidence":        78.0,
        "reason":            "C-series DTC (chassis/braking system). Connector damage is the most common root cause for this code range.",
    },
    {
        "id": "b_code",
        "match": lambda fc, notes, v: bool(re.search(r'\bB[0-9A-Fa-f]{4}\b', fc)),
        "failure_analysis":  "Connector damage",
        "warranty_decision": "Production Failure",
        "status":            "Approved",
        "confidence":        76.0,
        "reason":            "B-series DTC (body electronics). Consistent with connector or wiring loom damage.",
    },
]


def extract_dtc_features(dtc_str: str) -> dict:
    s = str(dtc_str).strip().upper() if dtc_str else ""
    if s in ("", "NA", "NAN", "NONE"):
        return {"dtc_count":0,"has_P":0,"has_U":0,"has_C":0,"has_B":0,"dtc_text":"none",
                **{f"dtc_{d.lower()}": 0 for d in HIGH_VALUE_DTCS}}
    codes = [c.strip() for c in s.split(",") if c.strip()]
    return {
        "dtc_count": len(codes),
        "has_P": int(any(c.startswith("P") for c in codes)),
        "has_U": int(any(c.startswith("U") for c in codes)),
        "has_C": int(any(c.startswith("C") for c in codes)),
        "has_B": int(any(c.startswith("B") for c in codes)),
        "dtc_text": " ".join(codes),
        **{f"dtc_{d.lower()}": int(d in codes) for d in HIGH_VALUE_DTCS},
    }


def match_complaint(user_text: str) -> str:
    if not user_text:
        return "OBD Light ON"
    ul = user_text.lower()
    kmap = {
        "jerk": "Engine jerking during acceleration",
        "accel": "Engine jerking during acceleration",
        "not start": "Vehicle not starting",
        "won't start": "Vehicle not starting",
        "start": "Starting Problem",
        "fuel": "High fuel consumption",
        "obd": "OBD Light ON",
        "pickup": "Low pickup",
        "overheat": "Engine overheating",
        "idle": "Rough idling",
        "rough": "Rough idling",
        "brake": "Brake warning light ON",
        "warning": "OBD Light ON",
        "abs": "ABS warning light ON",
        "battery": "Battery warning light ON",
        "stall": "Engine stalling",
        "multiple": "Multiple warning lights ON",
        "transmission": "Transmission jerking",
    }
    for kw, complaint in kmap.items():
        if kw in ul:
            return complaint
    matches = get_close_matches(user_text, KNOWN_COMPLAINTS, n=1, cutoff=0.25)
    return matches[0] if matches else "OBD Light ON"


def voltage_band(v: float) -> str:
    """
    Bucket raw voltage into domain-meaningful bands.
    Mirrors the voltage thresholds already encoded in the rule engine
    (over_voltage rule: v > 16.0, low_voltage rule: v < 11.0) so that
    clf_wd sees the same non-linear boundary the rules exploit.
    """
    if v < 11.0:
        return "under_voltage"
    if v > 16.0:
        return "over_voltage"
    if v < 12.0:
        return "low_normal"
    if v > 14.5:
        return "high_normal"
    return "nominal"


def train_and_save():
    logger.info("[INIT] Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["DTC"]                = df["DTC"].fillna("").replace("none", "")
    df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
    df["Failure Analysis"]   = df["Failure Analysis"].fillna("NTF")
    df["Warranty Decision"]  = df["Warranty Decision"].fillna("According to Specification")
    df["Voltage"]            = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)

    # LabelEncoders are fit on the full dataset so every target class is known.
    # This is safe: target encoding does not expose test-set feature statistics.
    le_fa = LabelEncoder(); y_fa = le_fa.fit_transform(df["Failure Analysis"])
    le_wd = LabelEncoder(); y_wd = le_wd.fit_transform(df["Warranty Decision"])

    dtc_feats = pd.DataFrame(list(df["DTC"].apply(extract_dtc_features)))

    # ── Step 1: split the RAW dataframe (and every derived array) FIRST ──────
    # Splitting df before any fit_transform call ensures that no test-set
    # statistics (vocabulary, IDF weights, mean, std) can leak into the
    # transformers that will later be used for inference.
    df_tr, df_te, dtc_tr, dtc_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
        df, dtc_feats, y_fa, y_wd, test_size=0.2, random_state=42
    )

    # ── Feature Engineering: new warranty-signal columns ─────────────────────
    # Applied AFTER split so df_tr and df_te are independent.
    # 1. mileage_bracket — warranty decisions are threshold-sensitive to mileage;
    #    OHE captures the non-linearity better than a raw scaled value.
    _mileage_bins   = [0, 20_000, 60_000, 100_000, np.inf]
    _mileage_labels = ["low", "mid", "high", "very_high"]
    df_tr["mileage_bracket"] = pd.cut(
        df_tr["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)
    df_te["mileage_bracket"] = pd.cut(
        df_te["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)

    # 2. claim_age — years between vehicle manufacture year and claim date.
    #    A direct warranty-eligibility signal entirely absent from the current
    #    feature set. Raw 'Year' column (importance 0.016) is the vehicle year
    #    alone; claim_age combines it with the claim date for real signal.
    df_tr["claim_age"] = pd.to_datetime(df_tr["Date"]).dt.year - df_tr["Year"]
    df_te["claim_age"] = pd.to_datetime(df_te["Date"]).dt.year - df_te["Year"]

    # 3. voltage_band — OHE version of voltage buckets aligned to rule thresholds.
    df_tr["voltage_band"] = df_tr["Voltage"].apply(voltage_band)
    df_te["voltage_band"] = df_te["Voltage"].apply(voltage_band)

    dtc_flag_cols = (
        ["dtc_count","has_P","has_U","has_C","has_B"] +
        [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
    )

    # ── Step 2: fit_transform on the TRAINING slice only ─────────────────────
    ohe            = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    tfidf_d        = TfidfVectorizer(max_features=40)
    scaler         = StandardScaler()
    ohe_supplier   = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    mileage_scaler = StandardScaler()
    year_scaler    = StandardScaler()
    ohe_mileage      = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    ohe_vband        = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    claim_age_scaler = StandardScaler()

    X_c_tr = ohe.fit_transform(df_tr[["Customer Complaint"]])
    X_d_tr = tfidf_d.fit_transform(dtc_tr["dtc_text"])
    X_n_tr = dtc_tr[dtc_flag_cols].values
    X_v_tr = scaler.fit_transform(df_tr[["Voltage"]])
    X_s_tr = ohe_supplier.fit_transform(df_tr[["Supplier"]])
    X_m_tr = mileage_scaler.fit_transform(df_tr[["Mileage_km"]])
    X_y_tr = year_scaler.fit_transform(df_tr[["Year"]])
    X_mb_tr = ohe_mileage.fit_transform(df_tr[["mileage_bracket"]])
    X_vb_tr = ohe_vband.fit_transform(df_tr[["voltage_band"]])
    X_ca_tr = claim_age_scaler.fit_transform(df_tr[["claim_age"]])

    from scipy.sparse import csr_matrix
    X_tr = hstack([X_c_tr, X_d_tr, csr_matrix(X_n_tr), csr_matrix(X_v_tr),
                   X_s_tr, csr_matrix(X_m_tr), csr_matrix(X_y_tr),
                   X_mb_tr, X_vb_tr, csr_matrix(X_ca_tr)])

    # ── Step 3: transform() on the TEST slice only ────────────────────────────
    X_c_te = ohe.transform(df_te[["Customer Complaint"]])
    X_d_te = tfidf_d.transform(dtc_te["dtc_text"])
    X_n_te = dtc_te[dtc_flag_cols].values
    X_v_te = scaler.transform(df_te[["Voltage"]])
    X_s_te = ohe_supplier.transform(df_te[["Supplier"]])
    X_m_te = mileage_scaler.transform(df_te[["Mileage_km"]])
    X_y_te = year_scaler.transform(df_te[["Year"]])
    X_mb_te = ohe_mileage.transform(df_te[["mileage_bracket"]])
    X_vb_te = ohe_vband.transform(df_te[["voltage_band"]])
    X_ca_te = claim_age_scaler.transform(df_te[["claim_age"]])

    X_te = hstack([X_c_te, X_d_te, csr_matrix(X_n_te), csr_matrix(X_v_te),
                   X_s_te, csr_matrix(X_m_te), csr_matrix(X_y_te),
                   X_mb_te, X_vb_te, csr_matrix(X_ca_te)])

    logger.info("[INIT] Training Failure Analysis classifier...")
    logger.info("[INIT] Generating OOF FA probabilities for WD cascade (cv=5)...")
    fa_probs_tr = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_tr, yfa_tr,
        cv=5,
        method="predict_proba",
    )
    # Train the final clf_fa on ALL of X_tr (OOF probs used only for clf_wd training)
    clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_fa.fit(X_tr, yfa_tr)

    fa_probs_te = clf_fa.predict_proba(X_te)
    X_wd_tr = hstack([X_tr, csr_matrix(fa_probs_tr)])
    X_wd_te = hstack([X_te, csr_matrix(fa_probs_te)])

    logger.info("[INIT] Training Warranty Decision classifier with FA cascade...")
    clf_wd = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    clf_wd.fit(X_wd_tr, ywd_tr)

    fa_acc = accuracy_score(yfa_te, clf_fa.predict(X_te))
    wd_acc = accuracy_score(ywd_te, clf_wd.predict(X_wd_te))
    logger.info("Failure Analysis accuracy: %.3f (dataset is synthetically balanced)", fa_acc)
    logger.info("Warranty Decision accuracy: %.3f", wd_acc)

    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd, le_fa=le_fa, le_wd=le_wd,
                  ohe=ohe, tfidf_d=tfidf_d, scaler=scaler,
                  ohe_supplier=ohe_supplier, mileage_scaler=mileage_scaler,
                  year_scaler=year_scaler,
                  ohe_mileage=ohe_mileage, ohe_vband=ohe_vband,
                  claim_age_scaler=claim_age_scaler)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("[INIT] Models saved to %s", MODEL_PATH)
    return bundle


def load_models():
    if not os.path.exists(MODEL_PATH):
        return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_bundle = None


def run_rules(fault_code: str, notes: str, voltage: float) -> dict | None:
    """
    Run the rule engine against the claim inputs.

    Args:
        fault_code: DTC code(s)
        notes: Technician's free-text notes
        voltage: Measured voltage reading

    Returns:
        dict with keys: rule_id, status, warranty_decision, rule_confidence, failure_analysis, reason, rule_fired
        None if no rule matches
    """
    for rule in RULES:
        try:
            if rule["match"](fault_code, notes, voltage):
                reason = rule["reason"]
                if "{v:.1f}" in reason and voltage is not None:
                    reason = reason.replace("{v:.1f}", f"{voltage:.1f}")
                return {
                    "rule_id": rule["id"],
                    "status": rule["status"],
                    "warranty_decision": rule["warranty_decision"],
                    "rule_confidence": rule["confidence"],
                    "failure_analysis": rule["failure_analysis"],
                    "reason": reason,
                    "rule_fired": True,
                }
        except Exception:
            continue
    return {"rule_fired": False}


def run_ml(features: dict) -> dict:
    """
    Run ML scoring on the extracted features.

    Uses the trained RandomForest classifiers to predict:
    - Failure analysis (root cause classification)
    - Warranty decision (Production/Customer/According to Spec)

    Confidence is computed as the geometric mean of both predictions' probabilities,
    capped between 50-98%.

    Args:
        features: dict with keys: customer_complaint, dtc_text, dtc_count, voltage, has_P, has_U, has_C, has_B

    Returns:
        dict with keys: ml_warranty_decision, ml_failure_analysis, fa_prob, wd_prob, ml_confidence
    """
    global _bundle
    if _bundle is None:
        _bundle = load_models()

    dtc_flag_cols = (
        ["dtc_count","has_P","has_U","has_C","has_B"] +
        [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
    )
    df_row = pd.DataFrame([{
        "Customer Complaint": features.get("customer_complaint", "OBD Light ON"),
        "dtc_text": features.get("dtc_text", ""),
        **{col: features.get(col, 0) for col in dtc_flag_cols},
    }])

    from scipy.sparse import csr_matrix as _csr
    X_c = _bundle["ohe"].transform(df_row[["Customer Complaint"]])
    X_d = _bundle["tfidf_d"].transform(df_row["dtc_text"])
    X_n = df_row[dtc_flag_cols].values
    X_v = _bundle["scaler"].transform(
        pd.DataFrame([[features.get("voltage", 12.5)]], columns=["Voltage"])
    )

    X_s = _bundle["ohe_supplier"].transform(
        pd.DataFrame([[features.get("supplier", "Unknown")]], columns=["Supplier"])
    )
    X_m = _csr(_bundle["mileage_scaler"].transform(
        pd.DataFrame([[features.get("mileage_km", 50000.0)]], columns=["Mileage_km"])
    ))
    X_y = _csr(_bundle["year_scaler"].transform(
        pd.DataFrame([[features.get("year", 2024)]], columns=["Year"])
    ))

    # New features — must match train_and_save() column order exactly
    _raw_mileage = features.get("mileage_km", 50000.0)
    _mb_val = pd.cut(
        pd.Series([_raw_mileage]),
        bins=[0, 20_000, 60_000, 100_000, np.inf],
        labels=["low", "mid", "high", "very_high"]
    ).astype(str).iloc[0]
    X_mb = _bundle["ohe_mileage"].transform(
        pd.DataFrame([[_mb_val]], columns=["mileage_bracket"])
    )

    _vb_val = voltage_band(features.get("voltage", 12.5))
    X_vb = _bundle["ohe_vband"].transform(
        pd.DataFrame([[_vb_val]], columns=["voltage_band"])
    )

    _ca_val = float(features.get("claim_age", 1))
    X_ca = _csr(_bundle["claim_age_scaler"].transform(
        pd.DataFrame([[_ca_val]], columns=["claim_age"])
    ))

    X = hstack([X_c, X_d, _csr(X_n), _csr(X_v), X_s, X_m, X_y, X_mb, X_vb, X_ca])

    # FIX 5 + FIX 1: single proba call for FA, cascade into WD
    fa_proba_row = _bundle["clf_fa"].predict_proba(X)[0]
    fa_idx = int(np.argmax(fa_proba_row))
    fa_prob = float(fa_proba_row[fa_idx])

    X_wd = hstack([X, _csr(fa_proba_row.reshape(1, -1))])

    wd_proba_row = _bundle["clf_wd"].predict_proba(X_wd)[0]
    wd_idx = int(np.argmax(wd_proba_row))
    wd_prob = float(wd_proba_row[wd_idx])

    ml_failure_analysis = _bundle["le_fa"].inverse_transform([fa_idx])[0]
    ml_warranty_decision = _bundle["le_wd"].inverse_transform([wd_idx])[0]
    ml_confidence = round(min(98.0, max(0.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)

    return {
        "ml_warranty_decision": ml_warranty_decision,
        "ml_failure_analysis": ml_failure_analysis,
        "fa_prob": fa_prob,
        "wd_prob": wd_prob,
        "ml_confidence": ml_confidence,
    }


CONFIDENCE_THRESHOLD_FIRM = 85.0
CONFIDENCE_THRESHOLD_MANUAL = 65.0
AGREEMENT_BONUS = 5.0
DISAGREEMENT_GAP_THRESHOLD = 20.0
WEAK_INPUT_PENALTY = 0.85

RULE_WEIGHT_AGREE = 0.7
ML_WEIGHT_AGREE = 0.3
RULE_WEIGHT_DISAGREE = 0.55
ML_WEIGHT_DISAGREE = 0.35


def combine_scores(
    rule_result: dict | None,
    ml_result: dict,
    llm_stage1: dict | None,
) -> dict:
    """
    Combine rule engine and ML results into a final decision.

    Logic:
    1. If rule fires and agrees with ML: apply agreement bonus
    2. If rule fires but disagrees: use threshold-based fallback
    3. If no rule fires: use ML result with possible penalty

    Weights defined at module level control the blend.

    Args:
        rule_result: Output from run_rules()
        ml_result: Output from run_ml()
        llm_stage1: Output from understand_claim()

    Returns:
        dict with combined decision, confidence, and metadata
    """
    rule_fired = rule_result is not None and rule_result.get("rule_fired", False)

    if rule_fired:
        rule_conf = rule_result.get("rule_confidence", 0)
        rule_wd = rule_result.get("warranty_decision", "")
        ml_wd = ml_result.get("ml_warranty_decision", "")
        agreement = rule_wd == ml_wd
    else:
        rule_conf = 0
        agreement = False

    ml_conf = ml_result.get("ml_confidence", 50.0)

    if rule_fired and agreement:
        combined_confidence = (
            RULE_WEIGHT_AGREE * rule_conf
            + ML_WEIGHT_AGREE * ml_conf
            + AGREEMENT_BONUS
        )
    elif rule_fired and not agreement:
        combined_confidence = (
            RULE_WEIGHT_DISAGREE * rule_conf
            + ML_WEIGHT_DISAGREE * ml_conf
        )
    else:
        combined_confidence = ml_conf

    if llm_stage1 is not None and llm_stage1.get("category") == "other" and not rule_fired:
        combined_confidence *= WEAK_INPUT_PENALTY

    combined_confidence = round(min(98.0, max(0.0, combined_confidence)), 1)

    if rule_fired and not agreement:
        gap = abs(rule_conf - ml_conf)
        if gap <= DISAGREEMENT_GAP_THRESHOLD:
            status = rule_result.get("status", "Needs Manual Review")
        elif combined_confidence >= CONFIDENCE_THRESHOLD_FIRM:
            status = rule_result.get("status", "Needs Manual Review")
        elif combined_confidence >= CONFIDENCE_THRESHOLD_MANUAL:
            status = rule_result.get("status", "Needs Manual Review")
        else:
            status = "Needs Manual Review"
    elif rule_fired and agreement:
        if combined_confidence >= CONFIDENCE_THRESHOLD_FIRM:
            status = rule_result.get("status", "Needs Manual Review")
        elif combined_confidence >= CONFIDENCE_THRESHOLD_MANUAL:
            status = rule_result.get("status", "Needs Manual Review")
        else:
            status = "Needs Manual Review"
    else:
        if combined_confidence >= CONFIDENCE_THRESHOLD_MANUAL:
            status_map = {
                "Production Failure": "Approved",
                "According to Specification": "Approved",
                "Customer Failure": "Rejected",
            }
            status = status_map.get(ml_result.get("ml_warranty_decision", ""), "Needs Manual Review")
        else:
            status = "Needs Manual Review"

    if rule_fired:
        warranty_decision = rule_result.get("warranty_decision", ml_result.get("ml_warranty_decision", ""))
    else:
        warranty_decision = ml_result.get("ml_warranty_decision", "")

    if llm_stage1 is not None and rule_fired:
        decision_engine = "LLM+Rule+ML"
    elif rule_fired:
        decision_engine = "Rule+ML"
    else:
        decision_engine = "ML"

    return {
        "status": status,
        "warranty_decision": warranty_decision,
        "combined_confidence": combined_confidence,
        "agreement": agreement,
        "rule_fired": rule_fired,
        "rule_id": rule_result.get("rule_id") if rule_fired else None,
        "ml_warranty_decision": ml_result.get("ml_warranty_decision", ""),
        "ml_failure_analysis": ml_result.get("ml_failure_analysis", ""),
        "llm_failure_analysis": llm_stage1.get("failure_analysis") if llm_stage1 else None,
        "decision_engine": decision_engine,
    }


def assemble_output_from_fields(combined: dict, features: dict) -> dict:
    """
    Assemble final output from combined results (fallback when LLM formatter fails).
    """
    return {
        "status": combined["status"],
        "failure_analysis": (
            combined.get("llm_failure_analysis")
            or combined["ml_failure_analysis"]
        ),
        "warranty_decision": combined["warranty_decision"],
        "confidence": combined["combined_confidence"],
        "reason": (
            f"Rule '{combined['rule_id']}' fired. "
            f"ML {'agrees' if combined['agreement'] else 'disagrees'} "
            f"with confidence {combined['combined_confidence']}%."
            if combined["rule_fired"]
            else f"No rule matched. ML predicts {combined['ml_warranty_decision']} "
                 f"with confidence {combined['combined_confidence']}%."
        ),
        "matched_complaint": features.get("customer_complaint", "OBD Light ON"),
        "decision_engine": combined["decision_engine"],
    }


def predict(fault_code: str, technician_notes: str, voltage: float) -> dict:
    global _bundle
    if _bundle is None:
        _bundle = load_models()
        logger.info("[INIT] Models loaded")

    fc = (fault_code or "").strip()
    notes = (technician_notes or "").strip()
    v = float(voltage) if voltage is not None else None

    logger.info("INPUT predict | fault_code=%s voltage=%s notes_len=%d", 
                fc, v, len(notes))

    api_key_available = bool(os.getenv("OPENROUTER_API_KEY"))
    llm_available = api_key_available and len(notes) > 5

    llm_stage1 = None
    if llm_available:
        try:
            from llm_client import understand_claim_with_retry
            llm_stage1 = understand_claim_with_retry(notes, fc, v)
            if llm_stage1:
                decision_logger.log_decision("Stage 1 LLM", llm_stage1)
        except Exception as e:
            logger.warning("[STAGE 1] LLM failed, using fallback: %s", e)

    rule_result = run_rules(fc, notes, v)
    if rule_result.get("rule_fired"):
        decision_logger.log_decision("Rule Engine", rule_result)
        logger.info("Rule fired: %s with confidence %.1f", 
                    rule_result["rule_id"], rule_result["rule_confidence"])

    features = None
    if llm_available:
        try:
            from llm_client import translate_to_ml_features
            category = llm_stage1["category"] if llm_stage1 else "other"
            features = translate_to_ml_features(notes, fc, v, category)
        except Exception as e:
            logger.warning("[STAGE 3] LLM failed, using fallback: %s", e)

    if features is None:
        dtc_f = extract_dtc_features(fc)
        _voltage_val = v if v is not None else 12.5
        features = {
            "customer_complaint": match_complaint(notes),
            "dtc_text": dtc_f["dtc_text"],
            "dtc_count": dtc_f["dtc_count"],
            "voltage": _voltage_val,
            "has_P": dtc_f["has_P"],
            "has_U": dtc_f["has_U"],
            "has_C": dtc_f["has_C"],
            "has_B": dtc_f["has_B"],
            "supplier": "Unknown",
            "mileage_km": 50000.0,
            "year": 2024,
            "claim_age": 1,          # default: assume 1-year-old claim
        }

    ml_result = run_ml(features)
    decision_logger.log_decision("ML Model", {
        "warranty_decision": ml_result["ml_warranty_decision"],
        "failure_analysis": ml_result["ml_failure_analysis"],
        "confidence": ml_result["ml_confidence"],
    })

    combined = combine_scores(rule_result, ml_result, llm_stage1)
    decision_logger.log_decision("Combined", combined)

    output = None
    if llm_available:
        try:
            from llm_client import format_output
            output = format_output(combined, features)
        except Exception as e:
            logger.warning("[STAGE 6] LLM failed, using fallback: %s", e)

    if output is None:
        output = assemble_output_from_fields(combined, features)

    logger.info("OUTPUT predict | status=%s confidence=%.1f engine=%s",
                output["status"], output["confidence"], output["decision_engine"])

    return output


if __name__ == "__main__":
    train_and_save()
    print("\n-- Smoke Tests --")
    tests = [
        ("P0562",        "Engine overheating, low idle",               14.2),
        ("U0100",        "Communication error on CAN bus",              12.5),
        ("P0301",        "Moisture found inside connector, corroded",   12.0),
        ("",             "No fault found, intermittent complaint",      13.1),
        ("B1234",        "Starting problem, nothing visible",           18.5),
        ("C0045, P0987", "Brake warning light ON, vehicle shaking",     12.8),
    ]
    for fc, notes, v in tests:
        r = predict(fc, notes, v)
        print(f"  [{r['status']:25s}] FA: {r['failure_analysis'][:38]:38s} | {r['confidence']}%  [{r['decision_engine']}]")
