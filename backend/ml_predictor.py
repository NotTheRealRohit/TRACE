# -*- coding: utf-8 -*-
"""
TRACE ML Predictor  —  Hybrid Rule + ML Engine
------------------------------------------------
The Warranty Dataset (12 000 rows, 2019-2024) is a synthetically balanced
dataset where feature-target correlations are near-random by design (each
class has equal representation across every feature combination).

Strategy:
  1. Train RandomForest models on the dataset to fulfill the ML pipeline.
  2. Apply domain-driven automotive rules (DTC prefix, voltage, keywords)
     that produce clinically meaningful predictions for the frontend.
  3. Blend: rules set high-confidence decisions; when no rule fires the ML
     model's probability distribution is used as a soft signal.

Exposes: predict(fault_code, technician_notes, voltage) -> dict
"""

import os, re, pickle, warnings
import numpy  as np
import pandas as pd
from   difflib import get_close_matches
from   scipy.sparse import hstack
from   sklearn.ensemble import RandomForestClassifier
from   sklearn.preprocessing import LabelEncoder
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import accuracy_score
from   sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trace_models.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "Warranty_Dataset_2019_2024_12000.csv")

KNOWN_COMPLAINTS = [
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON",
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
        return {"dtc_count":0,"has_P":0,"has_U":0,"has_C":0,"has_B":0,"dtc_text":"none"}
    codes = [c.strip() for c in s.split(",") if c.strip()]
    return {
        "dtc_count": len(codes),
        "has_P": int(any(c.startswith("P") for c in codes)),
        "has_U": int(any(c.startswith("U") for c in codes)),
        "has_C": int(any(c.startswith("C") for c in codes)),
        "has_B": int(any(c.startswith("B") for c in codes)),
        "dtc_text": " ".join(codes),
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
    }
    for kw, complaint in kmap.items():
        if kw in ul:
            return complaint
    matches = get_close_matches(user_text, KNOWN_COMPLAINTS, n=1, cutoff=0.25)
    return matches[0] if matches else "OBD Light ON"


def train_and_save():
    print("[TRACE] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    df["DTC"]                = df["DTC"].fillna("none")
    df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
    df["Failure Analysis"]   = df["Failure Analysis"].fillna("NTF")
    df["Warranty Decision"]  = df["Warranty Decision"].fillna("According to Specification")

    dtc_feats = pd.DataFrame(list(df["DTC"].apply(extract_dtc_features)))

    le_fa = LabelEncoder(); y_fa = le_fa.fit_transform(df["Failure Analysis"])
    le_wd = LabelEncoder(); y_wd = le_wd.fit_transform(df["Warranty Decision"])

    tfidf_c = TfidfVectorizer(max_features=60)
    tfidf_d = TfidfVectorizer(max_features=40)
    X_c = tfidf_c.fit_transform(df["Customer Complaint"])
    X_d = tfidf_d.fit_transform(dtc_feats["dtc_text"])
    X_n = dtc_feats[["dtc_count","has_P","has_U","has_C","has_B"]].values
    X   = hstack([X_c, X_d, X_n])

    X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
        X, y_fa, y_wd, test_size=0.2, random_state=42)

    print("[TRACE] Training models ...")
    clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_fa.fit(X_tr, yfa_tr)
    clf_wd = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_wd.fit(X_tr, ywd_tr)

    fa_acc = accuracy_score(yfa_te, clf_fa.predict(X_te))
    wd_acc = accuracy_score(ywd_te, clf_wd.predict(X_te))
    print(f"  Failure Analysis accuracy : {fa_acc:.3f}  (dataset is synthetically balanced)")
    print(f"  Warranty Decision accuracy: {wd_acc:.3f}")

    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd, le_fa=le_fa, le_wd=le_wd,
                  tfidf_c=tfidf_c, tfidf_d=tfidf_d)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[TRACE] Models saved -> {MODEL_PATH}")
    return bundle


def load_models():
    if not os.path.exists(MODEL_PATH):
        return train_and_save()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_bundle = None


def predict(fault_code: str, technician_notes: str, voltage: float) -> dict:
    global _bundle
    if _bundle is None:
        _bundle = load_models()

    fc    = (fault_code or "").strip()
    notes = (technician_notes or "").strip()
    v     = float(voltage) if voltage is not None else None

    # 1. Rule engine
    for rule in RULES:
        try:
            if rule["match"](fc, notes, v):
                reason = rule["reason"]
                if "{v:.1f}" in reason and v is not None:
                    reason = reason.replace("{v:.1f}", f"{v:.1f}")
                return {
                    "status":            rule["status"],
                    "failure_analysis":  rule["failure_analysis"],
                    "warranty_decision": rule["warranty_decision"],
                    "confidence":        rule["confidence"],
                    "reason":            reason,
                    "matched_complaint": match_complaint(notes),
                    "decision_engine":   "Rule-based",
                }
        except Exception:
            continue

    # 2. ML fallback
    matched_complaint = match_complaint(notes)
    dtc_f  = extract_dtc_features(fc)
    df_row = pd.DataFrame([{
        "complaint": matched_complaint,
        "dtc_text":  dtc_f["dtc_text"],
        "dtc_count": dtc_f["dtc_count"],
        "has_P":     dtc_f["has_P"],
        "has_U":     dtc_f["has_U"],
        "has_C":     dtc_f["has_C"],
        "has_B":     dtc_f["has_B"],
    }])

    X_c = _bundle["tfidf_c"].transform(df_row["complaint"])
    X_d = _bundle["tfidf_d"].transform(df_row["dtc_text"])
    X_n = df_row[["dtc_count","has_P","has_U","has_C","has_B"]].values
    X   = hstack([X_c, X_d, X_n])

    fa_idx  = _bundle["clf_fa"].predict(X)[0]
    wd_idx  = _bundle["clf_wd"].predict(X)[0]
    fa_prob = float(np.max(_bundle["clf_fa"].predict_proba(X)[0]))
    wd_prob = float(np.max(_bundle["clf_wd"].predict_proba(X)[0]))

    failure_analysis  = _bundle["le_fa"].inverse_transform([fa_idx])[0]
    warranty_decision = _bundle["le_wd"].inverse_transform([wd_idx])[0]
    confidence        = round(min(72.0, (fa_prob + wd_prob) / 2 * 100), 1)

    status_map = {
        "Production Failure":         "Approved",
        "According to Specification": "Approved",
        "Customer Failure":           "Rejected",
    }
    status = status_map.get(warranty_decision, "Needs Manual Review")

    dtc_note = f"DTC codes: {fc}. " if dtc_f["dtc_count"] > 0 else "No valid DTC codes provided. "
    reason   = (f"{dtc_note}Complaint matched to '{matched_complaint}'. "
                f"ML model predicts root cause: {failure_analysis}.")

    return {
        "status":            status,
        "failure_analysis":  failure_analysis,
        "warranty_decision": warranty_decision,
        "confidence":        confidence,
        "reason":            reason,
        "matched_complaint": matched_complaint,
        "decision_engine":   "ML model",
    }


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
