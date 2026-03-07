# TRACE v3 — Implementation Plan

## Pipeline Overview

```
Input (fault_code, technician_notes, voltage)
  │
  ▼
[STAGE 1] LLM — Semantic Understanding         llm_client.py :: understand_claim()
  │
  ▼
[STAGE 2] Rule Engine — Structured Decision    ml_predictor.py :: run_rules()
  │
  ▼
[STAGE 3] LLM — ML Feature Translation         llm_client.py :: translate_to_ml_features()
  │                                             (fallback: match_complaint() + extract_dtc_features())
  ▼
[STAGE 4] ML — Always-on Confidence Scoring    ml_predictor.py :: run_ml()
  │
  ▼
[STAGE 5] Score Combiner                       ml_predictor.py :: combine_scores()
  │
  ▼
[STAGE 6] LLM — Output Formatter               llm_client.py :: format_output()
  │                                             (fallback: assemble_output_from_fields())
  ▼
ClaimResponse JSON  (schema unchanged)
```

---

## File-by-File Changes

### Files modified
- `llm_client.py` — 3 functions added, 1 extended
- `ml_predictor.py` — restructured into staged functions, combination logic added
- `main.py` — no schema changes, one import update

### Files unchanged
- `ClaimRequest` / `ClaimResponse` Pydantic models — schema stays identical
- Rule definitions in `RULES[]` — untouched
- `extract_dtc_features()` — kept as fallback
- `match_complaint()` — kept as fallback
- `train_and_save()` / `load_models()` — untouched

---

## Stage 1 — LLM Semantic Understanding

### Function signature
```python
# llm_client.py
def understand_claim(notes: str, dtc_code: str, voltage: float) -> dict | None
```

### What it does
Replaces the current `categorize_notes_with_retry()` as the first LLM call.
Extended to return richer output beyond just `category`.

### Prompt contract
Send a single user message. The system sets strict JSON-only mode.

**Prompt template:**
```
You are an automotive warranty analyst. Analyze the claim below and respond ONLY with JSON.

Technician Notes: {notes}
DTC Code: {dtc_code}
Measured Voltage: {voltage}

Classify into EXACTLY ONE category from this list:
  moisture_damage, physical_damage, ntf, electrical_issue,
  engine_symptom, communication_fault, other

Also provide:
- normalized_complaint: one of these exact strings:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- severity: "low" | "medium" | "high"
- failure_analysis: short root cause string (max 15 words)
- reasoning: brief explanation (max 30 words)
- confidence: float 0.0–1.0

Respond ONLY with this JSON structure, no preamble:
{
  "category": "...",
  "normalized_complaint": "...",
  "severity": "...",
  "failure_analysis": "...",
  "reasoning": "...",
  "confidence": 0.0
}
```

### Return shape
```python
{
    "category": str,               # one of 7 categories
    "normalized_complaint": str,   # one of 9 known labels
    "severity": str,               # low | medium | high
    "failure_analysis": str,
    "reasoning": str,
    "confidence": float            # 0.0–1.0
}
```

### Failure behaviour
- Any exception → return `None`
- Invalid JSON → return `None`
- Missing keys → fill with defaults: `category="other"`, `normalized_complaint="OBD Light ON"`, `severity="medium"`, `confidence=0.5`
- Wrap in `understand_claim_with_retry(max_retries=2)` — same retry+backoff pattern as existing `categorize_notes_with_retry()`

### Trigger condition (unchanged)
`len(notes.strip()) > 5 AND OPENROUTER_API_KEY is set`

---

## Stage 2 — Rule Engine

### Function signature
```python
# ml_predictor.py
def run_rules(fault_code: str, notes: str, voltage: float) -> dict | None
```

### What changes
The rule loop is extracted from `predict()` into its own function.
It no longer `return`s immediately — it returns the matched result or `None`.

### Return shape (rule fired)
```python
{
    "rule_id": str,                 # e.g. "p_code_engine"
    "status": str,
    "warranty_decision": str,
    "rule_confidence": float,       # hardcoded % from RULES[]
    "failure_analysis": str,
    "reason": str,
    "rule_fired": True
}
```

### Return shape (no rule fired)
```python
{ "rule_fired": False }
```

### Nothing else changes — rule definitions, order, and match logic are identical.

---

## Stage 3 — LLM ML Feature Translation

### Function signature
```python
# llm_client.py
def translate_to_ml_features(
    notes: str,
    dtc_code: str,
    voltage: float,
    llm_category: str          # from Stage 1
) -> dict | None
```

### What it does
Takes raw input + Stage 1 category and produces a clean feature dict
that maps directly to the ML pipeline's expected input format.

### Prompt contract
```
You are preparing structured features for a machine learning model.
Given the warranty claim below, extract clean structured features.

Technician Notes: {notes}
DTC Code: {dtc_code}
Measured Voltage: {voltage}
Pre-classified Category: {llm_category}

Rules:
- customer_complaint MUST be EXACTLY one of:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- dtc_codes: split comma-separated codes into a list, uppercase, strip spaces
- voltage: use the measured value as a float; if missing use 12.5
- has_P/U/C/B: 1 if any code starts with that letter, else 0

Respond ONLY with this JSON:
{
  "customer_complaint": "...",
  "dtc_codes": ["..."],
  "dtc_text": "...",
  "dtc_count": 0,
  "voltage": 0.0,
  "has_P": 0,
  "has_U": 0,
  "has_C": 0,
  "has_B": 0
}
```

### Return shape
```python
{
    "customer_complaint": str,   # one of 9 labels
    "dtc_codes": list[str],
    "dtc_text": str,             # space-joined for TF-IDF
    "dtc_count": int,
    "voltage": float,
    "has_P": int,                # 0 or 1
    "has_U": int,
    "has_C": int,
    "has_B": int
}
```

### Fallback (LLM fails or returns None)
Call existing functions directly:
```python
dtc_feats = extract_dtc_features(dtc_code)
complaint = match_complaint(notes)
features = {
    "customer_complaint": complaint,
    "dtc_text": dtc_feats["dtc_text"],
    "dtc_count": dtc_feats["dtc_count"],
    "voltage": voltage if voltage is not None else 12.5,
    "has_P": dtc_feats["has_P"],
    "has_U": dtc_feats["has_U"],
    "has_C": dtc_feats["has_C"],
    "has_B": dtc_feats["has_B"],
}
```

---

## Stage 4 — ML Always-On Scoring

### Function signature
```python
# ml_predictor.py
def run_ml(features: dict) -> dict
```

### What changes
The ML block is extracted from `predict()` into its own function.
ML now **always runs**, receiving the feature dict from Stage 3.

### Input
The `features` dict from Stage 3 (or its fallback).

### Internal logic (unchanged from current ML block)
```python
X_c = _bundle["ohe"].transform(df_row[["Customer Complaint"]])
X_d = _bundle["tfidf_d"].transform(df_row["dtc_text"])
X_n = df_row[["dtc_count","has_P","has_U","has_C","has_B"]].values
X_v = _bundle["scaler"].transform(pd.DataFrame([[features["voltage"]]], columns=["Voltage"]))
X   = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v)])

fa_idx  = clf_fa.predict(X)[0]
wd_idx  = clf_wd.predict(X)[0]
fa_prob = float(np.max(clf_fa.predict_proba(X)[0]))
wd_prob = float(np.max(clf_wd.predict_proba(X)[0]))
```

### Return shape
```python
{
    "ml_warranty_decision": str,   # Production Failure | Customer Failure | According to Specification
    "ml_failure_analysis": str,    # label from training set
    "fa_prob": float,              # raw RandomForest probability
    "wd_prob": float,              # raw RandomForest probability
    "ml_confidence": float         # geometric mean, clamped 50–98
}
```

### ml_confidence formula (unchanged)
```python
ml_confidence = round(min(98.0, max(50.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)
```

---

## Stage 5 — Score Combination

### Function signature
```python
# ml_predictor.py
def combine_scores(
    rule_result: dict | None,
    ml_result: dict,
    llm_stage1: dict | None
) -> dict
```

### Agreement check
```python
agreement = (
    rule_result is not None
    and rule_result["warranty_decision"] == ml_result["ml_warranty_decision"]
)
```

### Combination matrix

| Condition | combined_confidence formula | status source |
|---|---|---|
| Rule fired + ML agrees | `0.7 * rule_conf + 0.3 * ml_conf + 5` (agreement bonus) | Rule |
| Rule fired + ML disagrees | `0.6 * rule_conf + 0.1 * ml_conf` | Rule (but flag if gap > 20) |
| No rule fired | `ml_conf * 1.0` | ML status_map |
| LLM category = "other" + no rule | `ml_conf * 0.85` (weak input penalty) | ML status_map |

### Status override rules
```python
if combined_confidence >= 85:
    status = rule_status  # or ml_status if no rule
elif combined_confidence >= 65:
    status = rule_status  # hold, but reason notes uncertainty
else:
    status = "Needs Manual Review"

# Disagreement override
if rule_fired and not agreement:
    gap = abs(rule_conf - ml_conf)
    if gap > 20:
        status = "Needs Manual Review"
```

### Return shape
```python
{
    "status": str,
    "warranty_decision": str,
    "combined_confidence": float,
    "agreement": bool,
    "rule_fired": bool,
    "rule_id": str | None,
    "ml_warranty_decision": str,
    "ml_failure_analysis": str,
    "llm_failure_analysis": str | None,   # from Stage 1
    "decision_engine": str                # "LLM+Rule+ML" | "Rule+ML" | "ML"
}
```

### decision_engine value logic
```python
if llm_stage1 is not None and rule_fired:
    decision_engine = "LLM+Rule+ML"
elif rule_fired:
    decision_engine = "Rule+ML"
else:
    decision_engine = "ML"
```

---

## Stage 6 — LLM Output Formatter

### Function signature
```python
# llm_client.py
def format_output(combined: dict, features: dict) -> dict | None
```

### What it does
Takes the fully combined signal dict and produces all human-readable fields
in the exact shape of the current `ClaimResponse`.

### Prompt contract
```
You are a warranty claims report writer. Given the structured decision below,
write a clear, professional output for a technician to read.

Decision Data:
{combined_json}

Rules:
- status must be EXACTLY: "Approved", "Rejected", or "Needs Manual Review"
- warranty_decision must be EXACTLY one of:
    "Production Failure", "Customer Failure", "According to Specification"
- failure_analysis: synthesize llm_failure_analysis and ml_failure_analysis
  into one concise root cause sentence (max 20 words)
- reason: 1–2 sentences explaining the decision in plain language
- matched_complaint: use customer_complaint from features
- confidence: use combined_confidence exactly as provided (do not change)
- decision_engine: use as provided

Respond ONLY with this JSON:
{
  "status": "...",
  "failure_analysis": "...",
  "warranty_decision": "...",
  "confidence": 0.0,
  "reason": "...",
  "matched_complaint": "...",
  "decision_engine": "..."
}
```

### Return shape
Exact `ClaimResponse` field shape:
```python
{
    "status": str,
    "failure_analysis": str,
    "warranty_decision": str,
    "confidence": float,
    "reason": str,
    "matched_complaint": str,
    "decision_engine": str
}
```

### Fallback (LLM fails)
```python
# ml_predictor.py
def assemble_output_from_fields(combined: dict, features: dict) -> dict:
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
```

---

## Revised `predict()` Orchestrator

This replaces the current monolithic `predict()` in `ml_predictor.py`:

```python
def predict(fault_code: str, technician_notes: str, voltage: float) -> dict:
    global _bundle
    if _bundle is None:
        _bundle = load_models()

    fc    = (fault_code or "").strip()
    notes = (technician_notes or "").strip()
    v     = float(voltage) if voltage is not None else None

    api_key_available = bool(os.getenv("OPENROUTER_API_KEY"))
    llm_available     = api_key_available and len(notes) > 5

    # ── Stage 1: LLM Understand ──────────────────────────────
    llm_stage1 = None
    if llm_available:
        try:
            from llm_client import understand_claim_with_retry
            llm_stage1 = understand_claim_with_retry(notes, fc, v)
        except Exception as e:
            print(f"[TRACE] Stage 1 LLM failed: {e}")

    # ── Stage 2: Rule Engine ──────────────────────────────────
    rule_result = run_rules(fc, notes, v)

    # ── Stage 3: LLM Feature Translation ─────────────────────
    features = None
    if llm_available:
        try:
            from llm_client import translate_to_ml_features
            category = llm_stage1["category"] if llm_stage1 else "other"
            features = translate_to_ml_features(notes, fc, v, category)
        except Exception as e:
            print(f"[TRACE] Stage 3 LLM failed: {e}")

    if features is None:                        # fallback
        dtc_f    = extract_dtc_features(fc)
        features = {
            "customer_complaint": match_complaint(notes),
            "dtc_text":  dtc_f["dtc_text"],
            "dtc_count": dtc_f["dtc_count"],
            "voltage":   v if v is not None else 12.5,
            "has_P": dtc_f["has_P"], "has_U": dtc_f["has_U"],
            "has_C": dtc_f["has_C"], "has_B": dtc_f["has_B"],
        }

    # ── Stage 4: ML Scoring ───────────────────────────────────
    ml_result = run_ml(features)

    # ── Stage 5: Combine ──────────────────────────────────────
    combined = combine_scores(rule_result, ml_result, llm_stage1)

    # ── Stage 6: LLM Format Output ────────────────────────────
    output = None
    if llm_available:
        try:
            from llm_client import format_output
            output = format_output(combined, features)
        except Exception as e:
            print(f"[TRACE] Stage 6 LLM failed: {e}")

    if output is None:                          # fallback
        output = assemble_output_from_fields(combined, features)

    return output
```

---

## LLM Call Budget Per Request

| Scenario | Calls |
|---|---|
| Full pipeline (API key set, notes > 5 chars) | 3 (Stage 1 + Stage 3 + Stage 6) |
| No API key | 0 (Rule + ML only) |
| Stage 1 fails | 0 (full fallback, 0 more LLM calls attempted) |
| Stage 3 fails | Stage 6 still attempted (2 total) |
| Stage 6 fails | Hardcoded fallback assembles output (2 total) |

---

## Constants to Add

```python
# ml_predictor.py — top of file
CONFIDENCE_THRESHOLD_FIRM    = 85.0   # below this, reason notes uncertainty
CONFIDENCE_THRESHOLD_MANUAL  = 65.0   # below this, force Needs Manual Review
AGREEMENT_BONUS              = 5.0    # added when rule and ML agree
DISAGREEMENT_GAP_THRESHOLD   = 20.0   # gap that triggers Manual Review override
WEAK_INPUT_PENALTY           = 0.85   # multiplier when LLM returns "other"

RULE_WEIGHT_AGREE    = 0.7
ML_WEIGHT_AGREE      = 0.3
RULE_WEIGHT_DISAGREE = 0.6
ML_WEIGHT_DISAGREE   = 0.1
```

---

## Dependency Notes

No new pip packages required.
All 3 LLM calls use the existing `requests` + OpenRouter pattern in `llm_client.py`.
`response_format: {"type": "json_object"}` used on all 3 prompts for reliable JSON output.
