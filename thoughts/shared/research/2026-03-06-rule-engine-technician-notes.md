---
date: 2026-03-06T00:00:00+00:00
researcher: opencode
git_commit: 17359e29f348e811588dc87cd7297a3b918bb20e
branch: feature/openrouter-llm-integrate
repository: capProj-2
topic: "How the rule engine receives and processes technician notes"
tags: [research, codebase, rule-engine, ml-predictor, warranty-claims]
status: complete
last_updated: 2026-03-06
last_updated_by: opencode
---

# Research: How the Rule Engine Receives and Processes Technician Notes

**Date**: 2026-03-06T00:00:00+00:00
**Researcher**: opencode
**Git Commit**: 17359e29f348e811588dc87cd7297a3b918bb20e
**Branch**: feature/openrouter-llm-integrate
**Repository**: capProj-2

## Research Question

How does the rule engine currently receive and process technician notes? Specifically:
- Where are the keyword-matching lambdas defined?
- How are rules iterated?
- What gets passed downstream when a rule matches vs. when no rules match and it falls through to the ML pipeline?

## Summary

The rule engine in `ml_predictor.py` processes technician notes through a two-stage pipeline:
1. **Rule Engine** (lines 236-252): Iterates through 9 handcrafted rules with keyword-matching lambdas; first match wins
2. **ML Fallback** (lines 254-300): When no rule matches, normalizes notes via `match_complaint()` and uses TF-IDF + RandomForest/DecisionTree for prediction

## Detailed Findings

### 1. Keyword-Matching Lambda Definitions

The `RULES` list is defined at **`backend/ml_predictor.py:42-130`** (also duplicated in `ml_predictor_DecisionTree.py:42-130`).

Each rule is a dict with a `"match"` lambda that takes `(fc, notes, v)` parameters:

| Rule ID | Matching Logic | Keywords/Pattern |
|---------|---------------|------------------|
| `over_voltage` | `v > 16.0` | Voltage threshold |
| `low_voltage` | `v < 11.0` | Voltage threshold |
| `moisture` | `any(k in notes.lower() for k in (...))` | water, moisture, wet, flood, rain, humid, corrosion, corroded |
| `physical_damage` | same pattern | crack, broken, impact, collision, bent, misuse, dropped, physical damage |
| `ntf` | same pattern | no fault, ntf, no trouble, no issue, no defect, intermittent, cannot reproduce |
| `u_code` | `re.search(r'\bU[0-9A-Fa-f]{4}\b', fc)` | U-series DTCs |
| `p_code_engine` | P0xxx + symptom keywords | jerk, pickup, acceleration, overheat, fuel, idle, rough |
| `c_code` | `re.search(r'\bC[0-9A-Fa-f]{4}\b', fc)` | C-series DTCs |
| `b_code` | `re.search(r'\bB[0-9A-Fa-f]{4}\b', fc)` | B-series DTCs |

### 2. Rule Iteration Process

In **`backend/ml_predictor.py:236-252`**:

```python
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
```

**Key behavior**:
- Rules are evaluated in order, **first match wins**
- Execution returns immediately on first match
- Exceptions are silently swallowed (try/except with continue)
- Voltage values are templated into reason strings if present

### 3. Downstream Data Flow

#### When a Rule MATCHES (`backend/ml_predictor.py:242-250`):

```python
return {
    "status":            rule["status"],           # Approved/Rejected
    "failure_analysis":  rule["failure_analysis"],
    "warranty_decision": rule["warranty_decision"],
    "confidence":        rule["confidence"],        # Fixed per-rule (76-94%)
    "reason":            reason,                    # Templated with voltage if present
    "matched_complaint": match_complaint(notes),    # Normalized via keyword mapping
    "decision_engine":   "Rule-based",
}
```

#### When NO Rule Matches — ML Fallback (`backend/ml_predictor.py:254-300`):

1. **Note Normalization** (`ml_predictor.py:255`):
   - `match_complaint(notes)` maps free-text to one of 9 known complaints
   - Uses keyword mapping (lines 152-169) + fuzzy matching via `get_close_matches`

2. **Feature Extraction** (`ml_predictor.py:256-265`):
   - `extract_dtc_features(fc)` parses DTC codes into: dtc_count, has_P, has_U, has_C, has_B, dtc_text

3. **TF-IDF Vectorization** (`ml_predictor.py:267-270`):
   - Vectorizes complaint text and DTC text separately
   - Combines with numeric features

4. **ML Prediction** (`ml_predictor.py:272-279`):
   - RandomForest/DecisionTree predicts `failure_analysis` and `warranty_decision`
   - Confidence is capped at **72%** (much lower than rule-based 76-94%)

5. **Response** (`ml_predictor.py:292-300`):
```python
return {
    "status":            status,
    "failure_analysis":  failure_analysis,
    "warranty_decision": warranty_decision,
    "confidence":        confidence,
    "reason":            reason,
    "matched_complaint": matched_complaint,
    "decision_engine":   "ML model",
}
```

### 4. API Call Flow

```
POST /analyze → main.py:55-68
    ↓
ClaimRequest(fault_code, technician_notes, voltage)
    ↓
ml_predict(fc, notes, v) → ml_predictor.py:226
    ↓
Rule engine (lines 236-252)
    ↓ (if no match)
ML fallback (lines 254-300)
    ↓
ClaimResponse
```

**Important Note**: The current `main.py:16` imports from `ml_predictor_DecisionTree`, not `ml_predictor` (the RandomForest version). Both have identical rule engines.

## Code References

- `backend/ml_predictor.py:42-130` - RULES list with keyword-matching lambdas
- `backend/ml_predictor.py:236-252` - Rule iteration and matching logic
- `backend/ml_predictor.py:254-300` - ML fallback pipeline
- `backend/ml_predictor.py:148-171` - `match_complaint()` function
- `backend/ml_predictor.py:133-145` - `extract_dtc_features()` function
- `backend/main.py:55-68` - API endpoint definition

## Architecture Insights

1. **Hybrid Approach**: Rules provide high-confidence deterministic predictions (76-94%), while ML provides softer predictions (capped at 72%) for cases not covered by rules.

2. **First-Match-Wins**: Rules are evaluated in order; this matters because some rules (like `moisture`, `physical_damage`) check notes for keywords while others (like `u_code`, `c_code`, `b_code`) check only the DTC code.

3. **Note Processing**: Technician notes are processed twice:
   - First in rule lambdas (direct keyword matching via `any(k in notes.lower() for k in ...)`)
   - Second in `match_complaint()` for normalization to standardized complaint categories

4. **Silent Failures**: Rule evaluation swallows exceptions silently, which could mask bugs in lambda definitions.

## Open Questions

- The order of rules matters — is there documentation on why certain rules come first?
- What happens if multiple rules could match? Currently only the first is used.
- Could the rule engine be extended to support weighted rules or multiple rule firing?
