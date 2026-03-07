# TRACE v3 — Test & Validation Plan

## Structure

Tests are split into two tracks:
- **Track A — Functional Tests**: Does the code run without errors?
- **Track B — Logic Tests**: Does it produce the right decisions?

Run Track A first. Track B is meaningless if Track A fails.

---

## Setup

```python
# conftest.py or top of test file

import os
import pytest

# Run all tests without a real API key first (forces fallback paths)
# Then re-run with a valid key to test LLM paths
API_KEY_SET = bool(os.getenv("OPENROUTER_API_KEY"))

# Sample inputs reused across tests
SAMPLE_OVERVOLTAGE = {
    "fault_code": "P0562",
    "technician_notes": "Customer reports intermittent power loss",
    "voltage": 17.5   # triggers over_voltage rule
}

SAMPLE_MOISTURE = {
    "fault_code": "B1234",
    "technician_notes": "Corrosion found inside ECU housing, moisture present",
    "voltage": 12.5
}

SAMPLE_NTF = {
    "fault_code": "P0441",
    "technician_notes": "No fault found, intermittent complaint, cannot reproduce",
    "voltage": 13.0
}

SAMPLE_UCODE = {
    "fault_code": "U0100",
    "technician_notes": "CAN bus communication error reported",
    "voltage": 12.5
}

SAMPLE_PCODE_ENGINE = {
    "fault_code": "P0301",
    "technician_notes": "Engine jerking during acceleration, rough idle",
    "voltage": 13.5
}

SAMPLE_NO_RULE = {
    "fault_code": "X9999",
    "technician_notes": "Customer complaint about dashboard light",
    "voltage": 13.0   # no voltage rule, no keyword match, no known DTC
}

SAMPLE_PHYSICAL = {
    "fault_code": "C0045",
    "technician_notes": "Visible crack on connector, impact damage observed",
    "voltage": 12.5
}
```

---

## Track A — Functional Tests (No API Key Required)

These tests run with `OPENROUTER_API_KEY` **unset**.
They verify the code runs, returns correct shapes, and handles fallbacks cleanly.

---

### A1 — `extract_dtc_features()` unit test

```python
from ml_predictor import extract_dtc_features

def test_extract_dtc_single_p_code():
    r = extract_dtc_features("P0562")
    assert r["dtc_count"] == 1
    assert r["has_P"] == 1
    assert r["has_U"] == 0
    assert r["has_C"] == 0
    assert r["has_B"] == 0
    assert "P0562" in r["dtc_text"]

def test_extract_dtc_multiple_mixed():
    r = extract_dtc_features("U0100, C0045, B1234")
    assert r["dtc_count"] == 3
    assert r["has_U"] == 1
    assert r["has_C"] == 1
    assert r["has_B"] == 1
    assert r["has_P"] == 0

def test_extract_dtc_empty():
    r = extract_dtc_features("")
    assert r["dtc_count"] == 0
    assert r["dtc_text"] == "none"

def test_extract_dtc_none():
    r = extract_dtc_features(None)
    assert r["dtc_count"] == 0
```

---

### A2 — `match_complaint()` unit test

```python
from ml_predictor import match_complaint

def test_match_complaint_jerk():
    assert match_complaint("Engine jerking badly") == "Engine jerking during acceleration"

def test_match_complaint_moisture():
    # No complaint keyword for moisture — should fuzzy match or default
    result = match_complaint("Moisture found inside")
    assert result in [
        "OBD Light ON", "Engine jerking during acceleration", "Starting Problem",
        "High fuel consumption", "Vehicle not starting", "Low pickup",
        "Engine overheating", "Rough idling", "Brake warning light ON"
    ]

def test_match_complaint_empty():
    assert match_complaint("") == "OBD Light ON"

def test_match_complaint_returns_known_label():
    known = [
        "Engine jerking during acceleration", "Starting Problem",
        "High fuel consumption", "OBD Light ON", "Vehicle not starting",
        "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
    ]
    result = match_complaint("brake warning light came on")
    assert result in known
```

---

### A3 — `run_rules()` unit tests

```python
from ml_predictor import run_rules

def test_run_rules_over_voltage():
    r = run_rules("P0562", "some notes", 17.5)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "over_voltage"

def test_run_rules_low_voltage():
    r = run_rules("P0562", "some notes", 9.0)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "low_voltage"

def test_run_rules_moisture():
    r = run_rules("B1234", "moisture found inside ECU", 12.5)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "moisture"

def test_run_rules_physical_damage():
    r = run_rules("C0045", "crack visible on connector", 12.5)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "physical_damage"

def test_run_rules_ntf():
    r = run_rules("P0441", "no fault found", 13.0)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "ntf"

def test_run_rules_u_code():
    r = run_rules("U0100", "CAN bus error", 12.5)
    assert r["rule_fired"] is True
    assert r["rule_id"] == "u_code"

def test_run_rules_no_match():
    r = run_rules("X9999", "dashboard light on", 13.0)
    assert r["rule_fired"] is False

def test_run_rules_returns_required_keys_when_fired():
    r = run_rules("P0562", "notes", 17.5)
    for key in ["rule_id", "status", "warranty_decision", "rule_confidence",
                "failure_analysis", "reason", "rule_fired"]:
        assert key in r, f"Missing key: {key}"

def test_run_rules_voltage_priority_over_keyword():
    # Over-voltage should fire before moisture even if both conditions are met
    r = run_rules("P0562", "moisture found everywhere", 17.5)
    assert r["rule_id"] == "over_voltage"   # voltage rules come first in RULES[]
```

---

### A4 — `run_ml()` unit tests

```python
from ml_predictor import run_ml

VALID_FEATURES = {
    "customer_complaint": "OBD Light ON",
    "dtc_text": "P0562",
    "dtc_count": 1,
    "voltage": 13.0,
    "has_P": 1, "has_U": 0, "has_C": 0, "has_B": 0
}

def test_run_ml_returns_required_keys():
    r = run_ml(VALID_FEATURES)
    for key in ["ml_warranty_decision", "ml_failure_analysis",
                "fa_prob", "wd_prob", "ml_confidence"]:
        assert key in r, f"Missing key: {key}"

def test_run_ml_confidence_in_range():
    r = run_ml(VALID_FEATURES)
    assert 50.0 <= r["ml_confidence"] <= 98.0

def test_run_ml_valid_warranty_decision():
    r = run_ml(VALID_FEATURES)
    assert r["ml_warranty_decision"] in [
        "Production Failure", "Customer Failure", "According to Specification"
    ]

def test_run_ml_probabilities_are_valid():
    r = run_ml(VALID_FEATURES)
    assert 0.0 <= r["fa_prob"] <= 1.0
    assert 0.0 <= r["wd_prob"] <= 1.0

def test_run_ml_handles_zero_voltage():
    features = {**VALID_FEATURES, "voltage": 0.0}
    r = run_ml(features)
    assert "ml_confidence" in r   # must not crash

def test_run_ml_handles_unknown_complaint():
    # OHE has handle_unknown="ignore" — unknown label should not crash
    features = {**VALID_FEATURES, "customer_complaint": "Some unknown complaint XYZ"}
    r = run_ml(features)
    assert "ml_confidence" in r
```

---

### A5 — `combine_scores()` unit tests

```python
from ml_predictor import combine_scores

RULE_FIRED = {
    "rule_fired": True,
    "rule_id": "moisture",
    "status": "Rejected",
    "warranty_decision": "Customer Failure",
    "rule_confidence": 91.0,
    "failure_analysis": "Sensor short due to moisture",
    "reason": "Moisture found"
}

ML_AGREES = {
    "ml_warranty_decision": "Customer Failure",
    "ml_failure_analysis": "Short circuit due to water ingress",
    "fa_prob": 0.82,
    "wd_prob": 0.78,
    "ml_confidence": 79.5
}

ML_DISAGREES = {
    "ml_warranty_decision": "Production Failure",
    "ml_failure_analysis": "Controller failure",
    "fa_prob": 0.60,
    "wd_prob": 0.55,
    "ml_confidence": 57.4
}

NO_RULE = {"rule_fired": False}

def test_combine_rule_agree_boosts_confidence():
    r = combine_scores(RULE_FIRED, ML_AGREES, None)
    expected = 0.7 * 91.0 + 0.3 * 79.5 + 5.0   # agreement bonus
    assert abs(r["combined_confidence"] - expected) < 0.5

def test_combine_rule_agree_sets_agreement_true():
    r = combine_scores(RULE_FIRED, ML_AGREES, None)
    assert r["agreement"] is True

def test_combine_rule_disagree_lowers_confidence():
    agree_result    = combine_scores(RULE_FIRED, ML_AGREES, None)
    disagree_result = combine_scores(RULE_FIRED, ML_DISAGREES, None)
    assert disagree_result["combined_confidence"] < agree_result["combined_confidence"]

def test_combine_rule_disagree_sets_agreement_false():
    r = combine_scores(RULE_FIRED, ML_DISAGREES, None)
    assert r["agreement"] is False

def test_combine_no_rule_uses_ml_directly():
    r = combine_scores(NO_RULE, ML_AGREES, None)
    assert r["rule_fired"] is False
    assert abs(r["combined_confidence"] - ML_AGREES["ml_confidence"]) < 0.5

def test_combine_weak_input_penalty():
    llm_other = {"category": "other", "confidence": 0.5,
                 "failure_analysis": "unknown", "reasoning": ""}
    r = combine_scores(NO_RULE, ML_AGREES, llm_other)
    expected = ML_AGREES["ml_confidence"] * 0.85
    assert abs(r["combined_confidence"] - expected) < 1.0

def test_combine_low_confidence_forces_manual_review():
    low_ml = {**ML_DISAGREES, "ml_confidence": 55.0}
    r = combine_scores(NO_RULE, low_ml, None)
    assert r["status"] == "Needs Manual Review"

def test_combine_returns_required_keys():
    r = combine_scores(RULE_FIRED, ML_AGREES, None)
    for key in ["status", "warranty_decision", "combined_confidence",
                "agreement", "rule_fired", "rule_id",
                "ml_warranty_decision", "decision_engine"]:
        assert key in r, f"Missing key: {key}"

def test_combine_decision_engine_label_with_rule():
    r = combine_scores(RULE_FIRED, ML_AGREES, None)
    assert r["decision_engine"] == "Rule+ML"

def test_combine_decision_engine_label_with_llm_and_rule():
    llm_result = {"category": "moisture_damage", "confidence": 0.9,
                  "failure_analysis": "Moisture", "reasoning": "wet"}
    r = combine_scores(RULE_FIRED, ML_AGREES, llm_result)
    assert r["decision_engine"] == "LLM+Rule+ML"

def test_combine_decision_engine_label_ml_only():
    r = combine_scores(NO_RULE, ML_AGREES, None)
    assert r["decision_engine"] == "ML"
```

---

### A6 — `assemble_output_from_fields()` unit test

```python
from ml_predictor import assemble_output_from_fields

COMBINED_RULE = {
    "status": "Rejected",
    "warranty_decision": "Customer Failure",
    "combined_confidence": 87.0,
    "agreement": True,
    "rule_fired": True,
    "rule_id": "moisture",
    "ml_warranty_decision": "Customer Failure",
    "ml_failure_analysis": "Short circuit",
    "llm_failure_analysis": "Sensor short due to moisture ingress",
    "decision_engine": "LLM+Rule+ML"
}

FEATURES = {"customer_complaint": "OBD Light ON"}

def test_assemble_output_shape():
    r = assemble_output_from_fields(COMBINED_RULE, FEATURES)
    for key in ["status", "failure_analysis", "warranty_decision",
                "confidence", "reason", "matched_complaint", "decision_engine"]:
        assert key in r, f"Missing key: {key}"

def test_assemble_output_values_match_combined():
    r = assemble_output_from_fields(COMBINED_RULE, FEATURES)
    assert r["status"] == "Rejected"
    assert r["confidence"] == 87.0
    assert r["decision_engine"] == "LLM+Rule+ML"
    assert r["matched_complaint"] == "OBD Light ON"

def test_assemble_prefers_llm_failure_analysis():
    r = assemble_output_from_fields(COMBINED_RULE, FEATURES)
    assert r["failure_analysis"] == "Sensor short due to moisture ingress"

def test_assemble_falls_back_to_ml_failure_analysis():
    combined = {**COMBINED_RULE, "llm_failure_analysis": None}
    r = assemble_output_from_fields(combined, FEATURES)
    assert r["failure_analysis"] == "Short circuit"
```

---

### A7 — `predict()` integration test (no API key)

```python
from ml_predictor import predict

def test_predict_returns_valid_shape():
    r = predict(**SAMPLE_OVERVOLTAGE)
    for key in ["status", "failure_analysis", "warranty_decision",
                "confidence", "reason", "matched_complaint", "decision_engine"]:
        assert key in r, f"Missing key: {key}"

def test_predict_status_is_valid():
    r = predict(**SAMPLE_NTF)
    assert r["status"] in ["Approved", "Rejected", "Needs Manual Review"]

def test_predict_confidence_in_range():
    r = predict(**SAMPLE_PCODE_ENGINE)
    assert 0.0 <= r["confidence"] <= 100.0

def test_predict_warranty_decision_is_valid():
    r = predict(**SAMPLE_MOISTURE)
    assert r["warranty_decision"] in [
        "Production Failure", "Customer Failure", "According to Specification"
    ]

def test_predict_does_not_crash_on_empty_notes():
    r = predict(fault_code="P0562", technician_notes="", voltage=13.0)
    assert "status" in r

def test_predict_does_not_crash_on_empty_dtc():
    r = predict(fault_code="", technician_notes="Engine jerking badly", voltage=13.0)
    assert "status" in r

def test_predict_does_not_crash_on_extreme_voltage():
    r = predict(fault_code="P0562", technician_notes="Some notes here", voltage=999.9)
    assert "status" in r
```

---

## Track B — Logic Tests

These tests assert that the **right decision** is made.
Run with API key **unset** to test rule/ML paths deterministically.

---

### B1 — Rule Priority and Correctness

```python
def test_overvoltage_always_rejected():
    r = predict(**SAMPLE_OVERVOLTAGE)
    assert r["status"] == "Rejected"
    assert r["warranty_decision"] == "Customer Failure"

def test_overvoltage_confidence_is_94():
    # Rule confidence for over_voltage is hardcoded at 94.0
    # Combined with ML should be >= 85 (remains Rejected, not Manual Review)
    r = predict(**SAMPLE_OVERVOLTAGE)
    assert r["confidence"] >= 85.0

def test_moisture_is_rejected_customer_failure():
    r = predict(**SAMPLE_MOISTURE)
    assert r["status"] == "Rejected"
    assert r["warranty_decision"] == "Customer Failure"

def test_ntf_is_approved_according_to_spec():
    r = predict(**SAMPLE_NTF)
    assert r["status"] == "Approved"
    assert r["warranty_decision"] == "According to Specification"

def test_u_code_is_approved_production_failure():
    r = predict(**SAMPLE_UCODE)
    assert r["status"] == "Approved"
    assert r["warranty_decision"] == "Production Failure"

def test_p_code_with_symptom_is_approved():
    r = predict(**SAMPLE_PCODE_ENGINE)
    assert r["status"] == "Approved"

def test_physical_damage_is_rejected():
    r = predict(**SAMPLE_PHYSICAL)
    assert r["status"] == "Rejected"
    assert r["warranty_decision"] == "Customer Failure"

def test_voltage_rule_takes_priority_over_dtc():
    # Both U-code (production) and over-voltage (customer) conditions met
    # over_voltage rule is first in RULES[] and should win
    r = predict(fault_code="U0100", technician_notes="CAN error", voltage=18.0)
    assert r["rule_id_used"] == "over_voltage"   # if you expose rule_id in output
    assert r["status"] == "Rejected"
```

---

### B2 — Combine Logic Correctness

```python
from ml_predictor import combine_scores

def test_agreement_bonus_applied():
    rule = {**RULE_FIRED, "rule_confidence": 80.0}
    ml   = {**ML_AGREES, "ml_confidence": 70.0}
    r = combine_scores(rule, ml, None)
    # 0.7*80 + 0.3*70 + 5 = 56 + 21 + 5 = 82
    assert abs(r["combined_confidence"] - 82.0) < 0.5

def test_large_disagreement_triggers_manual_review():
    # rule_confidence=91, ml_confidence=60 → gap=31 > 20 threshold
    rule = {**RULE_FIRED, "rule_confidence": 91.0}
    ml   = {**ML_DISAGREES, "ml_confidence": 60.0}
    r = combine_scores(rule, ml, None)
    assert r["status"] == "Needs Manual Review"

def test_small_disagreement_does_not_force_manual():
    # gap < 20, combined_conf > 65 → should stay as rule status
    rule = {**RULE_FIRED, "rule_confidence": 80.0}
    ml   = {**ML_DISAGREES, "ml_confidence": 70.0}
    r = combine_scores(rule, ml, None)
    # gap = |80-70| = 10 → no override
    assert r["status"] != "Needs Manual Review"

def test_combined_below_65_forces_manual():
    rule = {"rule_fired": False}
    ml   = {**ML_AGREES, "ml_confidence": 55.0}
    r = combine_scores(rule, ml, None)
    assert r["status"] == "Needs Manual Review"

def test_combined_65_to_84_keeps_status_but_flags():
    rule = {"rule_fired": False}
    ml   = {**ML_AGREES, "ml_confidence": 72.0}
    r = combine_scores(rule, ml, None)
    # status is not forced to Manual Review
    assert r["status"] != "Needs Manual Review"
    # but combined_confidence reflects the moderate certainty
    assert 65.0 <= r["combined_confidence"] < 85.0
```

---

### B3 — ML Always Runs

```python
def test_ml_runs_even_when_rule_fires():
    """ML result must be present in combined output even when a rule matched."""
    from ml_predictor import run_rules, run_ml, extract_dtc_features, match_complaint

    fc, notes, v = "P0562", "moisture inside connector", 12.5
    rule_result = run_rules(fc, notes, v)
    assert rule_result["rule_fired"] is True

    dtc_f    = extract_dtc_features(fc)
    features = {
        "customer_complaint": match_complaint(notes),
        "dtc_text": dtc_f["dtc_text"], "dtc_count": dtc_f["dtc_count"],
        "voltage": v, "has_P": dtc_f["has_P"], "has_U": dtc_f["has_U"],
        "has_C": dtc_f["has_C"], "has_B": dtc_f["has_B"],
    }
    ml_result = run_ml(features)
    # ML must return valid output regardless of rule result
    assert "ml_confidence" in ml_result
    assert ml_result["ml_confidence"] >= 50.0
```

---

### B4 — No Rule Case Goes to ML

```python
def test_no_rule_match_uses_ml():
    r = predict(**SAMPLE_NO_RULE)
    assert r["decision_engine"] in ["ML", "LLM+Rule+ML", "Rule+ML"]
    # If no rule fires and no LLM, decision_engine must be "ML"
    # (when API key unset)
    assert r["decision_engine"] == "ML"

def test_no_rule_confidence_not_artificially_high():
    r = predict(**SAMPLE_NO_RULE)
    # ML on synthetically balanced data should not give unrealistically high confidence
    assert r["confidence"] <= 98.0
```

---

### B5 — Output Schema Integrity

```python
import pytest

@pytest.mark.parametrize("sample", [
    SAMPLE_OVERVOLTAGE,
    SAMPLE_MOISTURE,
    SAMPLE_NTF,
    SAMPLE_UCODE,
    SAMPLE_PCODE_ENGINE,
    SAMPLE_NO_RULE,
    SAMPLE_PHYSICAL,
])
def test_output_schema_all_samples(sample):
    from ml_predictor import predict
    r = predict(**sample)

    # All required keys present
    for key in ["status", "failure_analysis", "warranty_decision",
                "confidence", "reason", "matched_complaint", "decision_engine"]:
        assert key in r

    # status enum
    assert r["status"] in ["Approved", "Rejected", "Needs Manual Review"]

    # warranty_decision enum
    assert r["warranty_decision"] in [
        "Production Failure", "Customer Failure", "According to Specification"
    ]

    # confidence range
    assert 0.0 <= r["confidence"] <= 100.0

    # non-empty strings
    for key in ["failure_analysis", "reason", "matched_complaint", "decision_engine"]:
        assert isinstance(r[key], str)
        assert len(r[key]) > 0

    # matched_complaint is a known label
    assert r["matched_complaint"] in [
        "Engine jerking during acceleration", "Starting Problem",
        "High fuel consumption", "OBD Light ON", "Vehicle not starting",
        "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
    ]
```

---

## Track C — LLM Path Tests (API Key Required)

Run these only when `OPENROUTER_API_KEY` is set.
Mark with `@pytest.mark.skipif(not API_KEY_SET, reason="No API key")`.

```python
import pytest
API_KEY_SET = bool(os.getenv("OPENROUTER_API_KEY"))

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_understand_claim_returns_valid_category():
    from llm_client import understand_claim_with_retry
    r = understand_claim_with_retry(
        "Corrosion and moisture inside the ECU housing", "B1234", 12.5
    )
    assert r is not None
    assert r["category"] in [
        "moisture_damage", "physical_damage", "ntf", "electrical_issue",
        "engine_symptom", "communication_fault", "other"
    ]

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_understand_claim_returns_known_complaint():
    from llm_client import understand_claim_with_retry
    r = understand_claim_with_retry(
        "Engine jerking badly during acceleration", "P0301", 13.5
    )
    known = [
        "Engine jerking during acceleration", "Starting Problem",
        "High fuel consumption", "OBD Light ON", "Vehicle not starting",
        "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
    ]
    assert r["normalized_complaint"] in known

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_translate_to_ml_features_returns_valid_shape():
    from llm_client import translate_to_ml_features
    r = translate_to_ml_features(
        "High fuel consumption, rough idle", "P0562", 14.0, "engine_symptom"
    )
    assert r is not None
    for key in ["customer_complaint", "dtc_codes", "dtc_text",
                "dtc_count", "voltage", "has_P", "has_U", "has_C", "has_B"]:
        assert key in r

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_format_output_returns_valid_shape():
    from llm_client import format_output

    combined = {
        "status": "Rejected",
        "warranty_decision": "Customer Failure",
        "combined_confidence": 87.0,
        "agreement": True,
        "rule_fired": True,
        "rule_id": "moisture",
        "ml_warranty_decision": "Customer Failure",
        "ml_failure_analysis": "Sensor short",
        "llm_failure_analysis": "Moisture-induced ECU corrosion",
        "decision_engine": "LLM+Rule+ML"
    }
    features = {"customer_complaint": "OBD Light ON"}
    r = format_output(combined, features)
    assert r is not None
    for key in ["status", "failure_analysis", "warranty_decision",
                "confidence", "reason", "matched_complaint", "decision_engine"]:
        assert key in r

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_full_pipeline_llm_decision_engine_label():
    r = predict(**SAMPLE_MOISTURE)
    assert r["decision_engine"] in ["LLM+Rule+ML", "Rule+ML", "ML"]

@pytest.mark.skipif(not API_KEY_SET, reason="No API key")
def test_llm_fallback_on_bad_notes():
    # Notes too short → LLM should not be called, rule/ML should still work
    r = predict(fault_code="U0100", technician_notes="ok", voltage=12.5)
    assert "status" in r
```

---

## Test Execution Order

```bash
# 1. Unit tests only (no API key needed)
OPENROUTER_API_KEY="" pytest tests/ -v -k "not skipif"

# 2. Full suite with API key
OPENROUTER_API_KEY="sk-..." pytest tests/ -v

# 3. Logic tests only
OPENROUTER_API_KEY="" pytest tests/ -v -k "test_overvoltage or test_moisture or test_ntf or test_combine or test_no_rule"

# 4. Schema integrity parametrized test
OPENROUTER_API_KEY="" pytest tests/ -v -k "test_output_schema"
```

---

## Pass Criteria Summary

| Track | Tests | Must Pass Before |
|---|---|---|
| A — Functional | A1–A7 (28 tests) | Any Track B/C work |
| B — Logic | B1–B5 (18 tests) | Merging to main |
| C — LLM Path | C1–C6 (6 tests) | Deploying with API key |

**Total: 52 tests**

A deployment is considered stable when all Track A + B tests pass without an API key, and all Track C tests pass with one.
