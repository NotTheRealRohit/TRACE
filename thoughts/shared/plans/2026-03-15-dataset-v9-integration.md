# Dataset v9 Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate synthetic_warranty_claims_v9.csv (100K rows) into ml_predictor.py while maintaining or improving model performance, and recalibrate rule confidences based on v9's real-world-like noisy patterns.

**Architecture:** Update DATA_PATH, expand HIGH_VALUE_DTCS with 113 new DTCs from v9, recalibrate 9 rule confidences to match v9's actual pattern correlations, update header docstring, retrain model, and verify all tests pass.

**Tech Stack:** Python, scikit-learn, RandomForest, pandas, pytest

---

## Overview

The current ml_predictor.py uses synthetic_warranty_claims_v2.csv (50K rows) with idealized 100% pattern correlations. The new v9 dataset (100K rows, 2019-2025) has noisier, more realistic patterns that require:

1. **DATA_PATH update**: Switch from v2 to v9
2. **HIGH_VALUE_DTCS expansion**: Add 113 new DTCs that v9 uses
3. **Rule confidence recalibration**: Adjust 9 rules based on v9's actual correlations
4. **Header docstring update**: Reflect v9 as the standard dataset
5. **Model retraining**: Delete pickle, retrain on v9
6. **Verification**: Run tests and evaluate model performance

### Key v9 Pattern Changes (from analysis)

| Rule | Old Confidence | v9 Actual | New Confidence |
|------|----------------|-----------|----------------|
| over_voltage | 94% | 93.7% | 93% |
| low_voltage | 83% | 95.4% | 95% |
| u_code | 85% | 56.5% | 57% |
| c_code | 78% | 80.4% | 80% |
| b_code | 76% | 80.7% | 80% |
| ntf | 82% | 95.7% | 95% |

---

## Phase 1: Update DATA_PATH and HIGH_VALUE_DTCS

### Task 1: Update DATA_PATH constant

**Files:**
- Modify: `backend/ml_predictor.py:92`

**Step 1: Write the failing test**

```python
# In backend/tests/test_ml_predictor.py, add test class
class TestV9DataPath:
    def test_data_path_points_to_v9(self):
        from ml_predictor import DATA_PATH
        assert "v9" in DATA_PATH, f"Expected v9 in path, got {DATA_PATH}"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_data_path_points_to_v9 -v`
Expected: FAIL - "Expected v9 in path, got ...v2..."

**Step 3: Update DATA_PATH**

In `backend/ml_predictor.py`, line 92:
```python
# Change from:
DATA_PATH  = os.path.join(BASE_DIR, "synthetic_warranty_claims_v2.csv")
# To:
DATA_PATH  = os.path.join(BASE_DIR, "synthetic_warranty_claims_v9.csv")
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_data_path_points_to_v9 -v`
Expected: PASS

---

### Task 2: Expand HIGH_VALUE_DTCS

**Files:**
- Modify: `backend/ml_predictor.py:102-105`

**Step 1: Write the failing test**

```python
def test_high_value_dtcs_includes_v9_codes(self):
    from ml_predictor import HIGH_VALUE_DTCS
    v9_important = ['P0302', 'P0303', 'P0305', 'P0306', 'P0351', 'P0352', 
                    'P0562', 'P0563', 'U0001', 'U0100', 'B1234', 'C0031']
    for dtc in v9_important:
        assert dtc in HIGH_VALUE_DTCS, f"{dtc} not in HIGH_VALUE_DTCS"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_high_value_dtcs_includes_v9_codes -v`
Expected: FAIL - missing key DTCs

**Step 3: Expand HIGH_VALUE_DTCS**

In `backend/ml_predictor.py`, replace lines 102-105:
```python
HIGH_VALUE_DTCS = [
    # Original 9 codes
    "P0300", "P0615", "P0481", "P1682", "P0301",
    "P0480", "P0073", "P0304", "P0482",
    # v9 additions - frequent DTCs with strong warranty signal
    "P0302", "P0303", "P0305", "P0306",  # Misfire codes
    "P0351", "P0352", "P0353", "P0354", "P0355", "P0356",  # Ignition
    "P0562", "P0563",  # OBD processor codes
    "P0601", "P0602", "P0604", "P0605", "P0606", "P0607", "P0608",  # Processor
    "P0610", "P0611", "P0613", "P0616", "P0617",  # Processor cont.
    "P0620", "P0691", "P0692", "P0693", "P0694",  # Power supply
    "P0420", "P0430",  # Catalyst
    "P0455", "P0456", "P0457",  # Evap
    "P0500", "P0501", "P0502",  # Vehicle speed
    "P0720",  # Output shaft speed
    "U0001", "U0002", "U0028", "U0029",  # CAN bus
    "U0037", "U0038",  # LIN bus
    "U0073", "U0100", "U0101", "U0103",  # Network communication
    "U0114", "U0121", "U0122", "U0128",  # Network cont.
    "U0131", "U0140", "U0155", "U0164", "U0184",  # Network cont.
    "U0401", "U0402", "U0422",  # Network invalid data
    "B1031", "B1045", "B1234", "B2960", "B3055",  # Body
    "C0031", "C0036", "C0045", "C0051", "C0082", "C0265", "C0460", "C0550",  # Chassis
    "P0038", "P0054", "P0069", "P0072",  # Sensor (O2/temp)
    "P0096", "P0097", "P0098",  # Intake air
    "P0101", "P0104", "P0111", "P0112", "P0113",  # MAF/IAT
    "P0116", "P0117", "P0118", "P0128",  # Coolant temp
    "P0131", "P0135", "P0141", "P0161",  # O2 sensor
    "P0171", "P0174",  # Fuel system
    "P0196", "P0197",  # Oil temp
    "P0316", "P0316", "P0325", "P0327", "P0328",  # Knock/ckp
    "P0340", "P0341", "P0342", "P0343",  # CMP
]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_high_value_dtcs_includes_v9_codes -v`
Expected: PASS

---

## Phase 2: Recalibrate Rule Confidences

### Task 3: Update rule confidences based on v9

**Files:**
- Modify: `backend/ml_predictor.py:107-195`

**Step 1: Write the failing test**

```python
def test_rule_confidences_match_v9(self):
    from ml_predictor import RULES
    rule_conf = {r['id']: r['confidence'] for r in RULES}
    expected = {
        'over_voltage': 93.0,
        'low_voltage': 95.0,
        'moisture': 91.0,  # Keep same - moisture pattern robust
        'physical_damage': 88.5,  # Keep same
        'ntf': 95.0,
        'u_code': 57.0,
        'p_code_engine': 80.5,  # Keep same
        'c_code': 80.0,
        'b_code': 80.0,
    }
    for rid, exp_conf in expected.items():
        actual = rule_conf.get(rid)
        assert actual == exp_conf, f"{rid}: expected {exp_conf}, got {actual}"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_rule_confidences_match_v9 -v`
Expected: FAIL - old confidences

**Step 3: Update RULES list**

In `backend/ml_predictor.py`, update these rules:

```python
# Line 114: over_voltage
"confidence": 93.0,  # was 94.0

# Line 124: low_voltage  
"confidence": 95.0,  # was 83.0

# Line 154: ntf
"confidence": 95.0,  # was 82.0

# Line 163: u_code
"confidence": 57.0,  # was 85.0

# Line 184: c_code
"confidence": 80.0,  # was 78.0

# Line 193: b_code
"confidence": 80.0,  # was 76.0
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py::TestV9DataPath::test_rule_confidences_match_v9 -v`
Expected: PASS

---

## Phase 3: Update Header Docstring

### Task 4: Update ml_predictor.py header

**Files:**
- Modify: `backend/ml_predictor.py:1-67`

**Step 1: Write the failing test**

```python
def test_header_docstring_mentions_v9(self):
    with open('backend/ml_predictor.py', 'r') as f:
        content = f.read()
    assert 'synthetic_warranty_claims_v9.csv' in content, "Header should mention v9"
    assert '100 000 rows' in content or '100K' in content, "Header should mention 100K"
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/study/git/capProj-2 && python3 -m pytest backend/tests/test_ml_predictor.py::TestV9DataPath::test_header_docstring_mentions_v9 -v`
Expected: FAIL

**Step 3: Update header docstring**

Replace lines 1-67 with:

```python
# -*- coding: utf-8 -*-
"""
TRACE ML Predictor  —  Hybrid LLM + Rule + ML Engine
------------------------------------------------------
Dataset
  synthetic_warranty_claims_v9.csv (100 000 rows, 2019-2025).
  Synthetically generated with real-world-like noise levels - pattern 
  correlations are approximately 93-96% rather than 100%, making this 
  a more realistic training dataset that better reflects production data.

Six-Stage Prediction Pipeline  (predict())
  Stage 1 — LLM Claim Understanding (optional)
      If OPENROUTER_API_KEY is set and notes are non-trivial, calls
      llm_client.understand_claim_with_retry() to categorise the claim and
      extract a structured failure analysis before any rule or ML logic runs.

  Stage 2 — Rule Engine  (run_rules())
      Nine deterministic automotive rules evaluated in priority order:
        • over_voltage    (V > 16 V  → Customer Failure / Rejected,   93 %)
        • low_voltage     (V < 11 V  → Production Failure / Approved, 95 %)
        • moisture        (keyword match in notes → Customer Failure,  91 %)
        • physical_damage (keyword match          → Customer Failure,  88.5 %)
        • ntf             (No-Trouble-Found keywords → Acc. to Spec, 95 %)
        • u_code          (U-series DTC → Production Failure,          57 %)
        • p_code_engine   (P0-series + symptom keyword → Prod. Failure, 80.5 %)
        • c_code          (C-series DTC → Production Failure,          80 %)
        • b_code          (B-series DTC → Production Failure,          80 %)
      First matching rule wins; returns rule_id, status, warranty_decision,
      confidence, failure_analysis, and a human-readable reason string.
      Note: Rule confidences have been recalibrated for v9's noisy patterns.

  Stage 3 — Feature Extraction
      If LLM is available: llm_client.translate_to_ml_features() maps the
      raw claim to structured ML features.
      Fallback: extract_dtc_features() parses DTC codes into prefix flags
      (has_P/U/C/B), count, high-value DTC one-hot flags (90+ DTCs),
      and TF-IDF text; match_complaint() fuzzy-maps free-text notes 
      to a known complaint label.

  Stage 4 — Cascaded RandomForest Scoring  (run_ml())
      Two RF classifiers (200 estimators each) trained on:
        Customer Complaint (OHE) · DTC text (TF-IDF 40) · DTC flags (90+) ·
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
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/d/study/git/capProj-2 && python3 -m pytest backend/tests/test_ml_predictor.py::TestV9DataPath::test_header_docstring_mentions_v9 -v`
Expected: PASS

---

## Phase 4: Retrain Model on v9

### Task 5: Delete old model and retrain

**Files:**
- Delete: `backend/trace_models.pkl`

**Step 1: Delete pickle file**

Run: `rm -f backend/trace_models.pkl`
Verify: `ls -la backend/trace_models.pkl` should show "No such file"

**Step 2: Run train_and_save to retrain**

Run: `cd backend && python3 -c "from ml_predictor import train_and_save; train_and_save()"`
Expected: Model trains on v9 (100K rows), outputs accuracy metrics

**Step 3: Verify model file created**

Run: `ls -la backend/trace_models.pkl`
Expected: File exists with recent timestamp

---

## Phase 5: Run All Tests

### Task 6: Verify all tests pass

**Step 1: Run full test suite**

Run: `cd backend && python3 -m pytest tests/test_ml_predictor.py -v`
Expected: All tests PASS (may have 1-2 minor variations due to v9 noise)

**Step 2: Run smoke tests**

Run: `cd backend && python3 ml_predictor.py`
Expected: All smoke tests produce valid predictions

---

## Phase 6: Evaluate Model Performance

### Task 7: Run model evaluation

**Step 1: Run evaluate_model.py**

Run: `cd backend && python3 evaluate_model.py`
Expected: 
- FA accuracy: ~85-95%
- WD accuracy: ~80-90%
- Note any degradation from v2 baseline

**Step 2: Document results**

Compare with v2 baseline (if available from previous runs):
- v2 FA accuracy: ~95%+
- v2 WD accuracy: ~90%+

Expected v9 may show slight degradation (~2-5%) due to intentional noise in dataset.

---

## Testing Strategy

### Unit Tests
- Test DATA_PATH points to v9
- Test HIGH_VALUE_DTCS contains v9 codes
- Test rule confidences match v9 recalibrated values
- Test header docstring mentions v9

### Integration Tests  
- All existing test_ml_predictor.py tests still pass
- predict() returns valid shapes and values

### Performance Tests
- Model trains successfully on 100K rows
- Evaluation shows expected accuracy ranges

---

## Success Criteria

### Automated Verification:
- [x] DATA_PATH contains "v9": `pytest test_data_path_points_to_v9`
- [x] HIGH_VALUE_DTCS has 90+ codes: `pytest test_high_value_dtcs_includes_v9_codes`
- [x] Rule confidences recalibrated: `pytest test_rule_confidences_match_v9`
- [x] Header mentions v9: `pytest test_header_docstring_mentions_v9`
- [x] Model trains on v9: `python3 ml_predictor.py` shows training

### Manual Verification:
- [x] Evaluate model performance: `python3 evaluate_model.py`

### Manual Verification:
- [ ] Evaluate model performance: `python3 evaluate_model.py`
- [ ] Run API smoke test: `curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"fault_code": "P0562", "technician_notes": "Engine overheating", "voltage": 14.2}'`

---

## Rollback Plan

If issues occur:
1. Revert DATA_PATH to v2
2. Revert HIGH_VALUE_DTCS to original 9 codes
3. Revert rule confidences to original values
4. Delete v9 pickle and retrain on v2

---

## References

- Research: `thoughts/shared/research/2026-03-15-dataset-v9-vs-v2-comparison.md`
- Research: `thoughts/shared/research/2026-03-15-dataset-ml-coupling-analysis.md`
- Original dataset: `backend/synthetic_warranty_claims_v2.csv`
- New dataset: `backend/synthetic_warranty_claims_v9.csv`
- Tests: `backend/tests/test_ml_predictor.py`
