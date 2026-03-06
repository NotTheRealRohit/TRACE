---
date: 2026-03-06T00:00:00+00:00
researcher: opencode
git_commit: 17359e29f348e811588dc87cd7297a3b918bb20e
branch: feature/openrouter-llm-integrate
repository: capProj-2
topic: "OpenRouter LLM Integration Plan"
tags: [planning, llm, openrouter, rule-engine, warranty-claims]
status: draft
last_updated: 2026-03-06
last_updated_by: opencode
---

# OpenRouter LLM Integration Plan

## Overview

Integrate OpenRouter LLM (`arcee-ai/trinity-large-preview:free`) into the rule engine to parse and categorize technician notes instead of manual keyword-matching lambdas. This enables more flexible interpretation of free-text notes while maintaining the hybrid rule-based + ML pipeline architecture.

## Project Context

**Language/Stack**: Python 3.9+ / FastAPI
**Test Runner**: pytest
**Test Command**: `pytest` (to be determined based on test files created)

## Current State Analysis

The current rule engine (`backend/ml_predictor.py:42-130`) uses hardcoded keyword-matching lambdas:
- 9 rules with fixed keyword lists (moisture, physical_damage, ntf, etc.)
- First-match-wins evaluation
- No flexibility for varied technician phrasing

The `.env` file already contains: `OPENROUTER_API_KEY="sk-or-v1-..."`

### What Exists:
- `backend/ml_predictor.py` - Main ML + rule engine predictor
- `backend/ml_predictor_DecisionTree.py` - DecisionTree variant (same rule logic)
- `backend/main.py` - FastAPI endpoints
- `backend/.env` - OpenRouter API key

### What's Missing:
- LLM client module for OpenRouter API
- Structured prompt design for note categorization
- Integration layer in rule engine

## Desired End State

After implementation:
1. Technician notes are first processed by LLM to extract categories
2. LLM output is parsed and mapped to rule-like decisions
3. Fallback to existing keyword rules if LLM fails or is unavailable
4. Decision engine field shows "LLM" when LLM is used, "Rule-based" for keyword rules, "ML model" for ML fallback
5. Confidence score reflects LLM's certainty (if provided) or defaults to 80%

### Key Discoveries:
- OpenRouter API uses OpenAI-compatible `/api/v1/chat/completions` endpoint
- Model: `arcee-ai/trinity-large-preview:free`
- Supports structured JSON output via `response_format`
- API key already in `.env`

## What We're NOT Doing

- Replacing the ML model entirely — LLM augments rule engine only
- Real-time streaming responses — synchronous calls for now
- Caching LLM responses — each request hits the API
- Fine-tuning or training — using pre-trained model as-is
- Replacing voltage-based rules (still use numeric thresholds)

## Implementation Approach

**Hybrid LLM + Rules Strategy:**
1. LLM processes technician notes first (replacing keyword matching)
2. If LLM succeeds: use LLM categorization, return with "LLM" decision engine
3. If LLM fails (timeout, error): fall back to existing keyword rules
4. If no rules match: continue to ML fallback

---

## Phase 1: LLM Client Module

### Overview
Create a dedicated module for OpenRouter API communication with structured output parsing.

### TDD — Tests to Write First

Before writing production code, write these failing tests:

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `test_llm_client.py` | `test_returns_category_for_valid_notes` | LLM returns parsed category for normal input |
| `test_llm_client.py` | `test_handles_empty_notes` | Graceful handling of empty/missing notes |
| `test_llm_client.py` | `test_handles_api_error` | Raises exception on API failure |
| `test_llm_client.py` | `test_parses_json_response` | Correctly parses JSON from LLM response |

**Run each test to confirm RED before implementing:**
```bash
pytest -k "test_returns_category_for_valid_notes"
pytest -k "test_handles_empty_notes"
pytest -k "test_handles_api_error"
pytest -k "test_parses_json_response"
```

### Changes Required:

#### 1. New File: `backend/llm_client.py`
**Purpose**: OpenRouter API client with structured output

```python
"""
OpenRouter LLM Client for TRACE Warranty Claims
-----------------------------------------------
Handles API calls to OpenRouter for technician note categorization.
"""

import os
import json
import requests
from typing import Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "arcee-ai/trinity-large-preview:free"

def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return api_key

CATEGORIZATION_PROMPT = """You are a warranty claim analyst for automotive electronics.
Analyze the technician's notes and classify the claim into ONE of these categories:

Categories:
- moisture_damage: water, moisture, wet, flood, rain, humidity, corrosion
- physical_damage: crack, broken, impact, collision, bent, misuse, dropped
- ntf: no fault found, ntf, no trouble, no issue, no defect, intermittent, cannot reproduce
- electrical_issue: voltage abnormal, electrical short, wiring, connector
- engine_symptom: jerking, pickup, acceleration, overheating, fuel, idle, rough
- communication_fault: CAN bus, LIN bus, communication error, U-code
- other: none of the above

Technician Notes: {notes}
DTC Code: {dtc_code}
Measured Voltage: {voltage}

Respond ONLY with JSON in this exact format:
{{
  "category": "category_name",
  "confidence": 0.85,
  "failure_analysis": "short description of root cause",
  "reasoning": "brief explanation"
}}
"""

def categorize_notes(notes: str, dtc_code: str, voltage: Optional[float]) -> dict:
    """
    Call OpenRouter LLM to categorize technician notes.
    
    Args:
        notes: Technician's free-text notes
        dtc_code: Fault code (e.g., "P0562")
        voltage: Measured voltage reading
    
    Returns:
        dict with keys: category, confidence, failure_analysis, reasoning
    
    Raises:
        RuntimeError: If API call fails
    """
    api_key = get_api_key()
    
    prompt = CATEGORIZATION_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        voltage=voltage if voltage is not None else "not provided"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }
    
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    try:
        parsed = json.loads(content)
        return {
            "category": parsed.get("category", "other"),
            "confidence": parsed.get("confidence", 0.8),
            "failure_analysis": parsed.get("failure_analysis", "Unknown"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {content}")
```

#### 2. Update: `backend/requirements.txt`
**Changes**: Add `requests` library

```
requests>=2.31.0
```

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: All 4 tests fail before implementation
- [x] GREEN achieved: All 4 tests pass after implementation
- [x] No regressions: Existing smoke tests still pass
- [x] Code lints cleanly: `ruff check backend/`

#### Manual Verification:
- [x] API key loads correctly from .env
- [ ] Test API call with sample notes returns valid JSON

---

## Phase 2: Integrate LLM into Rule Engine

### Overview
Modify `ml_predictor.py` to call LLM client before iterating keyword rules.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `test_predictor_llm.py` | `test_llm_called_before_rules` | LLM gets called when notes provided |
| `test_predictor_llm.py` | `test_fallback_to_rules_on_llm_error` | Falls back to keyword rules when LLM fails |
| `test_predictor_llm.py` | `test_decision_engine_label_llm` | Returns "LLM" in decision_engine field |
| `test_predictor_llm.py` | `test_empty_notes_skips_llm` | Skips LLM call when notes empty |

**Run each test to confirm RED before implementing:**
```bash
pytest -k "test_llm_called_before_rules"
pytest -k "test_fallback_to_rules_on_llm_error"
pytest -k "test_decision_engine_label_llm"
pytest -k "test_empty_notes_skips_llm"
```

### Changes Required:

#### 1. Modify: `backend/ml_predictor.py`
**Changes**: Add LLM integration at the start of `predict()` function

At line ~235 (before rule engine loop), add:

```python
# 0. LLM-powered note analysis (new)
llm_result = None
if notes and len(notes.strip()) > 5:
    try:
        from llm_client import categorize_notes
        llm_result = categorize_notes(notes, fc, v)
    except Exception as LLM_ERROR:
        print(f"[TRACE] LLM call failed: {LLM_ERROR}, falling back to rules")
        llm_result = None

# 0b. If LLM succeeded, map to response
if llm_result:
    category = llm_result["category"]
    category_to_rule = {
        "moisture_damage": "moisture",
        "physical_damage": "physical_damage", 
        "ntf": "ntf",
        "electrical_issue": "low_voltage" if (v and v < 11.0) else "over_voltage" if (v and v > 16.0) else None,
        "engine_symptom": "p_code_engine",
        "communication_fault": "u_code",
    }
    
    matched_rule_id = category_to_rule.get(category)
    if matched_rule_id:
        for rule in RULES:
            if rule["id"] == matched_rule_id:
                return {
                    "status": rule["status"],
                    "failure_analysis": llm_result["failure_analysis"],
                    "warranty_decision": rule["warranty_decision"],
                    "confidence": round(llm_result["confidence"] * 100, 1),
                    "reason": f"LLM categorization: {llm_result['reasoning']}",
                    "matched_complaint": match_complaint(notes),
                    "decision_engine": "LLM",
                }
    
    return {
        "status": "Needs Manual Review",
        "failure_analysis": llm_result["failure_analysis"],
        "warranty_decision": "According to Specification",
        "confidence": round(llm_result["confidence"] * 100, 1),
        "reason": f"LLM categorization: {llm_result['reasoning']}",
        "matched_complaint": match_complaint(notes),
        "decision_engine": "LLM",
    }
```

**Note**: This replaces lines 236-252 (the keyword rule iteration) with LLM-first approach. The keyword rules become fallback.

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: All 4 tests fail before implementation
- [x] GREEN achieved: All 4 tests pass
- [x] No regressions: Smoke tests pass
- [x] Code lints cleanly

#### Manual Verification:
- [ ] Test API with sample notes: returns LLM decision_engine
- [ ] Test with invalid API key: falls back to rules gracefully

---

## Phase 3: Error Handling & Resilience

### Overview
Ensure the system degrades gracefully when LLM is unavailable. Add circuit breaker pattern and logging.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `test_resilience.py` | `test_timeout_handled` | 30s timeout doesn't hang forever |
| `test_resilience.py` | `test_invalid_json_handled` | Malformed LLM JSON doesn't crash |
| `test_resilience.py` | `test_rate_limit_handled` | Rate limit returns 429 gracefully |

### Changes Required:

#### 1. Enhance: `backend/llm_client.py`
**Changes**: Add timeout, better error handling

```python
def categorize_notes(notes: str, dtc_code: str, voltage: Optional[float], timeout: int = 30) -> dict:
    # ... existing code with timeout=timeout in request
    # Add handling for 429 status (rate limit)
    if response.status_code == 429:
        raise RuntimeError("Rate limited by OpenRouter, try again later")
```

#### 2. Add: `backend/llm_client.py`
**Changes**: Simple retry with exponential backoff

```python
import time

def categorize_notes_with_retry(notes: str, dtc_code: str, voltage: Optional[float], max_retries: int = 2) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            return categorize_notes(notes, dtc_code, voltage)
        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            if "rate" in str(e).lower():
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            raise
    return None
```

### Success Criteria:

#### Automated Verification:
- [ ] RED confirmed: All 3 tests fail before implementation
- [ ] GREEN achieved: All 3 tests pass
- [ ] Timeout completes within 60s total

---

## Phase 4: Testing & Validation

### Overview
Run end-to-end smoke tests and validate the full integration.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `test_e2e.py` | `test_full_pipeline_llm` | Full predict() with LLM returns valid response |
| `test_e2e.py` | `test_fallback_chain` | LLM → Rules → ML fallback works |
| `test_e2e.py` | `test_response_schema` | Response matches ClaimResponse schema |

### Changes Required:

#### 1. Integration tests in `backend/tests/`
Create test files matching the above TDD table.

#### 2. Update smoke tests in `ml_predictor.py`
Add LLM-based test cases to existing smoke tests.

### Success Criteria:

#### Automated Verification:
- [ ] RED confirmed: All 3 tests fail before implementation
- [ ] GREEN achieved: All 3 tests pass
- [ ] All existing smoke tests pass

#### Manual Verification:
- [ ] API endpoint `/analyze` returns correct decision_engine
- [ ] Check logs for LLM calls and fallbacks

---

## Testing Strategy

### Unit Tests (follow Red-Green-Refactor):

| Test Scope | Use When |
|------------|----------|
| Unit | LLM client parsing, error handling |
| Integration | Full predict() pipeline with mock LLM |
| API | POST /analyze endpoint |

### Integration Tests:
- Mock LLM responses using `responses` library or similar
- Test fallback chain: LLM → Rules → ML

### Manual Testing Steps:
1. Start backend: `uvicorn main:app --reload --port 8000`
2. Test with curl:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"fault_code": "P0562", "technician_notes": "Engine overheating, low idle", "voltage": 14.2}'
   ```
3. Verify `decision_engine` is "LLM"
4. Test fallback by providing empty notes or invalid API key

---

## Performance Considerations

- **Latency**: LLM calls add ~1-3s per request (network + model inference)
- **Rate Limits**: Free tier has limits; implement caching if needed
- **Timeout**: 30s timeout prevents hanging requests

---

## Migration Notes

- No database migrations needed
- Existing `trace_models.pkl` continues to work for ML fallback
- Backward compatible: if LLM fails, existing code path works

---

## References

- Research document: `thoughts/shared/research/2026-03-06-rule-engine-technician-notes.md`
- OpenRouter API: https://openrouter.ai/docs/api/reference/overview
- Model: `arcee-ai/trinity-large-preview:free`
