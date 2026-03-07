---
date: 2026-03-07T14:45:00+00:00
researcher: opencode
git_commit: c0fae4a26ba751d5730b4db13429197f29008d77
branch: feature/logging
repository: capProj-2
topic: "Implementation Plan: Consistent Logging with Enhanced Format and Code Comments"
tags: [research, logging, implementation-plan, backend]
status: draft
last_updated: 2026-03-07
last_updated_by: opencode
---

# Consistent Logging Implementation Plan

## Overview

Implement consistent logging across the TRACE backend with enhanced format including filename:method:lineNumber, meaningful decision/value logging, and comprehensive code comments.

## Project Context

**Language/Stack**: Python/FastAPI + scikit-learn
**Test Runner**: pytest
**Test Command**: `python3 -m pytest`

## Current State Analysis

Based on research (see `thoughts/shared/research/2026-03-07-logging-management.md`):

1. **llm_client.py**: Uses Python `logging` module with basic format (no file/method/line)
2. **ml_predictor.py**: Uses `print()` statements instead of logging
3. **main.py**: No logging at all - only traceback.print_exc()

Current format in llm_client.py (line 24):
```
"%(asctime)s [%(levelname)s] %(name)s - %(message)s"
```

## Desired End State

1. **Enhanced Log Format** including:
   - Timestamp
   - Log Level
   - Logger Name
   - **Filename:method:lineNumber**
   - Message with contextual values

2. **Consistent Logging** across all backend files:
   - `main.py` - API request/response logging
   - `ml_predictor.py` - ML pipeline logging
   - `llm_client.py` - Enhanced existing logging
   - New `logging_config.py` - Centralized configuration

3. **Meaningful Log Messages**:
   - Decisions made at each stage
   - Input/output values for debugging
   - Flow traceability (Stage 1, 2, 3, etc.)

4. **Code Comments** explaining complex logic

### Key Discoveries:
- Current logging in llm_client.py:22-27 needs enhancement
- ml_predictor.py has key functions: predict(), run_rules(), run_ml(), combine_scores()
- main.py has two endpoints: health_check() and analyze_claim()
- 6-stage pipeline documented in AGENTS.md

## What We're NOT Doing

- JSON-structured logging for container orchestration (not needed currently)
- Log rotation/persistence (stdout is sufficient for now)
- Logging to database or external services

## Implementation Approach

Create a centralized logging configuration module, then update each file to use proper logging with enhanced format. Use TDD to verify log output format.

---

## Phase 1: Create Centralized Logging Configuration

### Overview
Create a shared logging configuration module with enhanced format and utility functions.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `tests/test_logging_config.py` | `format_includes_filename` | Log output contains filename |
| `tests/test_logging_config.py` | `format_includes_method` | Log output contains function name |
| `tests/test_logging_config.py` | `format_includes_line` | Log output contains line number |
| `tests/test_logging_config.py` | `get_logger_returns_logger` | get_logger() returns valid logger |
| `tests/test_logging_config.py` | `log_decision_includes_values` | Decision logs include input values |

### Changes Required:

#### 1. Create `backend/logging_config.py`
**New File**: `backend/logging_config.py`

```python
"""
TRACE Logging Configuration
---------------------------
Centralized logging setup with enhanced format including
filename, method name, and line number for debugging.
"""

import os
import logging
import sys
from typing import Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

TRACE_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s "
    "%(filename)s:%(funcName)s:%(lineno)d - %(message)s"
)
TRACE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with TRACE format."""
    lvl = (level or LOG_LEVEL).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format=TRACE_FORMAT,
        datefmt=TRACE_DATE_FORMAT,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with TRACE naming."""
    return logging.getLogger(name)


class DecisionLogger:
    """Helper class for logging decisions with context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_stage(self, stage: int, stage_name: str, **kwargs) -> None:
        """Log stage entry with parameters."""
        params = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"[STAGE {stage}] {stage_name} | {params}")

    def log_decision(self, decision_type: str, result: dict, **context) -> None:
        """Log a decision with result values."""
        self.logger.info(
            f"Decision: {decision_type} | Result: {result} | Context: {context}"
        )

    def log_input(self, func_name: str, **inputs) -> None:
        """Log function inputs."""
        self.logger.debug(f"INPUT {func_name} | {inputs}")

    def log_output(self, func_name: str, **outputs) -> None:
        """Log function outputs."""
        self.logger.debug(f"OUTPUT {func_name} | {outputs}")
```

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: test failures for missing format elements
- [x] GREEN achieved: `python3 -m pytest tests/test_logging_config.py -v` passes
- [x] Format validation: `%(filename)s:%(funcName)s:%(lineno)d` appears in logs

#### Manual Verification:
- [x] Run backend and verify log output shows file:method:line format

---

## Phase 2: Update llm_client.py Logging

### Overview
Enhance existing logging in llm_client.py with new format and decision logging.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `tests/test_llm_client_logging.py` | `categorize_notes_logs_stage` | Stage 1 logging present |
| `tests/test_llm_client_logging.py` | `translate_logs_stage` | Stage 3 logging present |
| `tests/test_llm_client_logging.py` | `format_logs_decision` | Decision output logged |

### Changes Required:

#### 1. Update `backend/llm_client.py`
**File**: `backend/llm_client.py`
**Changes**: 
- Import from `logging_config` instead of basicConfig
- Add stage logging for understand_claim, translate_to_ml_features, format_output
- Log decision values (category, confidence, failure_analysis)

```python
# Replace lines 17-27 with:
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger("trace.llm_client")

# Add after categorize_notes docstring (around line 65):
logger.info("[STAGE 1] LLM Understanding | dtc=%s voltage=%s notes_len=%d",
            dtc_code or "none", voltage, len(notes))

# Add after successful categorization:
logger.info("Decision: categorize_notes | category=%s confidence=%.2f",
            category, confidence)
```

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: tests fail without new logging
- [x] GREEN achieved: `python3 -m pytest tests/test_llm_client_logging.py -v` passes

#### Manual Verification:
- [x] API call produces logs with format: `filename:function:line`

---

## Phase 3: Add Logging to ml_predictor.py

### Overview
Replace print() statements with proper logging in ml_predictor.py.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `tests/test_ml_predictor_logging.py` | `predict_logs_stage1` | Stage 1 (LLM) logged |
| `tests/test_ml_predictor_logging.py` | `predict_logs_rules` | Rule engine results logged |
| `tests/test_ml_predictor_logging.py` | `predict_logs_ml` | ML results logged |
| `tests/test_ml_predictor_logging.py` | `predict_logs_combined` | Combined decision logged |

### Changes Required:

#### 1. Update `backend/ml_predictor.py`
**File**: `backend/ml_predictor.py`
**Changes**: 
- Import logging_config (around line 19)
- Replace print() with logger calls
- Add decision logging at key stages

```python
# Add after existing imports (around line 30):
from logging_config import get_logger, DecisionLogger

logger = get_logger("trace.ml_predictor")
decision_logger = DecisionLogger(logger)

# Replace print("[TRACE] Loading dataset...") (line 176):
logger.info("[INIT] Loading dataset from %s", DATA_PATH)

# Replace print("[TRACE] Training models...") (line 204):
logger.info("[INIT] Training ML models...")

# In predict() function (around line 456):
def predict(fault_code: str, technician_notes: str, voltage: float) -> dict:
    """Predict warranty decision with logging."""
    global _bundle
    if _bundle is None:
        _bundle = load_models()
        logger.info("[INIT] Models loaded")

    fc = (fault_code or "").strip()
    notes = (technician_notes or "").strip()
    v = float(voltage) if voltage is not None else None
    
    logger.info("INPUT predict | fault_code=%s voltage=%s notes_len=%d", 
                fc, v, len(notes))

    # Stage 1: LLM Understanding
    llm_stage1 = None
    if llm_available:
        try:
            from llm_client import understand_claim_with_retry
            llm_stage1 = understand_claim_with_retry(notes, fc, v)
            decision_logger.log_decision("Stage 1 LLM", llm_stage1)
        except Exception as e:
            logger.warning("[STAGE 1] LLM failed, using fallback: %s", e)

    # Stage 2: Rule Engine
    rule_result = run_rules(fc, notes, v)
    if rule_result.get("rule_fired"):
        decision_logger.log_decision("Rule Engine", rule_result)
        logger.info("Rule fired: %s with confidence %.1f", 
                    rule_result["rule_id"], rule_result["rule_confidence"])

    # Stage 4: ML Scoring
    ml_result = run_ml(features)
    decision_logger.log_decision("ML Model", {
        "warranty_decision": ml_result["ml_warranty_decision"],
        "failure_analysis": ml_result["ml_failure_analysis"],
        "confidence": ml_result["ml_confidence"],
    })

    # Stage 5: Combine
    combined = combine_scores(rule_result, ml_result, llm_stage1)
    decision_logger.log_decision("Combined", combined)

    logger.info("OUTPUT predict | status=%s confidence=%.1f engine=%s",
                output["status"], output["confidence"], output["decision_engine"])
```

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: tests fail without logging
- [x] GREEN achieved: `python3 -m pytest tests/test_ml_predictor_logging.py -v` passes

#### Manual Verification:
- [x] Smoke test shows proper log format with file:method:line

---

## Phase 4: Add Logging to main.py

### Overview
Add logging to FastAPI endpoints in main.py.

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `tests/test_main_logging.py` | `analyze_endpoint_logs_request` | Request logging |
| `tests/test_main_logging.py` | `analyze_endpoint_logs_response` | Response logging |
| `tests/test_main_logging.py` | `analyze_endpoint_logs_error` | Error logging |

### Changes Required:

#### 1. Update `backend/main.py`
**File**: `backend/main.py`
**Changes**: 
- Import logging_config
- Add request/response logging

```python
# Add after existing imports (around line 13):
from logging_config import get_logger

logger = get_logger("trace.api")

# Update analyze_claim function (around line 58):
@app.post("/analyze", response_model=ClaimResponse)
def analyze_claim(claim: ClaimRequest):
    """Analyze warranty claim with logging."""
    logger.info("REQUEST /analyze | fault_code=%s voltage=%s",
                claim.fault_code, claim.voltage)
    
    try:
        result = ml_predict(
            fault_code=claim.fault_code,
            technician_notes=claim.technician_notes,
            voltage=claim.voltage,
        )
        logger.info("RESPONSE /analyze | status=%s confidence=%.1f engine=%s",
                    result["status"], result["confidence"], 
                    result.get("decision_engine", "unknown"))
        return ClaimResponse(**result)
    except Exception as e:
        logger.error("ERROR /analyze | %s: %s", type(e).__name__, str(e),
                     exc_info=True)
        raise HTTPException(status_code=500, 
                          detail=f"Prediction error: {str(e)}")
```

### Success Criteria:

#### Automated Verification:
- [x] RED confirmed: tests fail without logging
- [x] GREEN achieved: `python3 -m pytest tests/test_main_logging.py -v` passes

#### Manual Verification:
- [x] API call shows request/response in logs

---

## Phase 5: Add Code Comments

### Overview
Add meaningful comments explaining complex logic across all backend files.

### TDD — Tests to Write First

No tests required - this is documentation only.

### Changes Required:

Add docstrings and comments to:
1. `logging_config.py` - Module and class documentation
2. `ml_predictor.py` - Complex functions (combine_scores, run_ml, run_rules)
3. `llm_client.py` - API interaction functions
4. `main.py` - Endpoint documentation

Example comment style:
```python
def combine_scores(rule_result, ml_result, llm_stage1):
    """
    Combine rule engine and ML results using weighted scoring.
    
    Logic:
    1. If rule fires and agrees with ML: apply agreement bonus
    2. If rule fires but disagrees: use threshold-based fallback
    3. If no rule fires: use ML result with possible penalty
    
    Weights defined at module level control the blend.
    """
```

### Success Criteria:

#### Automated Verification:
- [x] All functions have docstrings

#### Manual Verification:
- [x] Code review confirms meaningful comments

---

## Testing Strategy

### Unit Tests

| Test Scope | Location | Purpose |
|------------|----------|---------|
| Logging config | `tests/test_logging_config.py` | Validate format and utilities |
| LLM client | `tests/test_llm_client_logging.py` | Stage logging |
| ML predictor | `tests/test_ml_predictor_logging.py` | Decision logging |
| API | `tests/test_main_logging.py` | Request/response logging |

### Integration Tests
- Full API call: verify log flow shows all stages
- Smoke test: verify log format consistency

### Manual Testing Steps
1. Run backend: `uvicorn main:app --reload --port 8000`
2. Call `/analyze` endpoint
3. Observe log output with file:method:line format

## Performance Considerations

- Minimal overhead from structured logging
- DEBUG level may produce large logs in production
- Consider async logging for high-volume scenarios (not needed now)

## Migration Notes

1. Remove print() statements after verifying logging works
2. Update AGENTS.md to document new LOG_LEVEL options
3. Backup before removing print() statements

## References

- Original research: `thoughts/shared/research/2026-03-07-logging-management.md`
- llm_client.py: current logging implementation
- AGENTS.md: pipeline stages documentation
