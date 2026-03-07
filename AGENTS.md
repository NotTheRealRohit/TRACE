# AGENTS.md — TRACE Project

## Project Overview

- **Project**: TRACE (Technical Resolution and Claims Evaluation)
- **Stack**: FastAPI (Python) + scikit-learn ML + vanilla HTML frontend
- **Purpose**: Warranty claim analysis with hybrid rule-based + ML engine

---

## Directory Structure

```
/
├── backend/                 # FastAPI application
│   ├── main.py              # API endpoints (v2.0)
│   ├── ml_predictor.py      # ML predictor (RandomForest) with LLM integration
│   ├── ml_predictor_DecisionTree.py  # Alternative ML model
│   ├── llm_client.py        # OpenRouter LLM client for note categorization
│   ├── logging_config.py    # Centralized logging configuration
│   ├── requirements.txt    # Python dependencies
│   ├── synthetic_warranty_claims_v2.csv  # Training dataset (12K rows)
│   ├── trace_models.pkl     # Trained models (generated)
│   ├── .env                 # Environment variables (API keys)
│   ├── .env.example         # Example environment template
│   ├── backup/              # Backup of previous versions
│   └── tests/               # Unit and integration tests
├── frontend/
│   └── index.html           # Single-page frontend
├── docker-compose.yml       # Full stack deployment
├── .opencode/               # OpenCode configuration and agents
└── AGENTS.md               # This file
```

---

## Build, Run & Test Commands

### Environment Notes

This project uses **pyenv** for Python version management. Some commands may require specific invocations:

| Command | Direct | Via Python Module | Notes |
|---------|--------|------------------|-------|
| Python | `python` or `python3` | `python3 -m pytest` | Use `python3` to avoid ambiguity |
| pytest | `pytest` (if in PATH) | `python3 -m pytest` | Preferred: use module invocation |
| ruff | `ruff` (if installed) | N/A | Install via: `pip install ruff` |
| black | `black` (if installed) | `python3 -m black` | Install via: `pip install black` |

**Installation:**
```bash
# Install linting tools (if not already installed)
pip install ruff black isort pytest python-dotenv
```

### Backend

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Set OpenRouter API key (required for LLM features)
export OPENROUTER_API_KEY="your-api-key-here"

# Run the API server (development)
uvicorn main:app --reload --port 8000

# Run with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes (for LLM features) | API key for OpenRouter LLM service |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

**Note:** Create a `.env` file in `backend/` directory with your API keys. See `.env.example` for the template.

### Run Single Test / Smoke Test

```bash
# Run the built-in smoke tests in ml_predictor.py
cd backend
python3 -c "from ml_predictor import predict; print(predict('P0562', 'Engine overheating', 14.2))"

# Or run the main module's test block
python3 ml_predictor.py

# Run all tests (use module invocation to avoid PATH issues)
python3 -m pytest

# Run specific test
python3 -m pytest -k "test_name"

# Run with verbose output
python3 -m pytest -v

# Run logging-specific tests
python3 -m pytest backend/tests/test_logging_config.py -v
python3 -m pytest backend/tests/test_ml_predictor_logging.py -v
python3 -m pytest backend/tests/test_main_logging.py -v
python3 -m pytest backend/tests/test_llm_client_logging.py -v

# Run core functionality tests
python3 -m pytest backend/tests/test_ml_predictor.py -v
python3 -m pytest backend/tests/test_llm_client.py -v
python3 -m pytest backend/tests/test_e2e.py -v
```

### API Testing

```bash
# Health check
curl http://localhost:8000/

# Analyze claim
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"fault_code": "P0562", "technician_notes": "Engine overheating", "voltage": 14.2}'
```

### Docker

```bash
# Start full stack
docker-compose up --build

# Start only backend
docker-compose up backend
```

### Linting & Code Quality

```bash
# Install linting tools (if not already installed)
pip install ruff black isort

# Format code (in-place)
python3 -m black backend/

# Sort imports
python3 -m isort backend/

# Lint
ruff check backend/

# All-in-one (format + lint)
python3 -m black backend/ && python3 -m isort backend/ && ruff check backend/
```

---

## Code Style Guidelines

### General

- Python 3.9+ type hints required for all function signatures
- Use `pydantic` models for all API request/response schemas
- Follow FastAPI best practices (dependency injection, async where appropriate)

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables | snake_case | `fault_code`, `technician_notes` |
| Functions | snake_case | `extract_dtc_features()`, `predict()` |
| Classes | PascalCase | `ClaimRequest`, `ClaimResponse` |
| Constants | UPPER_SNAKE | `BASE_DIR`, `MODEL_PATH` |
| Modules | snake_case | `ml_predictor.py` |

### Imports

```python
# Stdlib first, then third-party, then local
import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

from ml_predictor import predict as ml_predict  # local imports last
```

- Use explicit imports (no `from x import *`)
- Group imports with blank lines: stdlib, third-party, local
- Sort within groups alphabetically

### Formatting

- **Line length**: 100 characters max (120 if necessary)
- **Indentation**: 4 spaces (no tabs)
- **Blank lines**: Two between top-level definitions, one between functions
- **Strings**: Use f-strings for interpolation, double quotes for static strings

### Type Hints

```python
# Required for all public functions
def extract_dtc_features(dtc_str: str) -> dict:
    ...

def predict(fault_code: str, technician_notes: str, voltage: float) -> dict:
    ...

# Use Pydantic for API schemas
class ClaimRequest(BaseModel):
    fault_code: str
    technician_notes: str
    voltage: float
```

### Error Handling

- Use `try/except` blocks with specific exception types
- Always log or print traceback for debugging
- Return proper HTTP status codes via `HTTPException`

```python
try:
    result = ml_predict(...)
    return ClaimResponse(**result)
except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
```

### ML/Data Code

- Suppress warnings appropriately: `warnings.filterwarnings("ignore")`
- Use meaningful variable names for model components
- Include accuracy metrics in training output
- Save/load models with `pickle`

### File Headers

Include docstrings for modules and complex functions:

```python
"""
TRACE ML Predictor  —  Hybrid Rule + ML Engine
------------------------------------------------
Description of what this module does.
...
"""
```

### Git Conventions

- Commit messages: present tense, imperative mood ("Add endpoint" not "Added endpoint")
- Branch naming: `feature/description`, `fix/description`
- No secrets in code (use environment variables)

---

## API Contract

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/analyze` | Analyze warranty claim |

### Request Schema (ClaimRequest)

```python
{
    "fault_code": "P0562",           # DTC code(s), comma-separated
    "technician_notes": "string",    # Free-text observations
    "voltage": 12.5                  # Measured voltage
}
```

### Response Schema (ClaimResponse)

```python
{
    "status": "Approved" | "Rejected" | "Needs Manual Review",
    "failure_analysis": "Root cause prediction",
    "warranty_decision": "Production Failure" | "Customer Failure" | "According to Specification",
    "confidence": 85.0,             # 0-100
    "reason": "Human-readable explanation",
    "matched_complaint": "Engine overheating",
    "decision_engine": "Rule-based" | "ML model"
}
```

---

## Common Tasks

### Add a new rule-based check

Edit `RULES` list in `ml_predictor.py`:

```python
{
    "id": "your_rule_id",
    "match": lambda fc, notes, v: <condition>,
    "failure_analysis": "...",
    "warranty_decision": "...",
    "status": "Approved" | "Rejected",
    "confidence": 80.0,
    "reason": "...",
},
```

### Add new API endpoint

1. Define Pydantic model in `main.py` schemas section
2. Add endpoint function with `@app.post()` decorator
3. Return appropriate response model

### Retrain ML model

```bash
cd backend
python3 -c "from ml_predictor import train_and_save; train_and_save()"
```

Or simply delete `trace_models.pkl` and restart the server — it auto-trains on startup.

### Using Logging

Import and use the logging configuration in any module:

```python
from logging_config import setup_logging, get_logger, DecisionLogger

# Initialize logging at application startup
setup_logging()  # Uses LOG_LEVEL env var, defaults to INFO

# Get a logger for your module
logger = get_logger("trace.module")

# Use DecisionLogger for structured logging
decision_logger = DecisionLogger(logger)
decision_logger.log_stage(1, "Processing", fault_code="P0562")
decision_logger.log_decision("ML", {"status": "Approved"}, confidence=85.0)
```

**Log Format:** `%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(funcName)s:%(lineno)d - %(message)s`

### LLM Integration Architecture

The prediction pipeline uses a 6-stage hybrid approach:

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

**Stage Details:**

1. **STAGE 1 - LLM Understanding** (`llm_client.py::understand_claim()`)
   - Analyzes claim semantically via OpenRouter
   - Categorizes: moisture_damage, physical_damage, ntf, electrical_issue, engine_symptom, communication_fault, other
   - Returns: category, normalized_complaint, severity, failure_analysis, reasoning, confidence

2. **STAGE 2 - Rule Engine** (`ml_predictor.py::run_rules()`)
   - Voltage thresholds (over-voltage >16V, under-voltage <11V)
   - Keyword detection (moisture, physical damage, NTF)
   - DTC prefix analysis (P=Powertrain, U=Network, C=Chassis, B=Body)

3. **STAGE 3 - Feature Translation** (`llm_client.py::translate_to_ml_features()`)
   - Fallback: `match_complaint()` + `extract_dtc_features()` when LLM unavailable
   - Extracts: customer_complaint, dtc_codes, dtc_text, dtc_count, voltage, has_P/U/C/B

4. **STAGE 4 - ML Scoring** (`ml_predictor.py::run_ml()`)
   - RandomForest classifier (always runs)
   - Returns: ml_warranty_decision, ml_failure_analysis, ml_confidence

5. **STAGE 5 - Score Combiner** (`ml_predictor.py::combine_scores()`)
   - Blends rule + ML results
   - Sets decision_engine: "LLM+Rule+ML", "Rule+ML", or "ML"

6. **STAGE 6 - Output Formatter** (`llm_client.py::format_output()`)
   - Fallback: `assemble_output_from_fields()` when LLM unavailable
   - Returns final ClaimResponse JSON

**API Key:** Stored in `backend/.env` as `OPENROUTER_API_KEY`
