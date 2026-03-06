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
│   ├── main.py              # API endpoints
│   ├── ml_predictor.py      # ML predictor (RandomForest)
│   ├── ml_predictor_DecisionTree.py  # Alternative ML model
│   ├── requirements.txt     # Python dependencies
│   ├── Warranty_Dataset_2019_2024_12000.csv
│   └── trace_models.pkl     # Trained models (generated)
├── frontend/
│   └── index.html           # Single-page frontend
├── docker-compose.yml       # Full stack deployment
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
pip install ruff black isort pytest
```

### Backend

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run the API server (development)
uvicorn main:app --reload --port 8000

# Run with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

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
