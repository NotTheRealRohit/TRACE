---
date: 2026-03-06T12:00:00+00:00
researcher: opencode
git_commit: 2e23b01240c2e79549fa93b67cfd78bd7f43c930
branch: feature/openrouter-llm-integrate
repository: capProj-2
topic: "Local Python environment setup for TRACE project"
tags: [research, setup, python, fastapi, ml]
status: complete
last_updated: 2026-03-06
last_updated_by: opencode
---

# Research: Local Python Environment Setup for TRACE Project

**Date**: 2026-03-06T12:00:00+00:00
**Researcher**: opencode
**Git Commit**: 2e23b01240c2e79549fa93b67cfd78bd7f43c930
**Branch**: feature/openrouter-llm-integrate
**Repository**: capProj-2

## Research Question

How to create a local Python environment to run the TRACE project, understand what's currently present, and how the code runs.

## Summary

The TRACE (Technical Resolution and Claims Evaluation) project is a warranty claim analysis system built with FastAPI and scikit-learn. It uses a hybrid approach combining rule-based logic with machine learning to analyze automotive warranty claims. The project structure includes a backend API, static HTML frontend, and Docker Compose orchestration. Setting up locally requires creating a Python virtual environment, installing dependencies, and running the uvicorn server.

## Detailed Findings

### Project Architecture

The TRACE project follows a simple two-tier architecture:

- **Backend**: FastAPI application providing REST API endpoints
- **Frontend**: Static HTML served by nginx (or directly in development)
- **Orchestration**: Docker Compose for containerized deployment

### Core Components

#### Backend API (`backend/main.py`)
- FastAPI application running on port 8000
- Two main endpoints:
  - `GET /` - Health check endpoint (line 55-57)
  - `POST /analyze` - Claim analysis endpoint (line 60-76)
- Uses Pydantic models for request/response validation
- CORS middleware enabled for all origins

#### ML Predictor (`backend/ml_predictor.py`)
- Hybrid rule-based + RandomForest ML engine
- Contains domain-specific automotive rules (lines 42-150+)
- Rules cover scenarios like:
  - Over-voltage detection (>16V)
  - Low voltage detection (<11V)
  - Moisture/water damage
  - Physical damage
  - NTF (No Trouble Found)
  - U-series DTC codes (communication faults)
- Auto-trains model on first import if no pickle file exists
- Uses TF-IDF vectorization for text features

#### Dataset (`backend/Warranty_Dataset_2019_2024_12000.csv`)
- 12,000 synthetic warranty claims (2019-2024)
- Balanced dataset with feature-target correlations

#### Configuration
- Environment variables loaded from `backend/.env`
- Contains `OPENROUTER_API_KEY` for LLM integration

### Dependencies

Located in `backend/requirements.txt`:
- fastapi==0.111.0
- uvicorn[standard]==0.30.1
- pydantic==2.7.1
- scikit-learn==1.5.0
- scipy==1.13.1
- numpy==1.26.4
- pandas==2.2.2
- requests>=2.31.0
- python-dotenv>=1.0.0

### Running the Application

#### Local Development Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

5. Access the API:
   - Health check: http://localhost:8000/
   - Analyze endpoint: POST http://localhost:8000/analyze

#### Testing the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"fault_code": "P0562", "technician_notes": "Engine overheating", "voltage": 14.2}'
```

#### Docker Deployment

```bash
# Full stack
docker-compose up --build

# Backend only
docker-compose up backend
```

### Code Flow

1. Client sends POST request to `/analyze` with:
   - `fault_code`: DTC code(s) (e.g., "P0562")
   - `technician_notes`: Free-text observations
   - `voltage`: Measured voltage reading

2. FastAPI validates request against `ClaimRequest` schema

3. `ml_predict` function is called with the three parameters

4. Inside `ml_predict`:
   - First, rule-based checks are evaluated (lines 42-150)
   - If a rule matches, high-confidence decision is returned
   - If no rule fires, ML model provides probability distribution

5. Response is formatted as `ClaimResponse` with:
   - `status`: Approved | Rejected | Needs Manual Review
   - `failure_analysis`: Root cause prediction
   - `warranty_decision`: Production Failure | Customer Failure | According to Specification
   - `confidence`: 0-100
   - `reason`: Human-readable explanation
   - `matched_complaint`: How technician notes were interpreted
   - `decision_engine`: "Rule-based" | "ML model"

## Code References

- `backend/main.py:55-57` - Health check endpoint
- `backend/main.py:60-76` - Analyze claim endpoint
- `backend/main.py:37-49` - Pydantic request/response schemas
- `backend/ml_predictor.py:42-150` - Rule-based decision engine
- `backend/ml_predictor.py:30-34` - Model and data paths
- `backend/requirements.txt` - Python dependencies

## Architecture Insights

1. **Hybrid Decision Engine**: The system prioritizes rule-based decisions (high confidence) over ML predictions. ML is used as a fallback when no rules match.

2. **Auto-training**: Models are automatically trained on first import if `trace_models.pkl` doesn't exist.

3. **Simple Deployment**: No database required - uses CSV for data storage and pickle for model persistence.

4. **CORS Open**: Backend allows all origins (`allow_origins=["*"]`), suitable for development but needs restriction in production.

5. **API-First Design**: Clear separation between frontend (static HTML) and backend (REST API).

## Historical Context (from thoughts/)

- `thoughts/shared/plans/2026-03-06-openrouter-llm-integration.md` - Contains plans for integrating OpenRouter LLM into the project

## Related Research

- `thoughts/shared/research/2026-03-06-rule-engine-technician-notes.md` - Previous research on rule engine handling of technician notes

## Open Questions

1. What specific LLM integration is planned in the `feature/openrouter-llm-integrate` branch?
2. How does the DecisionTree model (`ml_predictor_DecisionTree.py`) differ from the RandomForest implementation?
3. What testing coverage exists for the ML predictor?
