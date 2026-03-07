# TRACE - Technical Resolution and Claims Evaluation

## Project Overview

TRACE (Technical Resolution and Claims Evaluation) is a **warranty claim analysis system** for automotive electronics. It uses a hybrid approach combining **rule-based logic**, **machine learning**, and optionally **large language models (LLM)** to evaluate warranty claims and determine whether they should be approved or rejected.

The system analyzes technician observations, fault codes (DTCs), and voltage readings to predict:
- **Root cause** of the failure
- **Warranty coverage decision** (Production Failure, Customer Failure, or According to Specification)
- **Confidence level** in the decision

---

## What This Project Does

TRACE automates the warranty claim evaluation process that would traditionally require expert automotive technicians. When a vehicle comes in for warranty repair, the technician records:

1. **Fault Codes (DTCs)** - Diagnostic Trouble Codes from the vehicle's onboard computer
2. **Technician Notes** - Free-text observations about the vehicle's condition
3. **Voltage Reading** - Measured electrical voltage at the time of diagnosis

TRACE processes these three inputs through a multi-stage pipeline and produces a structured decision with:

| Output Field | Description |
|--------------|-------------|
| `status` | Final decision: **Approved**, **Rejected**, or **Needs Manual Review** |
| `failure_analysis` | Predicted root cause of the failure |
| `warranty_decision` | Classification: **Production Failure**, **Customer Failure**, or **According to Specification** |
| `confidence` | How certain the system is (0-100%) |
| `reason` | Human-readable explanation of the decision |
| `matched_complaint` | How the technician notes were interpreted |
| `decision_engine` | Which components contributed to the decision |

---

## What Is Expected From the User

### Input Requirements

Users (typically service technicians or warranty administrators) must provide:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `fault_code` | String | DTC code(s), comma-separated | `"P0562"` or `"U0100, P0301"` |
| `technician_notes` | String | Free-text observations from the technician | `"Engine overheating, low idle"` |
| `voltage` | Float | Measured voltage reading (V) | `14.2` |

### Expected User Actions

1. **Collect diagnostic information** - Retrieve fault codes from the vehicle's OBD system
2. **Document observations** - Write clear notes about symptoms and findings
3. **Take voltage reading** - Measure system voltage during diagnosis
4. **Submit claim** - Send data to the TRACE API endpoint
5. **Review decision** - Read the system's analysis and confidence level
6. **Manual review** - If confidence is below threshold, perform manual evaluation

---

## What Is Expected From the Application

TRACE must process incoming claims and produce accurate, reliable decisions. The application is expected to:

### 1. Accept and Validate Input

- Accept JSON payloads via REST API at `/analyze` endpoint
- Handle missing or invalid data gracefully
- Return appropriate error messages for malformed requests

### 2. Run Multi-Stage Analysis

The system processes claims through **six stages**:

```
Input (fault_code, technician_notes, voltage)
    │
    ▼
[STAGE 1] LLM — Semantic Understanding
    │  Categorize notes (moisture_damage, physical_damage, etc.)
    ▼
[STAGE 2] Rule Engine — Structured Decision
    │  Check against domain rules (voltage thresholds, keywords, DTC prefixes)
    ▼
[STAGE 3] LLM — ML Feature Translation
    │  Convert raw data to ML-ready features
    ▼
[STAGE 4] ML — Confidence Scoring
    │  Run RandomForest classifiers for failure analysis and warranty decision
    ▼
[STAGE 5] Score Combiner
    │  Blend rule + ML results with weighted averaging
    ▼
[STAGE 6] LLM — Output Formatter
    │  Generate human-readable response
    ▼
ClaimResponse JSON
```

### 3. Produce Reliable Decisions

- **Rule-based predictions** take precedence when domain rules clearly apply
- **ML predictions** provide fallback and additional confidence scoring
- **LLM enhancement** (when available) improves semantic understanding and output quality

### 4. Calculate Confidence Accurately

The system must compute confidence scores that reflect true certainty. For details on the confidence calculation logic, see [CONFIDENCE_CALCULATION.md](./CONFIDENCE_CALCULATION.md).

---

## Understanding the Results

### Status Values

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| **Approved** | Claim qualifies for warranty coverage | Process claim normally |
| **Rejected** | Claim denied (customer responsibility) | Do not process under warranty |
| **Needs Manual Review** | System cannot make confident decision | Human technician must evaluate |

### Warranty Decision Values

| Decision | Description | Maps To |
|----------|-------------|---------|
| **Production Failure** | Defect in manufacturing or component failure | **Approved** |
| **According to Specification** | Vehicle operating within normal parameters | **Approved** |
| **Customer Failure** | Damage caused by user misuse or external factors | **Rejected** |

### Confidence Score Interpretation

| Confidence | Interpretation |
|------------|----------------|
| ≥ 85% | High certainty - firm decision (auto-approve/reject) |
| 65-84% | Moderate certainty - decision made but review recommended |
| < 65% | Low certainty - manual review required |

### Decision Engine Indicators

The `decision_engine` field tells you which components contributed:

| Value | Meaning |
|-------|---------|
| `LLM+Rule+ML` | All three systems participated (highest capability) |
| `Rule+ML` | Rule engine and ML model (fallback without LLM) |
| `ML` | Only ML model (no rules matched, no LLM) |

---

## End-to-End Flow Example

### Step 1: Technician Submits Claim

```
POST /analyze
{
  "fault_code": "P0562",
  "technician_notes": "Engine overheating, low idle",
  "voltage": 14.2
}
```

### Step 2: System Processes Claim

1. **Stage 1 (LLM)**: Categorizes notes as `engine_symptom`, identifies complaint as "Engine overheating"
2. **Stage 2 (Rules)**: No rules match (voltage 14.2 is normal, no keywords)
3. **Stage 3 (LLM/Fallback)**: Extracts features: complaint="Engine overheating", dtc_count=1, has_P=1
4. **Stage 4 (ML)**: RandomForest predicts "Production Failure" with 72% probability
5. **Stage 5 (Combine)**: No rule fired, so uses ML result → 72% confidence
6. **Stage 6 (LLM/Fallback)**: Assembles human-readable output

### Step 3: Response Returned

```json
{
  "status": "Approved",
  "failure_analysis": "Battery voltage low causing system stress",
  "warranty_decision": "Production Failure",
  "confidence": 85.0,
  "reason": "Low system voltage detected during operation indicates component failure rather than customer misuse",
  "matched_complaint": "Engine overheating",
  "decision_engine": "Rule+ML"
}
```

### Step 4: Technician Reviews

- Status: **Approved** → Warranty claim can be processed
- Confidence: **85%** → Firm decision, no manual review needed
- Root cause: **Battery voltage low** → Technician knows what to address

---

## Key Components

### Backend (FastAPI)

| File | Purpose |
|------|---------|
| `main.py` | API endpoints (`/`, `/analyze`) |
| `ml_predictor.py` | Hybrid rule + ML engine, confidence calculation |
| `llm_client.py` | OpenRouter LLM integration |
| `synthetic_warranty_claims_v2.csv` | Training dataset (12,000 records) |
| `trace_models.pkl` | Trained RandomForest models |

### Frontend (HTML)

| File | Purpose |
|------|---------|
| `index.html` | Single-page interface for submitting claims |

---

## Rule Engine Reference

The rule engine contains 9 predefined rules based on automotive domain expertise:

| Rule ID | Condition | Confidence | Decision |
|---------|-----------|------------|----------|
| `over_voltage` | Voltage > 16V | 94.0% | Customer Failure |
| `moisture` | Keywords: water, moisture, wet... | 91.0% | Customer Failure |
| `physical_damage` | Keywords: crack, broken, impact... | 88.5% | Customer Failure |
| `u_code` | DTC starts with U (CAN bus) | 85.0% | Production Failure |
| `low_voltage` | Voltage < 11V | 83.0% | Production Failure |
| `ntf` | Keywords: no fault, NTF... | 82.0% | According to Specification |
| `p_code_engine` | P0xxx + engine symptoms | 80.5% | Production Failure |
| `c_code` | DTC starts with C | 78.0% | Production Failure |
| `b_code` | DTC starts with B | 76.0% | Production Failure |

---

## System Requirements

### Runtime Requirements

- **Python 3.9+**
- **FastAPI** - Web framework
- **scikit-learn** - ML models (RandomForest)
- **OpenRouter API Key** - For LLM enhancement (optional but recommended)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes (for LLM) | API key for OpenRouter LLM service |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

---

## Running the System

### Quick Start

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY="your-api-key-here"

# Run server
uvicorn main:app --reload --port 8000
```

### Docker Deployment

```bash
docker-compose up --build
```

This starts:
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

### Testing

```bash
# Smoke test
python3 -c "from ml_predictor import predict; print(predict('P0562', 'Engine overheating', 14.2))"

# Run all tests
python3 -m pytest
```

---

## Glossary

| Term | Definition |
|------|------------|
| **DTC** | Diagnostic Trouble Code - vehicle fault code (e.g., P0562, U0100) |
| **EOS** | Electrical Over Stress - damage from excessive voltage |
| **NTF** | No Trouble Found - vehicle operates within specifications |
| **RandomForest** | Ensemble ML algorithm using multiple decision trees |
| **OpenRouter** | Service providing access to various LLM models |
| **Confidence** | System's certainty in its decision (0-100%) |

---

## Additional Documentation

- [CONFIDENCE_CALCULATION.md](./CONFIDENCE_CALCULATION.md) - Detailed confidence algorithm
- [AGENTS.md](./AGENTS.md) - Developer notes and workflow
