# TRACE - Technical Resolution and Claims Evaluation

A warranty claim analysis system with a hybrid rule-based + ML engine for evaluating automotive warranty claims.

## Overview

TRACE uses FastAPI for the backend with scikit-learn ML models and OpenRouter LLM integration for intelligent claim analysis. The frontend is a vanilla HTML interface served via nginx.

## Prerequisites

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
- **Python 3.11+** (for local development without Docker)
- **OpenRouter API Key** (see below)

## Getting Your OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Create a free account or sign in
3. Navigate to **API Keys** in your dashboard
4. Create a new API key
5. Copy the key (you won't be able to see it again)

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd capProj-2
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your-api-key-here
LOG_LEVEL=INFO
```

> **Note**: The `.env` file is gitignored and should never be committed to version control.

## Running with Docker

### Build and Start All Services

```bash
docker-compose up --build
```

This will start:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000

### Start Only Backend

```bash
docker-compose up backend
```

### View Logs

```bash
docker-compose logs -f
```

### Stop Services

```bash
docker-compose down
```

## Running Locally (without Docker)

### 1. Create a Virtual Environment

```bash
cd backend
python -m venv venv
```

### 2. Activate the Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

**Linux/macOS:**
```bash
export OPENROUTER_API_KEY=your-api-key-here
export LOG_LEVEL=INFO
```

**Windows (CMD):**
```cmd
set OPENROUTER_API_KEY=your-api-key-here
set LOG_LEVEL=INFO
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your-api-key-here"
$env:LOG_LEVEL="INFO"
```

### 5. Start the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at http://localhost:8000

## Testing

### Run All Tests

```bash
cd backend
python -m pytest
```

### Run a Specific Test

```bash
python -m pytest -k "test_name"
```

### Run with Verbose Output

```bash
python -m pytest -v
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "fault_code": "P0562",
    "technician_notes": "Engine overheating",
    "voltage": 14.2
  }'
```

### Response Example

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

## Project Structure

```
capProj-2/
├── backend/
│   ├── main.py              # FastAPI endpoints
│   ├── ml_predictor.py     # ML predictor + rule engine
│   ├── llm_client.py       # OpenRouter LLM client
│   ├── requirements.txt    # Python dependencies
│   ├── .env                 # Environment variables (gitignored)
│   └── tests/               # Unit tests
├── frontend/
│   └── index.html           # Single-page frontend
├── docker-compose.yml       # Docker orchestration
└── README.md                # This file
```

## Troubleshooting

### OpenRouter API Key Issues

If you see errors about missing API keys, ensure:
1. The `OPENROUTER_API_KEY` is set in your environment
2. The key is valid and has not expired
3. You have sufficient credits in your OpenRouter account

### Port Already in Use

If port 8000 or 3000 is already in use, modify `docker-compose.yml` to use different ports:

```yaml
ports:
  - "8001:8000"  # Host:Container
```

### Model Training Errors

If the ML model fails to train, check:
1. The dataset file exists: `backend/synthetic_warranty_claims_v2.csv`
2. You have sufficient disk space
3. Python dependencies installed correctly

## License

MIT License
