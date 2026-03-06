"""
TRACE Backend API  —  FastAPI + ML Predictor Integration
---------------------------------------------------------
Run:
    pip install fastapi uvicorn scikit-learn scipy pandas
    uvicorn main:app --reload --port 8000
"""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

load_dotenv()

# Load predictor (trains model on first import if no pickle exists)
#from ml_predictor import predict as ml_predict
from ml_predictor import predict as ml_predict

app = FastAPI(title="TRACE Backend API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    fault_code:         str
    technician_notes:   str
    voltage:            float

class ClaimResponse(BaseModel):
    status:             str     # Approved | Rejected | Needs Manual Review
    failure_analysis:   str     # Root cause prediction
    warranty_decision:  str     # Production Failure | Customer Failure | According to Specification
    confidence:         float   # 0-100
    reason:             str     # Human-readable explanation
    matched_complaint:  str     # How technician notes were interpreted
    decision_engine:    str     # "Rule-based" | "ML model"

# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"message": "TRACE Backend Running ✅", "version": "2.0"}


@app.post("/analyze", response_model=ClaimResponse)
def analyze_claim(claim: ClaimRequest):
    """
    Accepts warranty claim inputs from the TRACE frontend,
    routes them through the ML predictor (hybrid rule + RandomForest),
    and returns a structured warranty decision.
    """
    try:
        result = ml_predict(
            fault_code        = claim.fault_code,
            technician_notes  = claim.technician_notes,
            voltage           = claim.voltage,
        )
        return ClaimResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
