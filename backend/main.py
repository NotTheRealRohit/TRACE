"""
TRACE Backend API  —  FastAPI + ML Predictor Integration
---------------------------------------------------------
Run:
    pip install fastapi uvicorn scikit-learn scipy pandas
    uvicorn main:app --reload --port 8000
"""

import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logging_config import get_logger
from ml_predictor import predict as ml_predict

logger = get_logger("trace.api")

load_dotenv()

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
    logger.info("REQUEST /analyze | fault_code=%s voltage=%s",
                claim.fault_code, claim.voltage)
    
    try:
        result = ml_predict(
            fault_code        = claim.fault_code,
            technician_notes  = claim.technician_notes,
            voltage           = claim.voltage,
        )
        logger.info("RESPONSE /analyze | status=%s confidence=%.1f engine=%s",
                    result["status"], result["confidence"], 
                    result.get("decision_engine", "unknown"))
        return ClaimResponse(**result)
    except Exception as e:
        logger.error("ERROR /analyze | %s: %s", type(e).__name__, str(e),
                     exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
