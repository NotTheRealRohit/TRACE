"""
OpenRouter LLM Client for TRACE Warranty Claims
-----------------------------------------------
Handles API calls to OpenRouter for technician note categorization.
"""

import os
import json
import time
import logging
import requests
from typing import Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "arcee-ai/trinity-large-preview:free"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("trace.llm_client")
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in environment")
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


def categorize_notes(notes: str, dtc_code: str, voltage: Optional[float], timeout: int = 30) -> dict:
    """
    Call OpenRouter LLM to categorize technician notes.

    Args:
        notes: Technician's free-text notes
        dtc_code: Fault code (e.g., "P0562")
        voltage: Measured voltage reading
        timeout: Request timeout in seconds (default 30)

    Returns:
        dict with keys: category, confidence, failure_analysis, reasoning

    Raises:
        RuntimeError: If API call fails
    """
    api_key = get_api_key()

    logger.info(
        "Categorizing notes | dtc=%s voltage=%s notes_len=%d",
        dtc_code or "none",
        voltage if voltage is not None else "N/A",
        len(notes),
    )
    logger.debug("Full technician notes: %s", notes)

    prompt = CATEGORIZATION_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        voltage=voltage if voltage is not None else "not provided",
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    t0 = time.monotonic()
    try:
        logger.debug("Sending request to OpenRouter | model=%s timeout=%ds", MODEL, timeout)
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.error("OpenRouter request timed out after %ds", timeout)
        raise RuntimeError("OpenRouter API request timed out")
    except requests.RequestException as e:
        logger.error("OpenRouter request failed: %s", str(e))
        raise RuntimeError(f"OpenRouter API request failed: {str(e)}")

    elapsed = time.monotonic() - t0
    logger.debug("OpenRouter responded in %.2fs | status=%d", elapsed, response.status_code)

    if response.status_code == 429:
        logger.warning("Rate limited by OpenRouter (429)")
        raise RuntimeError("Rate limited by OpenRouter, try again later")

    if response.status_code != 200:
        logger.error(
            "OpenRouter API error | status=%d body=%s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    logger.debug("Raw LLM response: %s", content)

    try:
        parsed = json.loads(content)
        category = parsed.get("category", "other")
        confidence = parsed.get("confidence", 0.8)
        logger.info(
            "Categorization complete | category=%s confidence=%.2f elapsed=%.2fs",
            category,
            confidence,
            elapsed,
        )
        return {
            "category": category,
            "confidence": confidence,
            "failure_analysis": parsed.get("failure_analysis", "Unknown"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", content)
        return None


FORMAT_OUTPUT_PROMPT = """You are a warranty claims report writer. Given the structured decision below,
write a clear, professional output for a technician to read.

Decision Data:
{combined_json}

Rules:
- status must be EXACTLY: "Approved", "Rejected", or "Needs Manual Review"
- warranty_decision must be EXACTLY one of:
    "Production Failure", "Customer Failure", "According to Specification"
- failure_analysis: synthesize llm_failure_analysis and ml_failure_analysis
  into one concise root cause sentence (max 20 words)
- reason: 1–2 sentences explaining the decision in plain language
- matched_complaint: use customer_complaint from features
- confidence: use combined_confidence exactly as provided (do not change)
- decision_engine: use as provided

Respond ONLY with this JSON:
{{
  "status": "...",
  "failure_analysis": "...",
  "warranty_decision": "...",
  "confidence": 0.0,
  "reason": "...",
  "matched_complaint": "...",
  "decision_engine": "..."
}}
"""


def format_output(combined: dict, features: dict, timeout: int = 30) -> dict | None:
    """
    Format the combined decision into human-readable output using LLM.

    Args:
        combined: Output from combine_scores()
        features: Output from translate_to_ml_features()
        timeout: Request timeout in seconds (default 30)

    Returns:
        dict with keys: status, failure_analysis, warranty_decision, confidence, reason, matched_complaint, decision_engine
        None if API call fails
    """
    api_key = get_api_key()

    logger.info("Formatting output | decision_engine=%s", combined.get("decision_engine", "unknown"))

    prompt = FORMAT_OUTPUT_PROMPT.format(
        combined_json=json.dumps(combined),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    t0 = time.monotonic()
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.error("OpenRouter request timed out after %ds", timeout)
        return None
    except requests.RequestException as e:
        logger.error("OpenRouter request failed: %s", str(e))
        return None

    elapsed = time.monotonic() - t0
    logger.debug("OpenRouter responded in %.2fs | status=%d", elapsed, response.status_code)

    if response.status_code != 200:
        logger.error("OpenRouter API error | status=%d", response.status_code)
        return None

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        return {
            "status": parsed.get("status", "Needs Manual Review"),
            "failure_analysis": parsed.get("failure_analysis", combined.get("ml_failure_analysis", "Unknown")),
            "warranty_decision": parsed.get("warranty_decision", combined.get("warranty_decision", "")),
            "confidence": parsed.get("confidence", combined.get("combined_confidence", 50.0)),
            "reason": parsed.get("reason", ""),
            "matched_complaint": parsed.get("matched_complaint", features.get("customer_complaint", "OBD Light ON")),
            "decision_engine": parsed.get("decision_engine", combined.get("decision_engine", "ML")),
        }
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", content)
        return None


UNDERSTAND_CLAIM_PROMPT = """You are an automotive warranty analyst. Analyze the claim below and respond ONLY with JSON.

Technician Notes: {notes}
DTC Code: {dtc_code}
Measured Voltage: {voltage}

Classify into EXACTLY ONE category from this list:
  moisture_damage, physical_damage, ntf, electrical_issue,
  engine_symptom, communication_fault, other

Also provide:
- normalized_complaint: one of these exact strings:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- severity: "low" | "medium" | "high"
- failure_analysis: short root cause string (max 15 words)
- reasoning: brief explanation (max 30 words)
- confidence: float 0.0–1.0

Respond ONLY with this JSON structure, no preamble:
{{
  "category": "...",
  "normalized_complaint": "...",
  "severity": "...",
  "failure_analysis": "...",
  "reasoning": "...",
  "confidence": 0.0
}}
"""


def understand_claim(notes: str, dtc_code: str, voltage: Optional[float], timeout: int = 30) -> dict | None:
    """
    Call OpenRouter LLM to perform semantic understanding of the claim.

    Args:
        notes: Technician's free-text notes
        dtc_code: Fault code (e.g., "P0562")
        voltage: Measured voltage reading
        timeout: Request timeout in seconds (default 30)

    Returns:
        dict with keys: category, normalized_complaint, severity, failure_analysis, reasoning, confidence
        None if API call fails
    """
    api_key = get_api_key()

    logger.info(
        "Understanding claim | dtc=%s voltage=%s notes_len=%d",
        dtc_code or "none",
        voltage if voltage is not None else "N/A",
        len(notes),
    )

    prompt = UNDERSTAND_CLAIM_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        voltage=voltage if voltage is not None else "not provided",
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    t0 = time.monotonic()
    try:
        logger.debug("Sending request to OpenRouter | model=%s timeout=%ds", MODEL, timeout)
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.error("OpenRouter request timed out after %ds", timeout)
        return None
    except requests.RequestException as e:
        logger.error("OpenRouter request failed: %s", str(e))
        return None

    elapsed = time.monotonic() - t0
    logger.debug("OpenRouter responded in %.2fs | status=%d", elapsed, response.status_code)

    if response.status_code == 429:
        logger.warning("Rate limited by OpenRouter (429)")
        return None

    if response.status_code != 200:
        logger.error(
            "OpenRouter API error | status=%d body=%s",
            response.status_code,
            response.text,
        )
        return None

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    logger.debug("Raw LLM response: %s", content)

    try:
        parsed = json.loads(content)
        return {
            "category": parsed.get("category", "other"),
            "normalized_complaint": parsed.get("normalized_complaint", "OBD Light ON"),
            "severity": parsed.get("severity", "medium"),
            "failure_analysis": parsed.get("failure_analysis", "Unknown"),
            "reasoning": parsed.get("reasoning", ""),
            "confidence": parsed.get("confidence", 0.5),
        }
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", content)
        return None


def understand_claim_with_retry(
    notes: str,
    dtc_code: str,
    voltage: Optional[float],
    max_retries: int = 2,
    timeout: int = 30,
) -> Optional[dict]:
    """
    Call OpenRouter LLM with retry logic for transient failures.

    Args:
        notes: Technician's free-text notes
        dtc_code: Fault code (e.g., "P0562")
        voltage: Measured voltage reading
        max_retries: Maximum number of retry attempts (default 2)
        timeout: Request timeout in seconds (default 30)

    Returns:
        dict with keys: category, normalized_complaint, severity, failure_analysis, reasoning, confidence
        None if all retries are exhausted
    """
    logger.info("Starting understand_claim with retry | max_retries=%d", max_retries)

    for attempt in range(max_retries):
        try:
            result = understand_claim(notes, dtc_code, voltage, timeout)
            if result is not None:
                if attempt > 0:
                    logger.info("Succeeded on retry attempt %d/%d", attempt + 1, max_retries)
                return result
        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed: %s", attempt + 1, max_retries, str(e)
            )
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt
            logger.info("Retrying in %ds...", sleep_time)
            time.sleep(sleep_time)

    return None


TRANSLATE_ML_FEATURES_PROMPT = """You are preparing structured features for a machine learning model.
Given the warranty claim below, extract clean structured features.

Technician Notes: {notes}
DTC Code: {dtc_code}
Measured Voltage: {voltage}
Pre-classified Category: {llm_category}

Rules:
- customer_complaint MUST be EXACTLY one of:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- dtc_codes: split comma-separated codes into a list, uppercase, strip spaces
- voltage: use the measured value as a float; if missing use 12.5
- has_P/U/C/B: 1 if any code starts with that letter, else 0

Respond ONLY with this JSON:
{{
  "customer_complaint": "...",
  "dtc_codes": ["..."],
  "dtc_text": "...",
  "dtc_count": 0,
  "voltage": 0.0,
  "has_P": 0,
  "has_U": 0,
  "has_C": 0,
  "has_B": 0
}}
"""


def translate_to_ml_features(
    notes: str,
    dtc_code: str,
    voltage: Optional[float],
    llm_category: str,
    timeout: int = 30,
) -> dict | None:
    """
    Translate raw claim data into ML-ready features using LLM.

    Args:
        notes: Technician's free-text notes
        dtc_code: Fault code (e.g., "P0562")
        voltage: Measured voltage reading
        llm_category: Category from Stage 1 understanding
        timeout: Request timeout in seconds (default 30)

    Returns:
        dict with keys: customer_complaint, dtc_codes, dtc_text, dtc_count, voltage, has_P, has_U, has_C, has_B
        None if API call fails
    """
    api_key = get_api_key()

    logger.info(
        "Translating to ML features | dtc=%s category=%s",
        dtc_code or "none",
        llm_category,
    )

    prompt = TRANSLATE_ML_FEATURES_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        voltage=voltage if voltage is not None else "not provided",
        llm_category=llm_category,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    t0 = time.monotonic()
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.error("OpenRouter request timed out after %ds", timeout)
        return None
    except requests.RequestException as e:
        logger.error("OpenRouter request failed: %s", str(e))
        return None

    elapsed = time.monotonic() - t0
    logger.debug("OpenRouter responded in %.2fs | status=%d", elapsed, response.status_code)

    if response.status_code != 200:
        logger.error("OpenRouter API error | status=%d", response.status_code)
        return None

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        return {
            "customer_complaint": parsed.get("customer_complaint", "OBD Light ON"),
            "dtc_codes": parsed.get("dtc_codes", []),
            "dtc_text": parsed.get("dtc_text", ""),
            "dtc_count": parsed.get("dtc_count", 0),
            "voltage": parsed.get("voltage", voltage if voltage is not None else 12.5),
            "has_P": parsed.get("has_P", 0),
            "has_U": parsed.get("has_U", 0),
            "has_C": parsed.get("has_C", 0),
            "has_B": parsed.get("has_B", 0),
        }
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", content)
        return None
