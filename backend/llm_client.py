"""
LLM Client for TRACE Warranty Claims
-------------------------------------
Supports both OpenAI and OpenRouter providers.
OpenAI (gpt-4o-mini) is used if OPENAI_API_KEY is set, otherwise falls back to OpenRouter.
"""

import os
import sys
import json
import time
import requests
from typing import Optional
from logging_config import setup_logging, get_logger

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"

_openai_client = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

setup_logging()
logger = get_logger("trace.llm_client")


def _get_provider() -> Optional[str]:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def get_api_key() -> str:
    provider = _get_provider()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    else:
        logger.error("No LLM provider API key found in environment")
        raise ValueError(
            "No LLM provider configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY"
        )

    if not api_key:
        logger.error("API key is not set in environment")
        raise ValueError(f"{provider.upper()}_API_KEY not set in environment")
    return api_key


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def _call_llm(prompt: str, timeout: int = 30) -> Optional[str]:
    provider = _get_provider()
    if not provider:
        logger.error("No LLM provider configured")
        raise RuntimeError(
            "No LLM provider configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY"
        )

    t0 = time.monotonic()

    if provider == "openai":
        return _call_openai(prompt, timeout, t0)
    else:
        return _call_openrouter(prompt, timeout, t0)


def _call_openai(prompt: str, timeout: int, t0: float) -> Optional[str]:
    client = _get_openai_client()
    model = OPENAI_MODEL

    logger.debug("Calling OpenAI | model=%s timeout=%ds", model, timeout)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            seed=42,
            timeout=timeout,
        )
    except Exception as e:
        logger.error("OpenAI request failed: %s", str(e))
        return None

    elapsed = time.monotonic() - t0
    logger.debug("OpenAI responded in %.2fs", elapsed)

    content = response.choices[0].message.content
    logger.debug("Raw LLM response: %s", content)
    return content


def _call_openrouter(prompt: str, timeout: int, t0: float) -> Optional[str]:
    api_key = get_api_key()
    model = OPENROUTER_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "seed": 42,
    }

    try:
        logger.debug("Calling OpenRouter | model=%s timeout=%ds", model, timeout)
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
    logger.debug(
        "OpenRouter responded in %.2fs | status=%d", elapsed, response.status_code
    )

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
    return content


def _parse_json_response(content: str, defaults: dict) -> Optional[dict]:
    try:
        parsed = json.loads(content)
        result = {}
        for key, default in defaults.items():
            result[key] = parsed.get(key, default)
        return result
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON: %s", content)
        return None


CATEGORIZATION_PROMPT = """You are a warranty claim analyst for automotive electronics.
Analyze the technician's notes and classify the claim into ONE of these categories:

Categories:
- moisture_damage: water, moisture, wet, flood, rain, humidity, corrosion
- physical_damage: crack, broken, impact, collision, bent, misuse, dropped
- ntf: no fault found, ntf, no trouble, no issue, no defect, intermittent, cannot reproduce
- electrical_issue: electrical short, wiring, connector
- engine_symptom: jerking, pickup, acceleration, overheating, fuel, idle, rough
- communication_fault: CAN bus, LIN bus, communication error, U-code
- other: none of the above

Technician Notes: {notes}
DTC Code: {dtc_code}

Respond ONLY with JSON in this exact format:
{{
  "category": "category_name",
  "confidence": 0.85,
  "failure_analysis": "short description of root cause",
  "reasoning": "brief explanation"
}}
"""


def categorize_notes(notes: str, dtc_code: str, timeout: int = 30) -> dict:
    provider = _get_provider()
    if not provider:
        logger.error("No LLM provider API key found in environment")
        raise RuntimeError(
            "No LLM provider configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY"
        )

    logger.info(
        "Categorizing notes | dtc=%s notes_len=%d provider=%s",
        dtc_code or "none",
        len(notes),
        provider,
    )
    logger.debug("Full technician notes: %s", notes)

    prompt = CATEGORIZATION_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
    )

    content = _call_llm(prompt, timeout)
    if content is None:
        raise RuntimeError("LLM API call failed")

    defaults = {
        "category": "other",
        "confidence": 0.8,
        "failure_analysis": "Unknown",
        "reasoning": "",
    }

    result = _parse_json_response(content, defaults)
    if result is None:
        raise RuntimeError("Failed to parse LLM response")

    logger.info(
        "Categorization complete | category=%s confidence=%.2f",
        result["category"],
        result["confidence"],
    )
    return result


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
- reason: 1-2 sentences explaining the decision in plain language
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
    provider = _get_provider()
    if not provider:
        logger.error("No LLM provider configured")
        return None

    logger.info(
        "[STAGE 6] LLM Output Formatting | decision_engine=%s provider=%s",
        combined.get("decision_engine", "unknown"),
        provider,
    )

    prompt = FORMAT_OUTPUT_PROMPT.format(
        combined_json=json.dumps(combined),
    )

    content = _call_llm(prompt, timeout)
    if content is None:
        return None

    defaults = {
        "status": "Needs Manual Review",
        "failure_analysis": combined.get("ml_failure_analysis", "Unknown"),
        "warranty_decision": combined.get("warranty_decision", ""),
        "confidence": combined.get("combined_confidence", 50.0),
        "reason": "",
        "matched_complaint": features.get("customer_complaint", "OBD Light ON"),
        "decision_engine": combined.get("decision_engine", "ML"),
    }

    return _parse_json_response(content, defaults)


UNDERSTAND_CLAIM_PROMPT = """You are an automotive warranty analyst. Analyze the claim below and respond ONLY with JSON.

Technician Notes: {notes}
DTC Code: {dtc_code}

Classify into EXACTLY ONE category from this list:
  moisture_damage, physical_damage, ntf, electrical_issue,
  engine_symptom, communication_fault, other

DISAMBIGUATION RULES (apply in order - first match wins):
1. If notes mention overheating, jerking, pickup, acceleration, fuel consumption, idle, rough -> engine_symptom (NOT electrical_issue)
2. If notes mention CAN bus, LIN bus, communication, network, U-code -> communication_fault
3. If notes mention moisture, water, wet, flood, rain, humidity, corrosion -> moisture_damage
4. If notes mention crack, broken, impact, collision, bent, misuse, dropped, physical damage -> physical_damage
5. If notes mention no fault, ntf, no trouble, no issue, no defect, intermittent, cannot reproduce -> ntf
6. If notes mention electrical short, wiring problems (without engine symptoms) -> electrical_issue
7. Otherwise -> other

Also provide:
- normalized_complaint: one of these exact strings:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- severity: "low" | "medium" | "high"
- failure_analysis: short root cause string (max 15 words)
- reasoning: brief explanation (max 30 words)
- confidence: float 0.0-1.0

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


def understand_claim(notes: str, dtc_code: str, timeout: int = 30) -> dict | None:
    provider = _get_provider()
    if not provider:
        logger.error("No LLM provider configured")
        return None

    logger.info(
        "[STAGE 1] LLM Understanding | dtc=%s notes_len=%d provider=%s",
        dtc_code or "none",
        len(notes),
        provider,
    )

    prompt = UNDERSTAND_CLAIM_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
    )

    content = _call_llm(prompt, timeout)
    if content is None:
        return None

    defaults = {
        "category": "other",
        "normalized_complaint": "OBD Light ON",
        "severity": "medium",
        "failure_analysis": "Unknown",
        "reasoning": "",
        "confidence": 0.5,
    }

    return _parse_json_response(content, defaults)


def understand_claim_with_retry(
    notes: str,
    dtc_code: str,
    max_retries: int = 2,
    timeout: int = 30,
) -> Optional[dict]:
    """Call LLM with retry logic for transient failures."""
    logger.info("Starting understand_claim with retry | max_retries=%d", max_retries)

    for attempt in range(max_retries):
        try:
            result = understand_claim(notes, dtc_code, timeout)
            if result is not None:
                if attempt > 0:
                    logger.info(
                        "Succeeded on retry attempt %d/%d", attempt + 1, max_retries
                    )
                return result
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, max_retries, str(e))
        if attempt < max_retries - 1:
            sleep_time = 2**attempt
            logger.info("Retrying in %ds...", sleep_time)
            time.sleep(sleep_time)

    return None


TRANSLATE_ML_FEATURES_PROMPT = """You are preparing structured features for a machine learning model.
Given the warranty claim below, extract clean structured features.

Technician Notes: {notes}
DTC Code: {dtc_code}
Pre-classified Category: {llm_category}

Rules:
- customer_complaint MUST be EXACTLY one of:
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
- dtc_codes: split comma-separated codes into a list, uppercase, strip spaces
- has_P/U/C/B: 1 if any code starts with that letter, else 0

Respond ONLY with this JSON:
{{
  "customer_complaint": "...",
  "dtc_codes": ["..."],
  "dtc_text": "...",
  "dtc_count": 0,
  "has_P": 0,
  "has_U": 0,
  "has_C": 0,
  "has_B": 0
}}
"""


def translate_to_ml_features(
    notes: str,
    dtc_code: str,
    llm_category: str,
    timeout: int = 30,
) -> dict | None:
    provider = _get_provider()
    if not provider:
        logger.error("No LLM provider configured")
        return None

    logger.info(
        "[STAGE 3] LLM Feature Translation | dtc=%s category=%s provider=%s",
        dtc_code or "none",
        llm_category,
        provider,
    )

    prompt = TRANSLATE_ML_FEATURES_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        llm_category=llm_category,
    )

    content = _call_llm(prompt, timeout)
    if content is None:
        return None

    defaults = {
        "customer_complaint": "OBD Light ON",
        "dtc_codes": [],
        "dtc_text": "",
        "dtc_count": 0,
        "has_P": 0,
        "has_U": 0,
        "has_C": 0,
        "has_B": 0,
    }

    return _parse_json_response(content, defaults)
