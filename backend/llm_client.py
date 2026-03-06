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
        raise RuntimeError(f"Failed to parse LLM response as JSON: {content}")


def categorize_notes_with_retry(
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
        dict with keys: category, confidence, failure_analysis, reasoning
        None if all retries are exhausted

    Raises:
        RuntimeError: If non-retryable error occurs
    """
    logger.info("Starting categorization with retry | max_retries=%d", max_retries)

    for attempt in range(max_retries):
        try:
            result = categorize_notes(notes, dtc_code, voltage, timeout)
            if attempt > 0:
                logger.info("Succeeded on retry attempt %d/%d", attempt + 1, max_retries)
            return result
        except RuntimeError as e:
            logger.warning(
                "Attempt %d/%d failed: %s", attempt + 1, max_retries, str(e)
            )
            if attempt == max_retries - 1:
                logger.error("All %d attempts exhausted, raising error", max_retries)
                raise
            if "rate" in str(e).lower():
                sleep_time = 2 ** attempt
                logger.info("Rate limited — retrying in %ds...", sleep_time)
                time.sleep(sleep_time)
                continue
            raise

    return None
