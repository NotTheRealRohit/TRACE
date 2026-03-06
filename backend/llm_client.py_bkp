"""
OpenRouter LLM Client for TRACE Warranty Claims
-----------------------------------------------
Handles API calls to OpenRouter for technician note categorization.
"""

import os
import json
import time
import requests
from typing import Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "arcee-ai/trinity-large-preview:free"


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
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
    
    prompt = CATEGORIZATION_PROMPT.format(
        notes=notes,
        dtc_code=dtc_code or "none",
        voltage=voltage if voltage is not None else "not provided"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "TRACE Warranty Claims",
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }
    
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
    except requests.Timeout:
        raise RuntimeError("OpenRouter API request timed out")
    except requests.RequestException as e:
        raise RuntimeError(f"OpenRouter API request failed: {str(e)}")
    
    if response.status_code == 429:
        raise RuntimeError("Rate limited by OpenRouter, try again later")
    
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    try:
        parsed = json.loads(content)
        return {
            "category": parsed.get("category", "other"),
            "confidence": parsed.get("confidence", 0.8),
            "failure_analysis": parsed.get("failure_analysis", "Unknown"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {content}")


def categorize_notes_with_retry(
    notes: str,
    dtc_code: str,
    voltage: Optional[float],
    max_retries: int = 2,
    timeout: int = 30
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
    for attempt in range(max_retries):
        try:
            return categorize_notes(notes, dtc_code, voltage, timeout)
        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise
            if "rate" in str(e).lower():
                sleep_time = 2 ** attempt
                print(f"[TRACE] Rate limited, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            raise
    return None
