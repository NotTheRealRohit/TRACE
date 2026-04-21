"""
Gather PoC demonstration data by running predict() on test claims
that exercise each rule and key pipeline paths.

Runs each test case twice:
  1. With LLM enabled  (if OPENROUTER_API_KEY or OPENAI_API_KEY is set)
  2. Without LLM       (Rule+ML fallback only)

This lets the PoC chapter compare LLM+Rule+ML vs Rule+ML outputs.

Usage:
    cd backend && python3 gather_data.py              # both modes
    cd backend && python3 gather_data.py --llm-only    # LLM mode only
    cd backend && python3 gather_data.py --no-llm      # fallback mode only

Output: structured JSON saved to
        Documentation/ProofOfConcept/poc_test_results.json
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

TEST_CASES = [
    # --- One test per rule ---
    {
        "id": "over_voltage",
        "description": "Over-voltage rule (>16V) -> Customer Failure",
        "fault_code": "P0562",
        "technician_notes": "Engine overheating, low idle",
        "voltage": 17.5,
    },
    {
        "id": "low_voltage",
        "description": "Low-voltage rule (<11V) -> Customer Failure",
        "fault_code": "P0562",
        "technician_notes": "Engine overheating, low idle",
        "voltage": 9.0,
    },
    {
        "id": "moisture",
        "description": "Moisture keyword rule -> Customer Failure",
        "fault_code": "P0301",
        "technician_notes": "Moisture found inside connector, corroded",
        "voltage": 12.5,
    },
    {
        "id": "physical_damage",
        "description": "Physical damage keyword rule -> Customer Failure",
        "fault_code": "P0301",
        "technician_notes": "Cracked housing, impact damage visible",
        "voltage": 13.0,
    },
    {
        "id": "ntf",
        "description": "NTF keyword rule -> According to Specification",
        "fault_code": "",
        "technician_notes": "No fault found, intermittent complaint",
        "voltage": 13.0,
    },
    {
        "id": "u_code",
        "description": "U-code CAN/LIN rule -> Production Failure",
        "fault_code": "U0100",
        "technician_notes": "Communication error on CAN bus",
        "voltage": 14.2,
    },
    {
        "id": "p_code_engine",
        "description": "P-code + engine symptom rule -> Production Failure",
        "fault_code": "P0562",
        "technician_notes": "Engine overheating, low idle",
        "voltage": 14.2,
    },
    {
        "id": "c_code",
        "description": "C-code chassis rule -> Production Failure",
        "fault_code": "C0045",
        "technician_notes": "Brake warning light ON",
        "voltage": 13.5,
    },
    {
        "id": "b_code",
        "description": "B-code body rule -> Production Failure",
        "fault_code": "B1234",
        "technician_notes": "Starting problem, nothing visible",
        "voltage": 13.0,
    },
    # --- Priority test: over_voltage should win over moisture ---
    {
        "id": "priority_test",
        "description": "Priority: over_voltage (rule 1) beats moisture (rule 3)",
        "fault_code": "P0301",
        "technician_notes": "Moisture found inside connector, corroded",
        "voltage": 17.5,
    },
    # --- ML-only path: no rule fires ---
    {
        "id": "ml_only",
        "description": "No rule fires, ML-only decision path",
        "fault_code": "P0442",
        "technician_notes": "Minor EVAP leak detected during routine inspection",
        "voltage": 13.8,
    },
]


def run_test_suite(mode: str) -> list[dict]:
    """Run all test cases in the given mode ('llm' or 'no_llm')."""
    # Stash original keys
    saved_keys = {}
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        saved_keys[key] = os.environ.get(key)

    if mode == "no_llm":
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

    # Import after env is configured; reload to pick up env changes
    import importlib
    import ml_predictor
    importlib.reload(ml_predictor)
    from ml_predictor import predict

    has_llm_key = any(saved_keys.values()) if mode == "llm" else False
    mode_label = "LLM+Rule+ML" if (mode == "llm" and has_llm_key) else "Rule+ML"
    print(f"\n{'='*80}")
    print(f"  Mode: {mode_label}")
    print(f"{'='*80}")

    results = []
    for tc in TEST_CASES:
        r = predict(tc["fault_code"], tc["technician_notes"], tc["voltage"])
        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "mode": mode,
            "input": {
                "fault_code": tc["fault_code"],
                "technician_notes": tc["technician_notes"],
                "voltage": tc["voltage"],
            },
            "output": r,
        })
        print(f"  [{tc['id']:20s}] {r['status']:25s} | WD: {r['warranty_decision']:30s} | "
              f"Conf: {r['confidence']:5.1f}% | Engine: {r['decision_engine']}")

    # Restore original keys
    for key, val in saved_keys.items():
        if val is not None:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)

    return results


def main():
    args = sys.argv[1:]
    run_llm = "--no-llm" not in args
    run_no_llm = "--llm-only" not in args

    has_any_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    if run_llm and not has_any_key:
        print("WARNING: No OPENROUTER_API_KEY or OPENAI_API_KEY found in environment or .env")
        print("         LLM mode will fall back to Rule+ML automatically.\n")

    all_results = {}

    if run_llm:
        all_results["llm"] = run_test_suite("llm")

    if run_no_llm:
        all_results["no_llm"] = run_test_suite("no_llm")

    # Save to JSON
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "Documentation", "ProofOfConcept", "poc_test_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
