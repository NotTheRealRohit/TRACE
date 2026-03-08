import os
import sys
import time

os.chdir("/mnt/d/study/git/capProj-2/backend")
sys.path.insert(0, "/mnt/d/study/git/capProj-2/backend")

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("Testing OpenRouter API Response Times")
print("=" * 60)

test_inputs = {
    "notes": "Engine overheating, low idle, vehicle struggling to start",
    "fault_code": "P0562",
    "voltage": 14.2
}

from llm_client import (
    understand_claim,
    translate_to_ml_features,
    format_output,
)

print(f"\nTest Input:")
print(f"  Notes: {test_inputs['notes']}")
print(f"  Fault Code: {test_inputs['fault_code']}")
print(f"  Voltage: {test_inputs['voltage']}")
print()

# Stage 1: understand_claim
print("-" * 60)
print("STAGE 1: understand_claim (Semantic Understanding)")
print("-" * 60)
t0 = time.monotonic()
result1 = understand_claim(
    test_inputs["notes"],
    test_inputs["fault_code"],
    test_inputs["voltage"],
    timeout=60
)
t1 = time.monotonic()
elapsed1 = t1 - t0
if result1:
    print(f"  Status: SUCCESS")
    print(f"  Time: {elapsed1:.2f}s")
    print(f"  Category: {result1.get('category')}")
    print(f"  Normalized Complaint: {result1.get('normalized_complaint')}")
else:
    print(f"  Status: FAILED")
    print(f"  Time: {elapsed1:.2f}s")

# Stage 3: translate_to_ml_features
print()
print("-" * 60)
print("STAGE 3: translate_to_ml_features (Feature Translation)")
print("-" * 60)
category = result1.get("category", "other") if result1 else "other"
t0 = time.monotonic()
result3 = translate_to_ml_features(
    test_inputs["notes"],
    test_inputs["fault_code"],
    test_inputs["voltage"],
    category,
    timeout=60
)
t1 = time.monotonic()
elapsed3 = t1 - t0
if result3:
    print(f"  Status: SUCCESS")
    print(f"  Time: {elapsed3:.2f}s")
    print(f"  Customer Complaint: {result3.get('customer_complaint')}")
    print(f"  DTC Count: {result3.get('dtc_count')}")
else:
    print(f"  Status: FAILED")
    print(f"  Time: {elapsed3:.2f}s")

# Stage 6: format_output (needs combined data)
print()
print("-" * 60)
print("STAGE 6: format_output (Output Formatting)")
print("-" * 60)

combined_mock = {
    "status": "Approved",
    "warranty_decision": "Production Failure",
    "combined_confidence": 85.0,
    "agreement": True,
    "rule_fired": False,
    "ml_warranty_decision": "Production Failure",
    "ml_failure_analysis": "controller failure due to supplier production failure",
    "decision_engine": "ML"
}

features_mock = {
    "customer_complaint": "Engine overheating",
    "dtc_text": "P0562",
    "voltage": 14.2
}

t0 = time.monotonic()
result6 = format_output(combined_mock, features_mock, timeout=60)
t1 = time.monotonic()
elapsed6 = t1 - t0
if result6:
    print(f"  Status: SUCCESS")
    print(f"  Time: {elapsed6:.2f}s")
    print(f"  Status: {result6.get('status')}")
    print(f"  Reason: {result6.get('reason', '')[:80]}...")
else:
    print(f"  Status: FAILED")
    print(f"  Time: {elapsed6:.2f}s")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Stage 1 (understand_claim):        {elapsed1:.2f}s")
print(f"  Stage 3 (translate_to_ml_features): {elapsed3:.2f}s")
print(f"  Stage 6 (format_output):           {elapsed6:.2f}s")
print(f"  ─────────────────────────────────────────")
print(f"  Total LLM API time:                 {elapsed1 + elapsed3 + elapsed6:.2f}s")
print("=" * 60)
