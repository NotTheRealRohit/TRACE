---
description: Extensively test if an implementation actually functions correctly with real inputs, live API calls, and pipeline execution
---

# Test Implementation

You are tasked with functionally testing an implementation — not just running the test suite, but actually executing the code with real inputs, making live API calls, and verifying the system behaves correctly end-to-end.

This command replaces the "Manual Verification" steps in `implement_plan`. You simulate what a human tester would do, but do it systematically and exhaustively.

**Do not trust the test suite alone.** Unit tests can pass while real behaviour is broken. Your job is to find those gaps.

---

## Initial Response

When invoked:

1. **If a plan path is provided**, proceed immediately to Step 1
2. **If no plan path provided**, ask:
   ```
   Please provide the plan file path so I can test the implementation.
   Example: /test-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

   Optionally specify a phase:
   /test-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md --phase 3
   ```

---

## Process Steps

### Step 1: Load Full Context

1. **Read the plan file completely** (no limit/offset)
   - Note all Manual Verification steps in each completed phase
   - Note the Project Context (language, test runner)
   - Identify the output schema and category registry

2. **Read all implementation files completely**
   - The main parser/interpreter module
   - The OpenRouter client wrapper
   - The category registry file
   - The pipeline integration file
   - All test fixture files (golden notes, example inputs)

3. **Run the automated test suite first** — if it fails, stop immediately:
   ```bash
   [test command from plan's Project Context]
   ```
   If tests fail: report immediately and route to `/implement_plan`. Do not proceed with functional testing on broken code.

---

### Step 2: Execute the Six Test Suites

Work through each suite systematically. Capture actual outputs for every test, not just pass/fail.

---

#### Suite 1: Schema Validation
Verify the output contract is airtight.

Run the interpreter against 5 inputs and for each output verify:
- [ ] All required keys are present: `raw_note`, `parsed_events`, `root_causes`, `category`, `subcategory`, `confidence`, `other_reason`
- [ ] `confidence` is a float between 0.0 and 1.0
- [ ] `parsed_events` and `root_causes` are non-empty lists
- [ ] `category` is always a string from the registry or exactly `"other"`
- [ ] When `category == "other"`, `other_reason` is a non-null, non-empty string
- [ ] When `category != "other"`, `other_reason` is null

**Test inputs to use:**
```
1. "engine stalling after driving through puddle"
2. "brakes making grinding noise on left side"
3. "check engine light came on after oil change"
4. "car won't start this morning, battery seems fine"
5. "something feels wrong" (deliberately vague — should trigger "other")
```

---

#### Suite 2: Category Accuracy (Golden Set)
This is the most important suite. Accuracy is the stated priority.

1. **Load the golden test set** from `tests/fixtures/golden_notes.json` (or wherever it lives)
2. **Run every note through the interpreter**
3. **For each result, check**:
   - Does `category` match the expected category?
   - Does `subcategory` match?
   - Is `confidence` reasonable (not uniformly 0.99 for everything)?
   - Are `root_causes` semantically correct (not just parroting the input)?

4. **Calculate and report accuracy**:
   ```
   Category accuracy:    X/20 (XX%)
   Subcategory accuracy: X/20 (XX%)
   Confidence range:     0.72 – 0.96 (looks calibrated / suspiciously uniform)
   ```

5. **For every wrong answer**, show:
   ```
   Input:    "brake pedal soft after sitting overnight"
   Expected: brake_system / hydraulic
   Got:      brake_system / mechanical  ← wrong subcategory
   Confidence: 0.94
   Root causes: ["brake fluid loss", "air in brake line"]  ← correct
   ```

**Minimum pass threshold**: 90% category accuracy on the full golden set.

---

#### Suite 3: Edge Cases
Test the inputs the happy path wasn't designed for.

Run each of these and capture the full output:

| Input | What to verify |
|-------|---------------|
| Multi-issue: `"engine overheating and brakes grinding simultaneously"` | Two separate root causes, correct primary category chosen |
| Typo-heavy: `"egine stallin aftr pudl"` | Still categorizes correctly despite typos |
| Very short: `"car broken"` | Falls into `"other"` with a meaningful reason |
| Very long: paste a 200-word maintenance log paragraph | Doesn't crash, extracts the most relevant event |
| Non-domain: `"my coffee maker is broken"` | `"other"` with reason, not forced into a car category |
| Empty string: `""` | Handled gracefully — error or `"other"`, not a crash |
| Repeated call: same input twice | Returns consistent results (LLMs can be non-deterministic) |

---

#### Suite 4: API Resilience
Test that the OpenRouter integration handles failure modes correctly.

**Test 4a — Malformed JSON response**
Temporarily mock the API to return a non-JSON string and verify:
- [ ] Retry logic triggers
- [ ] After max retries, raises a clear exception (not a silent wrong result)
- [ ] The exception message is useful for debugging

```python
# Run this directly
from unittest.mock import patch
with patch('your_module.call_openrouter', return_value="not json at all"):
    result = interpret_note("engine stalling")
    # Should raise, retry, or return a clearly marked failure
```

**Test 4b — Rate limit simulation**
Mock a 429 response and verify:
- [ ] Retry with backoff is triggered
- [ ] Does not silently return an empty or wrong result

**Test 4c — Timeout simulation**
Mock a timeout and verify:
- [ ] Handled gracefully
- [ ] Does not hang indefinitely

---

#### Suite 5: Pipeline Integration
Verify the new LLM layer doesn't break what was already working.

1. **Run a full end-to-end flow**: feed a note through the ingestion pipeline → LLM interpreter → existing ML model input
2. Verify:
   - [ ] The existing ML models receive their expected input format unchanged
   - [ ] The LLM output is validated before being passed downstream
   - [ ] If the LLM is mocked to be unavailable, the pipeline fails loudly (not silently)
   - [ ] No global state was changed between calls

3. **Run the existing ML model test suite** to confirm no regressions:
   ```bash
   [existing ML model test command]
   ```

---

#### Suite 6: Consistency & Determinism
LLMs are non-deterministic. Verify the system is stable enough for production.

Run the same 5 inputs **three times each** and check:
- [ ] `category` is consistent across all three runs (should be 100%)
- [ ] `subcategory` is consistent (should be ≥ 90%)
- [ ] `root_causes` are semantically consistent (exact match not required)
- [ ] `confidence` doesn't swing wildly (within ±0.15 is acceptable)

Report:
```
Consistency results (3 runs each, 5 inputs):
Category consistency:    5/5 inputs (100%)
Subcategory consistency: 4/5 inputs (80%) — "brake_system" fluctuated between mechanical/hydraulic
Confidence variance:     avg ±0.08 (acceptable)
```

---

### Step 3: Compile and Present Results

```
## Test Results: [Plan Name]

### Overall Verdict: PASS | PARTIAL PASS | FAIL

---

### Suite Results Summary

| Suite | Result | Notes |
|-------|--------|-------|
| 1. Schema Validation     | ✅ PASS | All 5 outputs valid |
| 2. Category Accuracy     | ❌ FAIL | 16/20 (80%) — below 90% threshold |
| 3. Edge Cases            | ⚠️ PARTIAL | Empty string causes unhandled crash |
| 4. API Resilience        | ✅ PASS | Retry logic works correctly |
| 5. Pipeline Integration  | ✅ PASS | Existing ML models unaffected |
| 6. Consistency           | ⚠️ PARTIAL | Subcategory fluctuates on brake inputs |

---

### Failures (with actual outputs)

**Suite 2 — Accuracy failure (4 wrong)**
1. Input: "brake pedal soft after sitting overnight"
   Expected: brake_system/hydraulic
   Got:      brake_system/mechanical
   [full output JSON]

2. ...

**Suite 3 — Empty string crash**
Input: ""
Expected: "other" with reason or graceful error
Got: KeyError: 'category' at parser.py:67

---

### Recommended Next Step
[See routing section below]
```

---

### Step 4: Route to the Correct Next Command

#### If all suites PASS:
```
✅ All functional tests passed. The implementation is working correctly.

Proceed to commit:
/commit
```

#### If FAIL due to a bug in the implementation (plan was correct):
```
❌ Functional testing failed — bugs found in the implementation.
The plan is correct but the code needs fixing.

Run:
/implement_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

Tell implement_plan:
"Fix these failing test cases using TDD — write a failing test for each 
one first, then fix the implementation:

1. Empty string input causes KeyError at parser.py:67
   Expected: return {"category": "other", "other_reason": "Input was empty"}
   
2. [next failing case]"

After fixes, re-run:
/test-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
```

#### If FAIL due to accuracy below threshold (prompt needs work):
```
❌ Category accuracy at XX% — below the 90% threshold.
This is a prompt engineering issue, not a code bug.

The plan's Phase 3 needs refinement. Run:
/iterate_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
— "Phase 3 accuracy is at XX%. These categories are being confused: 
  [list confused pairs]. Update Phase 3 to add few-shot examples 
  for these cases and update success criteria to require re-running 
  test-impl after prompt changes."

Then resume:
/implement_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

Then re-test:
/test-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
```

#### If FAIL due to pipeline regression:
```
❌ Existing ML pipeline is broken by this integration.
This is a plan-level issue — the integration approach needs rethinking.

Run:
/iterate_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
— "Pipeline integration in Phase 4 broke existing ML models. 
   Rethink the integration approach to ensure the LLM layer 
   is fully decoupled and bypassable."

Then:
/implement_plan
/test-impl
```

---

## Important Guidelines

- **Always capture actual outputs** — never just say "it worked"
- **Show full JSON** for failures — the details matter for diagnosis
- **Don't skip Suite 5** — pipeline regressions are the most dangerous failure mode
- **Accuracy below 90% is always a FAIL** — don't soften this
- **Route explicitly** — always end with the exact command to run next
- **One suite failure = overall FAIL** — unless it's a MINOR edge case with a clear workaround
