---
description: Extensively review code quality of a completed or in-progress implementation against its plan
---

# Review Implementation

You are tasked with performing a thorough code quality review of an implementation, judging it against both the plan it was built from and general engineering standards.

Your review must be **specific and actionable** — every finding must include a file:line reference and a concrete fix recommendation. Vague feedback like "improve readability" is not acceptable.

---

## Initial Response

When invoked:

1. **If a plan path is provided**, proceed immediately to Step 1
2. **If no plan path provided**, ask:
   ```
   Please provide the plan file path so I can review the implementation against it.
   Example: /review-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

   Optionally, specify a phase number to review just that phase:
   /review-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md --phase 3
   ```

---

## Process Steps

### Step 1: Load Full Context

1. **Read the plan file completely** (no limit/offset)
   - Understand the intended design, scope, and success criteria for each phase
   - Note the Project Context (language, test runner, test command)
   - Identify which phases are marked complete (checkboxes)

2. **Run git diff to see all changed files**:
   ```bash
   git diff main --name-only
   # or if on main:
   git diff HEAD~1 --name-only
   ```

3. **Read every changed file completely**
   - Source files
   - Test files
   - Config files
   - Any new fixtures or data files

4. **Run the full test suite** and capture output:
   ```bash
   [test command from plan's Project Context section]
   ```

5. **Run any linting/type checking** defined in the plan:
   ```bash
   [lint/typecheck command from plan]
   ```

---

### Step 2: Evaluate Against Six Dimensions

Review the implementation across all six dimensions below. For each finding, record:
- **Severity**: `BLOCKER` | `MAJOR` | `MINOR`
- **File:line** reference
- **What is wrong**
- **What the fix should be**

---

#### Dimension 1: Plan Fidelity
Does the implementation actually do what the plan specified?

Check:
- [ ] Every phase marked complete has all its specified changes implemented
- [ ] The output schema matches exactly what the plan defines
- [ ] Any "What We're NOT Doing" items have not crept into the implementation
- [ ] Success criteria from each completed phase are genuinely satisfied
- [ ] No shortcuts taken that defer required work silently

**BLOCKER if**: implementation diverges from plan intent without a clear, documented reason.

---

#### Dimension 2: TDD Compliance
Was the code actually written test-first?

Check:
- [ ] Every production code path has at least one corresponding test
- [ ] Tests assert specific behaviour, not just "it doesn't crash"
- [ ] Test names follow the project's naming convention
- [ ] No production logic exists that has zero test coverage
- [ ] Tests are in the correct directory for this project

Red flags:
- Tests that only assert `assert result is not None`
- Production functions with no test file at all
- Test files that import but never assert

**BLOCKER if**: a complete phase has zero tests for its core behaviour.

---

#### Dimension 3: Error Handling & Resilience
Are failure modes handled correctly?

For your Python/OpenRouter context, specifically check:
- [ ] API call failures (network errors, timeouts) are caught and handled
- [ ] HTTP 429 rate limit errors trigger retry logic, not silent failure
- [ ] Malformed or unparseable JSON responses are caught with retry
- [ ] `"other"` category is used correctly when nothing matches — not silently forced into a category
- [ ] `other_reason` is always a non-null string when `category == "other"`
- [ ] Confidence scores are not always suspiciously high (LLM calibration issue)
- [ ] Pipeline integration fails loudly, not silently, if the LLM is unavailable

**BLOCKER if**: unhandled exceptions can propagate to the existing ML pipeline.

---

#### Dimension 4: Code Quality & Conventions
Does the code fit naturally into the existing codebase?

Check:
- [ ] Follows the project's existing naming conventions (snake_case for Python etc.)
- [ ] No functions longer than ~40 lines without a clear reason
- [ ] No duplicated logic that should be extracted
- [ ] The category registry is the single source of truth — not hardcoded strings scattered across files
- [ ] Imports are clean and no unused dependencies added
- [ ] Config (API keys, model name, retry counts) is not hardcoded — uses env vars or config file

**MAJOR if**: new code introduces conventions inconsistent with the rest of the project.

---

#### Dimension 5: Test Quality
Are the tests actually useful?

Check:
- [ ] Golden test set covers at least 15 real notes (not just the car/puddle example)
- [ ] Tests cover the `"other"` fallback explicitly
- [ ] Tests cover multi-issue notes (one note with two root causes)
- [ ] Tests cover ambiguous/vague notes
- [ ] At least one test verifies that `other_reason` is populated correctly
- [ ] Integration test exists that makes a real (or mocked) OpenRouter call end-to-end
- [ ] No tests that pass trivially because the assertion is too weak

**MAJOR if**: golden test set has fewer than 10 notes.

---

#### Dimension 6: Pipeline Safety
Does the integration protect the existing ML pipeline?

Check:
- [ ] The LLM layer is opt-outable or bypassable if the API is down
- [ ] Existing ML model inputs are not broken by the new integration
- [ ] The LLM output is validated before being passed downstream
- [ ] No global state is mutated by the new code
- [ ] Logging is present for LLM calls (inputs, outputs, latency) for debugging

**BLOCKER if**: the existing pipeline can break due to the new integration.

---

### Step 3: Compile Results and Write to Disk

**Determine the output path** from the plan filename and phase:
- Plan: `thoughts/shared/plans/2026-03-06-openrouter-llm-integration.md`
- Phase: 2
- Output: `thoughts/shared/reviews/2026-03-06-phase-2-review.md`

**Write the findings file** to `thoughts/shared/reviews/YYYY-MM-DD-phase-N-review.md`:

```markdown
---
review_date: YYYY-MM-DD
plan: thoughts/shared/plans/YYYY-MM-DD-[name].md
phase: N
verdict: PASS | PASS WITH MINORS | FAIL
reviewed_by: opencode
---

# Review: [Plan Name] — Phase N

## Overall Verdict: PASS | PASS WITH MINORS | FAIL

---

## BLOCKERS (must fix before proceeding)
1. [file.py:42] — [What is wrong]
   Fix: [Concrete fix]

## MAJOR Issues (fix in current phase before moving on)
1. [file.py:15] — [What is wrong]
   Fix: [Concrete fix]

## MINOR Issues (can be addressed in a follow-up iterate_plan)
1. [file.py:88] — [What is wrong]
   Fix: [Concrete fix]

---

## What's Working Well
- [specific things done correctly]

---

## Recommended Next Step
[Exact command and context to pass it — see routing section]
```

After writing, confirm to the user:
```
Review findings written to: thoughts/shared/reviews/YYYY-MM-DD-phase-N-review.md
```

Then **also present the findings in chat** in the same format as above — do not make the user open the file to see the results.

---

### Step 4: Route to the Correct Next Command

Based on findings, give an explicit recommendation in chat AND include it in the written file:

#### If PASS:
```
✅ Review passed. Proceed to functional testing:

/test-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
```

#### If FAIL due to implementation issues only (plan was correct):
```
❌ Review failed — implementation issues found. The plan is correct but 
the code needs fixes. Run:

/implement_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

When implement_plan resumes, tell it:
"Fix the following review findings before re-running the phase:
- [paste BLOCKER and MAJOR findings here]"

After fixes, re-run:
/review-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
```

#### If FAIL due to plan being incomplete or wrong:
```
❌ Review failed — the plan itself needs updating before the implementation 
can be fixed. Run:

/iterate_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md \
  --review thoughts/shared/reviews/YYYY-MM-DD-phase-N-review.md

Then resume implementation:
/implement_plan thoughts/shared/plans/2025-03-05-openrouter-note-parser.md

Then re-run this review:
/review-impl thoughts/shared/plans/2025-03-05-openrouter-note-parser.md
```

---

## Important Guidelines

- **Never be vague** — every finding needs file:line and a concrete fix
- **Severity matters** — MINORs alone should not block progress
- **One BLOCKER = FAIL** — don't let a critical issue slip through as MAJOR
- **Read the actual code** — don't assume from filenames; read the full files
- **Re-run tests yourself** — don't trust that green checkboxes in the plan are correct
- **Be constructive** — the goal is a better implementation, not a perfect score
- **Always write findings to disk** — the review file is the contract that `iterate_plan` reads; never skip this step
