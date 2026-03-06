---
description: Iterate on existing implementation plans with thorough research and updates
---

# Iterate Implementation Plan

You are tasked with updating existing implementation plans based on user feedback or gate findings (review/test). You should be skeptical, thorough, and ensure changes are grounded in actual codebase reality.

---

## Initial Response

When this command is invoked:

1. **Parse the input to identify**:
   - Plan file path (e.g., `thoughts/shared/plans/2025-10-16-feature.md`)
   - Requested changes/feedback
   - Optional: `--review <path>` pointing to a specific review or test findings file

2. **Handle different input scenarios**:

   **If NO plan file provided**:
   ```
   I'll help you iterate on an existing implementation plan.

   Which plan would you like to update? Please provide the path to the plan file.
   Tip: You can list recent plans with `ls -lt thoughts/shared/plans/ | head`
   ```
   Wait for user input.

   **If plan file provided but NO feedback and NO --review flag**:
   ```
   I've found the plan at [path]. What changes would you like to make?
   ```
   Wait for user input.

   **If BOTH plan file AND feedback (or --review) provided**:
   Proceed immediately to Step 1. No preliminary questions.

---

## Process Steps

### Step 1: Read the Plan and Load Review Context

1. **Read the existing plan file COMPLETELY** (no limit/offset).
   Note: structure, phases, checkboxes, Project Context (language/stack/test runner), success criteria.

2. **Load the review/test findings**:

   **If `--review <path>` was provided by the user or by `review-impl`/`test-impl`**:
   - Read that file directly.

   **If no `--review` flag was given**:
   - List `thoughts/shared/reviews/` and find files that share the plan's date prefix or name fragment.
     ```bash
     ls -lt thoughts/shared/reviews/ | grep "PLAN-DATE-OR-NAME-FRAGMENT"
     ```
   - Read the most recent matching file (prefer `-review.md` or `-test.md` from the same date).
   - If multiple files exist (both review and test for the same phase), read both.
   - If no matching files exist, proceed on the user's stated feedback alone and note the absence.

3. **Understand what the review/test found**:
   - What was the overall verdict?
   - Which findings are BLOCKER / MAJOR?
   - What did the recommended next step say?
   - Does the failure require a plan change, or just a code fix?
     - **Plan change needed**: missing phase, wrong success criteria, scope gap, prompt issue
     - **Code fix only**: use `implement_plan` directly — do NOT iterate the plan for this

### Step 2: Research If Needed

**Only spawn research tasks if the changes require new technical understanding.**

If feedback requires understanding new code patterns or validating assumptions:

1. Create a research todo list using TodoWrite.
2. Spawn parallel sub-tasks:
   - **codebase-locator** — find relevant files
   - **codebase-analyzer** — understand implementation details
   - **codebase-pattern-finder** — find similar patterns
3. Read any new files identified fully into main context.
4. Wait for ALL sub-tasks to complete before proceeding.

### Step 3: Present Understanding and Confirm

Before touching the plan, confirm your understanding:

```
Based on the review findings and your feedback, I understand the plan needs:
- [Change 1 with specific detail — which phase, what addition]
- [Change 2 with specific detail]

Relevant finding from review (thoughts/shared/reviews/[file]):
- [Quoted or paraphrased finding that drives this change]

I plan to update the plan by:
1. [Specific modification — see insertion rules below]
2. [Another modification]

Does this align with your intent?
```

**Do not touch the plan until the user confirms.**

---

### Step 4: Update the Plan — Insertion Rules

This is the most critical step. `implement_plan` tracks all work via checkboxes (`- [x]` = done, `- [ ]` = todo). Every insertion must respect that contract.

#### Rule 1: Adding to an Incomplete Phase
*The phase has unchecked `- [ ]` items remaining.*

→ Append new TDD rows, Changes Required blocks, and unchecked Success Criteria **within** the phase's existing sections. `implement_plan` will pick them up naturally on its next run.

Example — appending a new test row to an incomplete phase's TDD table:
```markdown
| `test_llm_client.py` | `test_handles_rate_limit` | Retry triggers on 429 response |
```

Example — appending a new Changes Required block:
```markdown
#### 3. Update: `backend/llm_client.py`
**Changes**: Add 429 rate-limit handling
...
```

Example — appending a new unchecked criterion:
```markdown
- [ ] Rate limit retry verified: `pytest -k "test_handles_rate_limit"`
```

---

#### Rule 2: Adding to an Already-Completed Phase
*All success criteria in the phase are checked `- [x]`.*

→ **Do NOT uncheck existing items.** Insert a clearly demarcated additions block immediately after the phase's Success Criteria section:

```markdown
### ↺ Phase N — Additions
> Source: thoughts/shared/reviews/YYYY-MM-DD-phase-N-[review|test].md
> Reason: [one sentence — e.g. "Rate limit handling was missing from original implementation"]
> Added: YYYY-MM-DD

#### TDD — Additional Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| `test_resilience.py` | `test_handles_rate_limit_retry` | Retries on 429 before raising |

**Run each test to confirm RED before implementing:**
```bash
pytest -k "test_handles_rate_limit_retry"
```

#### Additional Changes Required:

#### 1. Update: `backend/llm_client.py`
**Changes**: [description]
...

#### Additional Success Criteria:

##### Automated Verification:
- [ ] RED confirmed: Additional tests fail before implementation
- [ ] GREEN achieved: Additional tests pass after implementation

##### Manual Verification:
- [ ] [any manual checks]
```

**Why this structure:**
- Existing `[x]` items are preserved — history is intact, `implement_plan` won't re-run them
- New `[ ]` items are visible — `implement_plan` resumes and finds them on its next scan
- The `↺` marker and source attribution make it immediately clear this was added post-review, and why
- The reason line ensures the plan remains self-documenting even without the review file open

---

#### Rule 3: Adding an Entirely New Phase
*The review/test identified a concern that doesn't fit into any existing phase.*

→ Insert a complete new `## Phase N` section in the appropriate position (before the Testing Strategy / References sections). Follow the full standard phase structure with all items unchecked `- [ ]`.

```markdown
## Phase N: [Name]

### Overview
[Why this phase exists — reference the review finding]
> Added from: thoughts/shared/reviews/YYYY-MM-DD-phase-N-[review|test].md

### TDD — Tests to Write First

| Test File / Class | Test Name | Validates |
|-------------------|-----------|-----------|
| ... | ... | ... |

### Changes Required:
...

### Success Criteria:

#### Automated Verification:
- [ ] RED confirmed: ...
- [ ] GREEN achieved: ...

#### Manual Verification:
- [ ] ...
```

---

#### Rule 4: What NOT to Change
- **Never uncheck** a `- [x]` item unless explicitly instructed by the user.
- **Never rewrite** a completed phase — only append or add a `↺ Additions` block.
- **Never change** the Project Context section (language/stack/test runner) unless explicitly requested.
- **Never add** a phase for something that is a code fix, not a plan gap — route those to `implement_plan` directly.

---

### Step 5: Sync and Review

Present the changes made:

```
I've updated the plan at `thoughts/shared/plans/[filename].md`

Changes made:
- [Specific change 1 — which phase, which rule applied]
- [Specific change 2]

The updated plan now:
- [Key improvement]
- [What implement_plan will pick up on next run]

Would you like any further adjustments?
```

---

## Important Guidelines

1. **Be Skeptical**:
   - Don't blindly accept change requests that seem problematic
   - Ask for clarification on vague feedback
   - Verify technical feasibility with code research
   - Point out potential conflicts with existing plan phases
   - Distinguish: is this a plan gap or a code bug? Only plan gaps belong here.

2. **Be Surgical**:
   - Precise edits, not wholesale rewrites
   - Preserve good content that doesn't need changing
   - Only research what's necessary for the specific changes
   - The `↺ Additions` pattern exists specifically to avoid rewriting completed phases

3. **Be Thorough**:
   - Read the entire existing plan before making changes
   - Read the review/test findings file before making changes
   - Research code patterns if changes require new technical understanding
   - Verify success criteria are measurable and use the correct test runner syntax

4. **Be Interactive**:
   - Confirm understanding before making changes
   - Show what you plan to change before doing it
   - Allow course corrections

5. **No Open Questions**:
   - If a requested change raises questions, ASK
   - Do NOT update the plan with unresolved questions
   - Every change must be complete and actionable

---

## Success Criteria Guidelines

When updating success criteria, maintain the two-category structure:

1. **Automated Verification** (runnable by agents):
   - Commands using the project's test runner (`pytest`, `npm test`, `go test ./...`, etc.)
   - Specific files that should exist
   - Compilation / type-checking / linting commands

2. **Manual Verification** (requires human):
   - UI/UX functionality
   - Performance under real conditions
   - Edge cases that are hard to automate

---

## Example Interaction Flows

**Scenario 1: review-impl routes here with a findings file**
```
User: /iterate_plan thoughts/shared/plans/2026-03-06-openrouter-llm-integration.md \
        --review thoughts/shared/reviews/2026-03-06-phase-2-review.md
Assistant: [Reads plan, reads review file, identifies that Phase 2 is complete but 
           missing retry logic → applies Rule 2 (↺ Additions block), confirms, updates]
```

**Scenario 2: No --review flag, auto-discovers findings**
```
User: /iterate_plan thoughts/shared/plans/2026-03-06-openrouter-llm-integration.md
      — accuracy is too low on phase 3
Assistant: [Reads plan, lists thoughts/shared/reviews/, finds 
           2026-03-06-phase-3-test.md, reads it, proceeds with changes]
```

**Scenario 3: User provides everything upfront**
```
User: /iterate_plan thoughts/shared/plans/2025-10-16-feature.md - add phase for error handling
Assistant: [Reads plan, researches error handling patterns, applies Rule 3 (new phase), 
           confirms, updates]
```

**Scenario 4: Review finding is a code bug, not a plan gap**
```
User: /iterate_plan ... — the test for empty notes is failing
Assistant: That's a code bug, not a plan gap. The plan already specifies 
           test_handles_empty_notes and the correct behaviour. Run:
           /implement_plan [path]
           Tell it: "Fix test_handles_empty_notes — empty input causes KeyError at parser.py:67"
```
