---
name: workflow-orchestrator
description: >
  This skill tells you which command to run, when to run it, and how to route when things fail. Read this before starting any new requirement or when you're unsure where you are in the workflow.
---

# Workflow Orchestrator Skill

## Purpose

This skill tells you which command to run, when to run it, and how to route when things fail. Read this before starting any new requirement or when you're unsure where you are in the workflow.

The workflow is built on five core commands:
- `research_codebase` — understand what already exists
- `create_plan` — produce a phased, TDD-first implementation plan
- `implement_plan` — execute the plan one phase at a time
- `review-impl` — code quality gate after each phase
- `test-impl` — functional gate after review passes
- `iterate_plan` — update the plan when reality diverges from it
- `commit` — clean atomic commits after all gates pass

---

## Stage Detection — Where Are You Right Now?

Before doing anything, identify which stage you're at:

| Situation | Stage | Start here |
|-----------|-------|------------|
| Brand new requirement, nothing exists yet | 0 — Fresh start | `research_codebase` |
| Research done, no plan file yet | 1 — Planning | `create_plan` |
| Plan file exists, no checkboxes ticked | 2 — Ready to implement | `implement_plan` |
| Plan file exists, some checkboxes ticked | 3 — Mid-implementation | `implement_plan` (it auto-resumes) |
| Phase complete, not yet reviewed | 4 — Needs review | `review-impl` |
| Review passed, not yet tested | 5 — Needs functional test | `test-impl` |
| Tests passed, changes not committed | 6 — Ready to commit | `commit` |
| Review or test failed | 7 — Fixing failures | See Failure Routing below |

---

## The Complete Workflow

```
NEW REQUIREMENT
      │
      ▼
┌─────────────────┐
│ research_codebase│  ← Understand existing pipeline, models, patterns
└────────┬────────┘
         │ findings fed into ↓
         ▼
┌─────────────────┐
│   create_plan   │  ← Interactive. Produces plan file in thoughts/shared/plans/
└────────┬────────┘
         │ plan approved by human ↓
         ▼
┌─────────────────────────────────────────────────────────┐
│                    PHASE LOOP                           │
│                                                         │
│  ┌──────────────┐                                       │
│  │implement_plan│  ← One phase at a time, TDD strictly  │
│  └──────┬───────┘                                       │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ review-impl  │  ← Code quality gate                  │
│  └──────┬───────┘                                       │
│         │ PASS ↓                                        │
│  ┌──────────────┐                                       │
│  │  test-impl   │  ← Functional gate                    │
│  └──────┬───────┘                                       │
│         │ PASS ↓                                        │
│         │ More phases? ──── YES ──→ back to implement   │
│         │ No more phases? ↓                             │
└─────────┼───────────────────────────────────────────────┘
          ▼
     ┌─────────┐
     │ commit  │
     └─────────┘
```

---

## Failure Routing

This is the most important section. When a gate fails, the cause determines the fix.

### `review-impl` fails

```
review-impl FAIL
      │
      ├── BLOCKER/MAJOR in implementation code
      │   (plan was correct, code is wrong)
      │   ↓
      │   /implement_plan [plan-path]
      │   Tell it: "Fix these review findings using TDD: [paste findings]"
      │   Then: /review-impl [plan-path]
      │
      └── BLOCKER because plan itself is incomplete or wrong
          (plan didn't account for something real)
          ↓
          /iterate_plan [plan-path] — [describe what the plan missed]
          Then: /implement_plan [plan-path]
          Then: /review-impl [plan-path]
```

### `test-impl` fails

```
test-impl FAIL
      │
      ├── Suite 1 (Schema) or Suite 3 (Edge cases) — code bug
      │   ↓
      │   /implement_plan [plan-path]
      │   Tell it: "Fix these failing cases using TDD — write failing test first:
      │            [paste exact inputs and expected outputs]"
      │   Then: /test-impl [plan-path]
      │
      ├── Suite 2 (Accuracy below 90%) — prompt engineering issue
      │   ↓
      │   /iterate_plan [plan-path]
      │   — "Accuracy at XX%. Confused pairs: [list them].
      │      Update Phase 3 to add few-shot examples for these cases."
      │   Then: /implement_plan [plan-path]
      │   Then: /test-impl [plan-path]
      │
      ├── Suite 4 (API resilience) — error handling bug
      │   ↓
      │   /implement_plan [plan-path]
      │   Tell it: "Add handling for [specific failure mode] using TDD"
      │   Then: /test-impl [plan-path]
      │
      ├── Suite 5 (Pipeline regression) — integration design issue
      │   ↓
      │   /iterate_plan [plan-path]
      │   — "Pipeline broke. Rethink Phase 4 integration to decouple LLM layer."
      │   Then: /implement_plan [plan-path]
      │   Then: /test-impl [plan-path]
      │
      └── Suite 6 (Consistency) — prompt stability issue
          ↓
          /iterate_plan [plan-path]
          — "Results inconsistent across runs for: [inputs].
             Add temperature or determinism constraints to Phase 3."
          Then: /implement_plan [plan-path]
          Then: /test-impl [plan-path]
```

---

## Command Invocation Reference

### Starting fresh
```
/research_codebase
→ [provide your research question about the existing codebase]
```

### Creating a plan
```
/create_plan
→ [paste requirement description, context, example inputs/outputs]
```
Feed `research_codebase` findings into `create_plan` under `## Existing Codebase Context`.
Let `create_plan` drive — it asks questions before writing anything.

### Implementing
```
/implement_plan thoughts/shared/plans/YYYY-MM-DD-description.md
```
- It auto-resumes from first unchecked phase
- Confirm manual verification steps before it proceeds to next phase
- If it reports a mismatch, run `iterate_plan` before continuing

### Reviewing
```
/review-impl thoughts/shared/plans/YYYY-MM-DD-description.md

# Review a specific phase only:
/review-impl thoughts/shared/plans/YYYY-MM-DD-description.md --phase 3
```

### Testing
```
/test-impl thoughts/shared/plans/YYYY-MM-DD-description.md

# Test a specific phase only:
/test-impl thoughts/shared/plans/YYYY-MM-DD-description.md --phase 3
```

### Iterating the plan
```
/iterate_plan thoughts/shared/plans/YYYY-MM-DD-description.md
— [one clear description of the change needed]
```
Always confirm its understanding before it edits the plan file.

### Committing
```
/commit
```
It reviews git diff, proposes commit messages, waits for your approval.

---

## Rules That Never Change

1. **`research_codebase` before `create_plan`** — never plan against assumptions when you can verify
2. **`review-impl` before `test-impl`** — don't functionally test code that hasn't passed quality review
3. **`iterate_plan` before re-implementing when the plan is wrong** — keep the plan as the source of truth
4. **Never skip a gate** — a phase is not done until both review and test pass
5. **One phase at a time** — don't ask `implement_plan` to do multiple phases in one run unless you explicitly accept reduced verification
6. **Failures route to the right fix** — a test failure is not always a code bug; check the failure routing table before acting

---

## Signs You're in the Wrong Stage

| What you're seeing | What it means | Correct action |
|--------------------|---------------|----------------|
| `implement_plan` reports a mismatch between plan and codebase | Plan is stale | `iterate_plan` first |
| `review-impl` finds zero test coverage for a phase | TDD wasn't followed | Re-implement that phase from RED |
| `test-impl` Suite 2 fails but Suite 1 passes | Prompt issue, not code | `iterate_plan` Phase 3 |
| `test-impl` Suite 5 fails | Pipeline coupling issue | `iterate_plan` Phase 4 |
| `commit` shows hundreds of unrelated changed files | Scope crept | Review "What We're NOT Doing" in plan |

---

## Checkpointing

At any point you can see where you are by running:
```bash
# See which plan phases are complete
grep -E "- \[.\]" thoughts/shared/plans/YYYY-MM-DD-description.md

# See what's changed since last commit
git diff --name-only

# See recent commits
git log --oneline -10
```

Use these to orient yourself if you return to a task after a break before invoking any command.
