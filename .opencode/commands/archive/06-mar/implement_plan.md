---
description: Implement technical plans from thoughts/shared/plans with verification, following TDD Red-Green-Refactor methodology
---

# Implement Plan

You are tasked with implementing an approved technical plan from `thoughts/shared/plans/`. These plans contain phases with specific changes and success criteria.

**TDD is mandatory.** Every production code change must be preceded by a failing test. No exceptions.

## Getting Started

When given a plan path:
- Read the plan completely and check for any existing checkmarks (- [x])
- Read the original ticket and all files mentioned in the plan
- **Read files fully** - never use limit/offset parameters, you need complete context
- Identify the project's language, test runner, and test command from the plan's **Project Context** section
- Think deeply about how the pieces fit together
- Create a todo list to track your progress
- Start implementing if you understand what needs to be done

If no plan path provided, ask for one.

## Implementation Philosophy

Plans are carefully designed, but reality can be messy. Your job is to:
- Follow the plan's intent while adapting to what you find
- Implement each phase fully before moving to the next
- **Follow Red-Green-Refactor strictly** — write a failing test before any production code
- Verify your work makes sense in the broader codebase context
- Update checkboxes in the plan as you complete sections

When things don't match the plan exactly, think about why and communicate clearly. The plan is your guide, but your judgment matters too.

If you encounter a mismatch:
- STOP and think deeply about why the plan can't be followed
- Present the issue clearly:
  ```
  Issue in Phase [N]:
  Expected: [what the plan says]
  Found: [actual situation]
  Why this matters: [explanation]

  How should I proceed?
  ```

---

## Identifying the Test Runner

Before starting, confirm the project's test tooling from the plan's **Project Context** section or by inspecting the repo:

| Signal | Likely Stack |
|--------|-------------|
| `package.json` with `jest` / `vitest` / `mocha` | JavaScript / TypeScript |
| `pytest.ini`, `pyproject.toml`, `setup.cfg` | Python |
| `go.mod` | Go |
| `Gemfile` with `rspec` | Ruby |
| `Cargo.toml` | Rust |
| `build.gradle` / `pom.xml` | JVM (Kotlin / Groovy / Java) |
| `mix.exs` | Elixir |
| `*.csproj` / `*.sln` | .NET / C# |

Use the test runner and filter syntax native to that stack throughout the rest of this workflow.

---

## TDD Workflow Per Phase

For **every** production code change in each phase, follow this cycle:

### 🔴 RED — Write a Failing Test First

1. Create or update a test file in the appropriate test directory for the project
2. Name the test using the project's naming convention:
   - JavaScript/TypeScript: `it('should <expected behaviour> when <condition>')`
   - Python: `def test_<expected_behaviour>_when_<condition>:`
   - Go: `func Test<ExpectedBehaviour>When<Condition>(t *testing.T)`
   - Ruby: `it 'should <expected behaviour> when <condition>'`
   - Rust: `fn test_<expected_behaviour>_when_<condition>()`
   - Adapt as needed for the project's conventions
3. Run the test to **confirm it fails**:
   ```bash
   # Replace with the project's actual test runner and filter syntax:
   [test-runner] [test-filter-flag] "[TestName]"

   # Examples:
   # JavaScript/TypeScript (Jest):   npx jest --testNamePattern "throws when user not found"
   # JavaScript/TypeScript (Vitest): npx vitest run --reporter=verbose -t "throws when user not found"
   # Python (pytest):                pytest -k "test_throws_when_user_not_found"
   # Go:                             go test ./... -run TestThrowsWhenUserNotFound
   # Ruby (RSpec):                   bundle exec rspec --example "throws when user not found"
   # Rust:                           cargo test throws_when_user_not_found
   # .NET:                           dotnet test --filter "DisplayName~ThrowsWhenUserNotFound"
   ```
4. **Do not proceed until the test fails with a meaningful error.**

Choose the right test scope for the code being tested:

| Scope | Use When |
|-------|----------|
| Unit | Isolated function/class/module with mocked dependencies |
| Integration | Multiple real components interacting |
| API / Controller | HTTP request/response contracts |
| Repository / DB | Data access layer, queries, migrations |
| End-to-End | Full user-facing flows through the system |

### 🟢 GREEN — Write Minimum Implementation

1. Write the **smallest amount of code** to make the failing test pass
2. Avoid adding features or logic not yet covered by a test
3. Run the specific test to confirm it passes:
   ```bash
   [test-runner] [test-filter-flag] "[TestName]"
   ```
4. Run **all tests** to confirm no regressions:
   ```bash
   [full test suite command for this module/package]
   ```

### 🔵 REFACTOR — Improve Without Breaking

1. Identify duplication, naming issues, or code smells
2. Make small, incremental improvements
3. Run all tests after **every change**:
   ```bash
   [full test suite command]
   ```
4. **Immediately revert** if any test goes red

---

## Pre-Code Checklist (Complete Before Writing Production Code)

Before implementing any production change, confirm:

- [ ] **RED Phase Complete**: Written a failing test first
- [ ] **Test Confirmed Failing**: Ran the test and verified it fails with a meaningful error
- [ ] **Test Naming Correct**: Follows the project's naming convention
- [ ] **Test Location Correct**: Placed in the correct test directory for this project
- [ ] **Correct Test Scope Selected**: Unit / Integration / API / Repository / E2E based on what's being tested
- [ ] **Understands What's Being Tested**: Can explain the behavior the test validates
- [ ] **Minimal Implementation Ready**: Know the smallest code needed to pass

**DO NOT write production code until all items are checked.**

---

## Verification Approach

After implementing a phase:
- Run the success criteria checks defined in the plan (the plan's automated verification section specifies the exact commands)
- Fix any issues before proceeding
- Update your progress in both the plan and your todos
- Check off completed items in the plan file itself using Edit
- **Pause for human verification**: After completing all automated verification for a phase, pause and inform the human that the phase is ready for manual testing. Use this format:
  ```
  Phase [N] Complete - Ready for Manual Verification

  Automated verification passed:
  - [List automated checks that passed]

  Please perform the manual verification steps listed in the plan:
  - [List manual verification items from the plan]

  Let me know when manual testing is complete so I can proceed to Phase [N+1].
  ```

If instructed to execute multiple phases consecutively, skip the pause until the last phase. Otherwise, assume you are doing just one phase.

Do not check off items in the manual testing steps until confirmed by the user.

---

## If You Get Stuck

When something isn't working as expected:
- First, make sure you've read and understood all the relevant code
- Consider if the codebase has evolved since the plan was written
- Present the mismatch clearly and ask for guidance

Use sub-tasks sparingly — mainly for targeted debugging or exploring unfamiliar territory.

## Resuming Work

If the plan has existing checkmarks:
- Trust that completed work is done
- Pick up from the first unchecked item
- Verify previous work only if something seems off

Remember: You're implementing a solution, not just checking boxes. Keep the end goal in mind and maintain forward momentum.
