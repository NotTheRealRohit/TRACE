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

## TDD Workflow Per Phase

For **every** production code change in each phase, follow this cycle:

### 🔴 RED — Write a Failing Test First

1. Create or update a test class in `src/test/java/` matching the production package
2. Name the test using: `should<ExpectedBehaviour>When<Condition>`
3. Run the test to **confirm it fails**:
   ```bash
   ./mvnw test -Dtest=<TestClass>#<testMethod>
   # or for a specific module:
   ./mvnw -pl <module> test -Dtest=<TestClass>#<testMethod>
   ```
4. **Do not proceed until the test fails with a meaningful error.**

Choose the right test type for the layer being tested:

| Layer      | Test Type                              |
|------------|----------------------------------------|
| Service    | Unit test (`@ExtendWith(MockitoExtension.class)`) |
| Controller | Slice test (`@WebMvcTest`)             |
| Repository | Slice test (`@DataJpaTest`)            |
| End-to-End | Integration test (`@SpringBootTest`)   |

### 🟢 GREEN — Write Minimum Implementation

1. Write the **smallest amount of code** to make the failing test pass
2. Avoid adding features or logic not yet covered by a test
3. Run the specific test to confirm it passes:
   ```bash
   ./mvnw -pl <module> test -Dtest=<TestClass>#<testMethod>
   ```
4. Run **all tests** to confirm no regressions:
   ```bash
   ./mvnw -pl <module> test
   ```

### 🔵 REFACTOR — Improve Without Breaking

1. Identify duplication, naming issues, or code smells
2. Make small, incremental improvements
3. Run all tests after **every change**:
   ```bash
   ./mvnw -pl <module> test
   ```
4. **Immediately revert** if any test goes red

---

## Pre-Code Checklist (Complete Before Writing Production Code)

Before implementing any production change, confirm:

- [ ] **RED Phase Complete**: Written a failing test first
- [ ] **Test Confirmed Failing**: Ran the test and verified it fails with a meaningful error
- [ ] **Test Naming Correct**: Follows `should<Expected>When<Condition>` pattern
- [ ] **Test Location Correct**: Placed in `src/test/java/` matching the production package
- [ ] **Correct Test Type Selected**: Unit / Slice / Integration based on the layer
- [ ] **Understands What's Being Tested**: Can explain the behavior the test validates
- [ ] **Minimal Implementation Ready**: Know the smallest code needed to pass

**DO NOT write production code until all items are checked.**

---

## Verification Approach

After implementing a phase:
- Run the success criteria checks (usually `./mvnw -pl <module> test` covers unit/slice tests)
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