---
description: Create detailed implementation plans with thorough research and iteration, incorporating TDD Red-Green-Refactor methodology
---

# Implementation Plan

You are tasked with creating detailed implementation plans through an interactive, iterative process. You should be skeptical, thorough, and work collaboratively with the user to produce high-quality technical specifications.

**All plans must be TDD-first.** Every phase must include test cases written before production code, following the Red-Green-Refactor cycle.

## Initial Response

When this command is invoked:

1. **Check if parameters were provided**:
   - If a file path or ticket reference was provided as a parameter, skip the default message
   - Immediately read any provided files FULLY
   - Begin the research process

2. **If no parameters provided**, respond with:
```
I'll help you create a detailed implementation plan. Let me start by understanding what we're building.

Please provide:
1. The task/ticket description (or reference to a ticket file)
2. Any relevant context, constraints, or specific requirements
3. Links to related research or previous implementations

I'll analyze this information and work with you to create a comprehensive plan.

Tip: You can also invoke this command with a ticket file directly: `/create_plan thoughts/allison/tickets/eng_1234.md`
For deeper analysis, try: `/create_plan think deeply about thoughts/allison/tickets/eng_1234.md`
```

Then wait for the user's input.

## Process Steps

### Step 1: Context Gathering & Initial Analysis

1. **Read all mentioned files immediately and FULLY**:
   - Ticket files (e.g., `thoughts/allison/tickets/eng_1234.md`)
   - Research documents (e.g., `thoughts/shared/research/<fileName>.md`)
   - Related implementation plans
   - Any JSON/data files mentioned
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files
   - **CRITICAL**: DO NOT spawn sub-tasks before reading these files yourself in the main context
   - **NEVER** read files partially — if a file is mentioned, read it completely

2. **Spawn initial research tasks to gather context**:
   Before asking the user any questions, use specialized agents to research in parallel:

   - Use the **codebase-locator** agent to find all files related to the ticket/task
   - Use the **codebase-analyzer** agent to understand how the current implementation works
   - If relevant, use the **thoughts-locator** agent to find any existing thoughts documents about this feature
   - If a Linear ticket is mentioned, use the **linear-ticket-reader** agent to get full details

   These agents will:
   - Find relevant source files, configs, and tests
   - Trace data flow and key functions
   - Return detailed explanations with file:line references

3. **Read all files identified by research tasks**:
   - After research tasks complete, read ALL files they identified as relevant
   - Read them FULLY into the main context
   - This ensures you have complete understanding before proceeding

4. **Analyze and verify understanding**:
   - Cross-reference the ticket requirements with actual code
   - Identify any discrepancies or misunderstandings
   - Note assumptions that need verification
   - Determine true scope based on codebase reality

5. **Present informed understanding and focused questions**:
   ```
   Based on the ticket and my research of the codebase, I understand we need to [accurate summary].

   I've found that:
   - [Current implementation detail with file:line reference]
   - [Relevant pattern or constraint discovered]
   - [Potential complexity or edge case identified]

   Questions that my research couldn't answer:
   - [Specific technical question that requires human judgment]
   - [Business logic clarification]
   - [Design preference that affects implementation]
   ```

   Only ask questions that you genuinely cannot answer through code investigation.

### Step 2: Research & Discovery

After getting initial clarifications:

1. **If the user corrects any misunderstanding**:
   - DO NOT just accept the correction
   - Spawn new research tasks to verify the correct information
   - Read the specific files/directories they mention
   - Only proceed once you've verified the facts yourself

2. **Create a research todo list** using TodoWrite to track exploration tasks

3. **Spawn parallel sub-tasks for comprehensive research**:
   - Create multiple Task agents to research different aspects concurrently
   - Use the right agent for each type of research:

   **For deeper investigation:**
   - **codebase-locator** - To find more specific files
   - **codebase-analyzer** - To understand implementation details
   - **codebase-pattern-finder** - To find similar features we can model after

   **For historical context:**
   - **thoughts-locator** - To find any research, plans, or decisions about this area
   - **thoughts-analyzer** - To extract key insights from the most relevant documents

   **For related tickets:**
   - **linear-searcher** - To find similar issues or past implementations

4. **Wait for ALL sub-tasks to complete** before proceeding

5. **Present findings and design options**:
   ```
   Based on my research, here's what I found:

   **Current State:**
   - [Key discovery about existing code]
   - [Pattern or convention to follow]

   **Design Options:**
   1. [Option A] - [pros/cons]
   2. [Option B] - [pros/cons]

   **Open Questions:**
   - [Technical uncertainty]
   - [Design decision needed]

   Which approach aligns best with your vision?
   ```

### Step 3: Plan Structure Development

Once aligned on approach:

1. **Create initial plan outline**:
   ```
   Here's my proposed plan structure:

   ## Overview
   [1-2 sentence summary]

   ## Implementation Phases:
   1. [Phase name] - [what it accomplishes]
   2. [Phase name] - [what it accomplishes]
   3. [Phase name] - [what it accomplishes]

   Does this phasing make sense? Should I adjust the order or granularity?
   ```

2. **Get feedback on structure** before writing details

### Step 4: Detailed Plan Writing

After structure approval:

1. **Write the plan** to `thoughts/shared/plans/YYYY-MM-DD-ENG-XXXX-description.md`
   - Format: `YYYY-MM-DD-ENG-XXXX-description.md` where:
     - YYYY-MM-DD is today's date
     - ENG-XXXX is the ticket number (omit if no ticket)
     - description is a brief kebab-case description
   - Examples:
     - With ticket: `2025-01-08-ENG-1478-parent-child-tracking.md`
     - Without ticket: `2025-01-08-improve-error-handling.md`

2. **Use this template structure**:

````markdown
# [Feature/Task Name] Implementation Plan

## Overview

[Brief description of what we're implementing and why]

## Current State Analysis

[What exists now, what's missing, key constraints discovered]

## Desired End State

[A specification of the desired end state after this plan is complete, and how to verify it]

### Key Discoveries:
- [Important finding with file:line reference]
- [Pattern to follow]
- [Constraint to work within]

## What We're NOT Doing

[Explicitly list out-of-scope items to prevent scope creep]

## Implementation Approach

[High-level strategy and reasoning]

---

## Phase 1: [Descriptive Name]

### Overview
[What this phase accomplishes]

### TDD — Tests to Write First

Before writing any production code in this phase, write these failing tests:

| Test Class | Test Method | Validates |
|------------|-------------|-----------|
| `FooServiceTest` | `shouldThrowExceptionWhenUserNotFound` | Throws when user lookup fails |
| `FooServiceTest` | `shouldSucceedWhenValidInput` | Happy path succeeds |

**Run each test to confirm RED before implementing:**
```bash
./mvnw -pl <module> test -Dtest=FooServiceTest#shouldThrowExceptionWhenUserNotFound
```

### Changes Required:

#### 1. [Component/File Group]
**File**: `path/to/file.ext`
**Changes**: [Summary of changes]

```[language]
// Specific code to add/modify
```

### Success Criteria:

#### Automated Verification:
- [ ] RED confirmed: failing tests written and confirmed before implementation
- [ ] GREEN achieved: `./mvnw -pl <module> test -Dtest=<TestClass>`
- [ ] No regressions: `./mvnw -pl <module> test`
- [ ] Code compiles cleanly: `./mvnw -pl <module> compile`

#### Manual Verification:
- [ ] [Feature works as expected when tested via UI/API]
- [ ] [Edge case handling verified manually]
- [ ] [No regressions in related features]

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 2: [Descriptive Name]

[Similar structure — TDD tests first, then changes, then success criteria]

---

## Testing Strategy

### Unit Tests (follow Red-Green-Refactor):
- **CartServiceTest**:
  - `shouldThrowExceptionWhenUserNotFound`
  - `shouldThrowExceptionWhenProductNotFound`
  - `shouldSucceedWhenUserAndProductExist`
- [Other test classes and methods]

Test naming convention: `should<ExpectedBehaviour>When<Condition>`

| Test Type | Annotation | Use When |
|-----------|------------|----------|
| Unit | `@ExtendWith(MockitoExtension.class)` | Service business logic |
| Controller Slice | `@WebMvcTest` | REST endpoints |
| Repository Slice | `@DataJpaTest` | Database queries |
| Integration | `@SpringBootTest` | End-to-end flows |

### Integration Tests:
- [End-to-end scenarios]

### Manual Testing Steps:
1. [Specific step to verify feature]
2. [Another verification step]
3. [Edge case to test manually]

## Performance Considerations

[Any performance implications or optimizations needed]

## Migration Notes

[If applicable, how to handle existing data/systems]

## References

- Original ticket: `thoughts/allison/tickets/eng_XXXX.md`
- Related research: `thoughts/shared/research/[relevant].md`
- TDD Skill: `.opencode/skills/tdd-red-green-refactor/SKILL.md`
- Similar implementation: `[file:line]`
````

### Step 5: Sync and Review

1. **Sync the thoughts directory** to ensure the plan is properly indexed

2. **Present the draft plan location**:
   ```
   I've created the initial implementation plan at:
   `thoughts/shared/plans/YYYY-MM-DD-ENG-XXXX-description.md`

   Please review it and let me know:
   - Are the phases properly scoped?
   - Are the TDD test cases specific enough to guide implementation?
   - Are the success criteria measurable?
   - Any technical details that need adjustment?
   - Missing edge cases or considerations?
   ```

3. **Iterate based on feedback** — be ready to:
   - Add missing phases
   - Adjust technical approach
   - Add more test cases to the TDD table
   - Clarify success criteria (automated vs manual)
   - Add/remove scope items

4. **Continue refining** until the user is satisfied

---

## Important Guidelines

1. **TDD is Non-Negotiable**:
   - Every phase must include a "Tests to Write First" section
   - Tests must be named `should<ExpectedBehaviour>When<Condition>`
   - The plan must specify RED confirmation before GREEN implementation
   - Production code without a prior failing test is a plan defect

2. **Be Skeptical**:
   - Question vague requirements
   - Identify potential issues early
   - Ask "why" and "what about"
   - Don't assume — verify with code

3. **Be Interactive**:
   - Don't write the full plan in one shot
   - Get buy-in at each major step
   - Allow course corrections
   - Work collaboratively

4. **Be Thorough**:
   - Read all context files COMPLETELY before planning
   - Research actual code patterns using parallel sub-tasks
   - Include specific file paths and line numbers
   - Write measurable success criteria with clear automated vs manual distinction

5. **Be Practical**:
   - Focus on incremental, testable changes
   - Consider migration and rollback
   - Think about edge cases
   - Include "what we're NOT doing"

6. **Track Progress**:
   - Use TodoWrite to track planning tasks
   - Update todos as you complete research
   - Mark planning tasks complete when done

7. **No Open Questions in Final Plan**:
   - If you encounter open questions during planning, STOP
   - Research or ask for clarification immediately
   - Do NOT write the plan with unresolved questions
   - The implementation plan must be complete and actionable
   - Every decision must be made before finalizing the plan

---

## Success Criteria Guidelines

**Always separate success criteria into two categories:**

1. **Automated Verification** (can be run by execution agents):
   - RED confirmed: failing test written and verified
   - GREEN achieved: test passes with minimal implementation
   - No regressions: full test suite still passes
   - Compilation and linting pass

2. **Manual Verification** (requires human testing):
   - UI/UX functionality
   - Performance under real conditions
   - Edge cases that are hard to automate
   - User acceptance criteria

**Format example:**
```markdown
### Success Criteria:

#### Automated Verification:
- [ ] RED confirmed: `./mvnw -pl order-service test -Dtest=CartServiceTest#shouldThrowExceptionWhenUserNotFound` fails first
- [ ] GREEN achieved: `./mvnw -pl order-service test -Dtest=CartServiceTest`
- [ ] No regressions: `./mvnw -pl order-service test`
- [ ] Compiles cleanly: `./mvnw -pl order-service compile`

#### Manual Verification:
- [ ] Add item to cart with invalid user returns 400
- [ ] Add item to cart with valid user and product succeeds
```

---

## Common Patterns

### For Database Changes:
- Start with schema/migration
- Add repository tests (`@DataJpaTest`) first
- Add store methods
- Add service tests, then business logic
- Add controller tests (`@WebMvcTest`), then expose via API
- Update clients

### For New Features:
- Research existing patterns first
- Start with data model
- Build backend logic — unit tests first for each service method
- Add API endpoints — controller slice tests first
- Implement UI last

### For Refactoring:
- Document current behavior with tests that capture it (characterization tests)
- Plan incremental changes
- Maintain backwards compatibility
- Include migration strategy

---

## Sub-task Spawning Best Practices

When spawning research sub-tasks:

1. **Spawn multiple tasks in parallel** for efficiency
2. **Each task should be focused** on a specific area
3. **Provide detailed instructions** including:
   - Exactly what to search for
   - Which directories to focus on
   - What information to extract
   - Expected output format
4. **Be EXTREMELY specific about directories** — include full path context
5. **Specify read-only tools** to use
6. **Request specific file:line references** in responses
7. **Wait for all tasks to complete** before synthesizing
8. **Verify sub-task results** — cross-check against the actual codebase

---

## Example Interaction Flow

```
User: /create_plan
Assistant: I'll help you create a detailed implementation plan...

User: We need to add interservice validation. See thoughts/allison/tickets/eng_1234.md
Assistant: Let me read that ticket file completely first...

[Reads file fully, researches codebase]

Based on the ticket and codebase research, here's what I understand...

[Proposes TDD-first phases, confirms test cases with user, writes plan]
```
