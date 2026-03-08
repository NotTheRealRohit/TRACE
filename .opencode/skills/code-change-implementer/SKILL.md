---
name: code-change-implementer
description: Carefully implement code changes into existing files while preserving execution flow, respecting line numbers, and validating syntax and library compatibility. Use this skill whenever the user asks to apply changes, edits, fixes, or improvements to an existing codebase — especially when a detailed implementation plan or evaluation report is provided. Always trigger this skill when the task involves modifying existing code rather than writing it from scratch, when multiple interdependent changes must be applied in order, or when the user says "implement the plan", "apply the fixes", "make these changes", or "update the code".
---

# Code Change Implementer

This skill governs how to apply code changes to existing files with surgical precision — preserving execution flow, respecting the original structure, and never introducing breakage through careless placement or untested assumptions.

The user will typically provide one or more of:
- An **existing source file** to be modified
- A **detailed implementation plan** describing what to change and where
- An **evaluation report** identifying bugs, accuracy issues, or structural problems
- A combination of all three

Your job is to implement exactly what is described, in the correct order, with verified syntax — and to test your own work before declaring it done.

---

## Phase 0 — Read Before You Write

Before touching a single line of code:

1. **Read the full source file** using the `view` tool. Do not skip sections. If the file is long, read it in range-chunks until you have seen every line. Pay attention to:
   - The overall structure (imports → constants → helpers → classes → main functions → entry point)
   - The execution sequence — which functions call which, and in what order
   - Every variable name, function signature, and return type that your changes will interact with

2. **Read the implementation plan in full** before starting. If a plan is provided, it is the authoritative specification. Identify:
   - Total number of change locations
   - Dependencies between changes (e.g., Change 3 must follow Change 1 because it uses a variable defined there)
   - Any explicit warnings about ordering or placement

3. **Map plan steps to line numbers** in the actual file. Write down the anchor line for each change — the exact line after which, or on which, the modification occurs. Do not rely on memory; re-read the relevant section of the file immediately before editing it.

> The most common source of breakage is editing from memory rather than from a confirmed view of the current file state.

---

## Phase 1 — Library and Syntax Verification

Before writing any new code, verify that every library, method, and syntax pattern you intend to use actually exists in the version available in the environment.

### Step 1.1 — Check installed library versions
```bash
pip show <library-name>          # e.g. pip show scikit-learn pandas numpy
python -c "import <lib>; print(<lib>.__version__)"
```

### Step 1.2 — For any method or parameter you are not 100% certain about, verify it
Common failure points:
- `sparse_output` vs `sparse` in `OneHotEncoder` (changed in sklearn 1.2)
- `cross_val_predict` `method` parameter name
- `pd.cut` `labels` parameter accepting specific types
- `hstack` sparse/dense type requirements

**If in doubt, search.** Use web search before writing the code:
```
web_search: "sklearn OneHotEncoder sparse_output parameter version"
web_search: "pandas pd.cut astype str NaN behaviour"
web_search: "scipy hstack dense sparse matrix mixing"
```

Read the official documentation page, not just a snippet. Confirm the exact parameter name, its type, and the version it was introduced.

### Step 1.3 — Check for breaking changes between versions
If the plan was written against a different library version than what is installed, there may be renamed parameters, changed defaults, or deprecated methods. Always check the changelog when the installed version is older than expected.

---

## Phase 2 — Implement in Plan Order

Apply changes **strictly in the order the plan specifies**. Never reorder steps, even if it seems harmless. Plans are written with execution dependency in mind — a variable declared in Change 3 may be required by Change 5. Skipping ahead or reordering causes `NameError`, `KeyError`, or silent incorrect behaviour.

### For each change:

**2a. Re-read the target section** — use `view` with a line range covering ±10 lines around the anchor point. Confirm the file currently looks exactly as the plan assumes it does before editing. If a previous change has shifted line numbers, adjust your anchor accordingly.

**2b. Identify the edit type:**
- **Insert** — new lines added between two existing lines
- **Replace** — existing line(s) swapped for new line(s)
- **Append** — new lines added at end of a block (e.g., end of a `hstack` call, end of a dict literal)

**2c. Write the edit using `str_replace`** — provide the smallest unique `old_str` that unambiguously identifies the location. Include enough surrounding context (1–3 lines) to be unique. Never use a one-word old_str if it appears elsewhere in the file.

**2d. After each edit, immediately re-read the modified section** to confirm the change looks exactly as intended. Check:
- Correct indentation (Python is indentation-sensitive)
- No missing commas in function arguments or dict literals
- No missing closing brackets from a multi-line expression
- Import additions did not break the existing import line

**2e. Annotate your progress** — after each successful edit, note "Change N complete — anchor line now at approx line X" so subsequent edits use the correct shifted line numbers.

---

## Phase 3 — Execution Sequence Integrity Check

After all changes are applied, read the modified function or module from top to bottom and verify the execution sequence is correct.

For each new variable or object introduced, answer:
- Is it **defined before it is used**? Trace the first use back to the definition. If the definition comes after the use in file order, this is a bug.
- Is it **defined in the right scope**? A variable defined inside an `if` block is not available outside it.
- Does it **carry the correct type** into the next step? If `fit_transform` returns a sparse matrix and the next step expects a dense array, wrapping is needed.

For sequential data pipelines specifically (the most common context for this skill), verify the **feature matrix column order** is identical across:
1. The training assembly (`hstack` in training function)
2. The test assembly (`hstack` in evaluation section)
3. The inference assembly (`hstack` in prediction/scoring function)

All three must have identical column order. A mismatch here produces wrong predictions with no error message — the most dangerous class of bug.

Use a comment block as a column-order manifest if this is a concern:
```python
# Column order: [complaint_ohe | dtc_tfidf | dtc_flags | voltage | supplier_ohe | mileage | year | NEW_mb | NEW_vb | NEW_ca]
# This must match run_ml() hstack and train_and_save() X_te hstack exactly.
```

---

## Phase 4 — Testing

Run tests **in this sequence** — from cheapest to most expensive. Stop and fix before advancing if a test fails.

### T1 — Syntax check (free, instant)
```bash
python -m py_compile <filename>.py && echo "Syntax OK"
```
Catches syntax errors, mismatched brackets, bad indentation. Must pass before running anything else.

### T2 — Import check (free, ~1 second)
```bash
python -c "import <module_name>; print('Import OK')"
```
Catches missing dependencies, circular imports, module-level errors (e.g., a function call at import time that fails).

### T3 — Unit test on new helpers (cheap, seconds)
For any new standalone helper function (e.g., `voltage_band`, feature engineering functions), test it directly with known inputs and expected outputs:
```bash
python -c "
from module import new_function
assert new_function(input1) == expected1, f'Got {new_function(input1)}'
assert new_function(input2) == expected2
print('Helper tests passed')
"
```

### T4 — Shape / type assertion test (medium, depends on data load)
For data pipeline changes, verify that the assembled feature matrices have the correct shape and type:
```bash
python -c "
# Run only the feature engineering + assembly, assert shapes
from module import train_and_save
bundle = train_and_save()
# Assert new bundle keys exist
assert 'new_transformer_key' in bundle, 'Missing bundle key'
print('Shape/type tests passed')
"
```

### T5 — End-to-end inference test (medium-high)
Run a small set of known-output cases through the full prediction pipeline:
```bash
python -c "
from module import predict
result = predict(known_input_1, known_input_2, known_input_3)
assert result['key'] == expected_value, f'Got {result}'
print('Inference test passed')
"
```

### T6 — Accuracy regression (high cost — run last)
Only after all prior tests pass, run the full evaluation to confirm the accuracy metric improved or held steady compared to the baseline reported in the evaluation output.

---

## Phase 5 — Handling Errors

### Syntax errors from `str_replace`
If `str_replace` fails because `old_str` is not found:
1. Re-read the current file with `view` — do not guess
2. The previous edit may have shifted content or introduced subtle whitespace differences
3. Use a shorter, more specific `old_str` that is guaranteed unique

### Import errors at runtime
```bash
python -c "import module" 2>&1
```
Read the full traceback. It shows exactly which line failed and why. Never guess — the traceback is the answer.

### Unexpected `AttributeError` or `TypeError` on a library method
This usually means a version mismatch. Run:
```bash
pip show <library>
```
Then web search: `"<library> <version> <method_name> parameter <param>"` and confirm the correct syntax for that specific version.

### `KeyError` on a bundle or dict
The key was not added during save (training phase) but is being read at inference. Check:
1. The dict/bundle definition in the training function — is the key present?
2. The key name spelling — copy-paste from the definition to the read site, do not retype

### Silent wrong results (no error, but incorrect output)
This is the hardest failure mode. If all tests pass but accuracy is worse or predictions are wrong:
1. Check feature matrix column order across all three assembly sites (Phase 3)
2. Check that `fit_transform` was not accidentally called on test data
3. Check that the OOF/cascade pattern uses the correct model (inference model vs. OOF helper model)
4. Add shape-print debugging temporarily: `print("X shape:", X.shape)` in both training and inference paths and confirm they match

---

## Key Rules (Never Violate)

- **Never edit from memory.** Always `view` the target section immediately before editing it.
- **Never reorder plan steps** unless you have explicitly traced all variable dependencies and confirmed safety.
- **Never call `fit_transform` on test data.** Train slice: `fit_transform`. Test/inference slice: `transform` only.
- **Never assume a library method signature.** Verify from docs or web search if there is any doubt.
- **Never declare a change complete without re-reading the edited section.**
- **Never skip T1 (syntax check).** It costs nothing and catches the most common errors immediately.
- **Preserve existing behaviour for code paths you did not change.** After edits, check that the untouched code paths (e.g., fallback branches, alternative conditions) still function correctly by tracing their variable dependencies through the new code.

---

## Quick Reference: Common Edit Patterns

### Adding an import to an existing `from X import A, B` line
```python
# old_str
from sklearn.model_selection import train_test_split
# new_str
from sklearn.model_selection import train_test_split, cross_val_predict
```

### Inserting a block after a specific line (use the line as anchor)
```python
# old_str — the line immediately before the insertion point
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)

# new_str — original line + new block below it
    df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)

    # New block starts here
    df["new_feature"] = df["existing_col"].apply(some_function)
```

### Extending a multi-line `hstack` call
```python
# old_str
    X_tr = hstack([X_c_tr, X_d_tr, csr_matrix(X_n_tr),
                   X_s_tr, csr_matrix(X_m_tr)])
# new_str
    X_tr = hstack([X_c_tr, X_d_tr, csr_matrix(X_n_tr),
                   X_s_tr, csr_matrix(X_m_tr),
                   X_new_tr, csr_matrix(X_another_tr)])
```

### Adding keys to a dict literal (bundle save)
```python
# old_str
    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd,
                  ohe=ohe, scaler=scaler)
# new_str
    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd,
                  ohe=ohe, scaler=scaler,
                  new_transformer=new_transformer)
```

### Replacing a single problematic line inside a block
```python
# old_str — include one line of context above and below for uniqueness
    clf_fa.fit(X_tr, yfa_tr)

    fa_probs_tr = clf_fa.predict_proba(X_tr)   # ← the line to replace
    fa_probs_te = clf_fa.predict_proba(X_te)

# new_str
    clf_fa.fit(X_tr, yfa_tr)

    fa_probs_tr = cross_val_predict(            # ← replacement
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_tr, yfa_tr, cv=5, method="predict_proba"
    )
    fa_probs_te = clf_fa.predict_proba(X_te)
```
