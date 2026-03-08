# ML Predictor — Warranty Accuracy Improvement Plan
**File:** `ml_predictor.py`  
**Goal:** Break the 88% warranty accuracy ceiling  
**Current Baselines:**
- Failure Analysis accuracy: **98.45%**
- Warranty Decision accuracy: **88.48%** ← target
- Warranty CV accuracy: **88.33% ± 0.25%**

---

## Table of Contents

1. [FIX 1 — Cascade Failure Analysis into Warranty Classifier](#fix-1)
2. [FIX 2 — Add Supplier and Mileage Features](#fix-2)
3. [FIX 3 — Enable Class Weights on Warranty Classifier](#fix-3)
4. [FIX 4 — Remove the Artificial 50% Confidence Floor](#fix-4)
5. [FIX 5 — Reuse predict_proba to Avoid Double Inference](#fix-5)
6. [FIX 6 — Increase ML Weight on Rule Disagreement](#fix-6)
7. [FIX 7 — Add Year as a Temporal Feature](#fix-7)
8. [FIX 8 — Explicit High-Value DTC Code Flags](#fix-8)
9. [Bundle Schema Changes Summary](#bundle-schema)
10. [run_ml() Refactor Summary](#run_ml-refactor)
11. [Expected Cumulative Gains](#expected-gains)

---

<a name="fix-1"></a>
## FIX 1 — Cascade Failure Analysis Probabilities into Warranty Classifier

### Why
Both classifiers currently train on the **same `X` matrix** (lines 212–215). The warranty decision is structurally downstream of failure analysis — e.g. "Track burnt due to EOS" maps almost deterministically to "Customer Failure". The FA classifier is already at 98.45%, yet the warranty classifier never sees its output. This is the single largest source of recoverable accuracy.

### Expected Gain
**+3–5% warranty accuracy**

### Changes Required

#### In `train_and_save()` — Lines 208–226

**Current code (lines 208–215):**
```python
X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
    X, y_fa, y_wd, test_size=0.2, random_state=42)

logger.info("[INIT] Training ML models...")
clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_fa.fit(X_tr, yfa_tr)
clf_wd = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_wd.fit(X_tr, ywd_tr)
```

**Replace with:**
```python
from scipy.sparse import csr_matrix

X_tr, X_te, yfa_tr, yfa_te, ywd_tr, ywd_te = train_test_split(
    X, y_fa, y_wd, test_size=0.2, random_state=42)

logger.info("[INIT] Training Failure Analysis classifier...")
clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_fa.fit(X_tr, yfa_tr)

# Cascade: append FA probabilities as features for Warranty classifier
fa_probs_tr = clf_fa.predict_proba(X_tr)          # shape (n_train, 6)
fa_probs_te = clf_fa.predict_proba(X_te)          # shape (n_test,  6)
X_wd_tr = hstack([X_tr, csr_matrix(fa_probs_tr)])
X_wd_te = hstack([X_te, csr_matrix(fa_probs_te)])

logger.info("[INIT] Training Warranty Decision classifier with FA cascade...")
clf_wd = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",    # also covers FIX 3
    random_state=42,
    n_jobs=-1
)
clf_wd.fit(X_wd_tr, ywd_tr)
```

**Update accuracy evaluation (lines 217–220) — test set must use `X_wd_te`:**
```python
fa_acc = accuracy_score(yfa_te, clf_fa.predict(X_te))
wd_acc = accuracy_score(ywd_te, clf_wd.predict(X_wd_te))   # ← was X_te
logger.info("Failure Analysis accuracy: %.3f", fa_acc)
logger.info("Warranty Decision accuracy: %.3f", wd_acc)
```

**Update bundle save (lines 222–226) — store `clf_fa` reference so `run_ml` can cascade:**
```python
bundle = dict(
    clf_fa=clf_fa,
    clf_wd=clf_wd,
    le_fa=le_fa,
    le_wd=le_wd,
    ohe=ohe,
    tfidf_d=tfidf_d,
    scaler=scaler,
    # NEW: ohe_supplier and mileage_scaler added in FIX 2
)
```

#### In `run_ml()` — Lines 273–328

The cascade must be replicated at inference time. After building `X`, compute FA probabilities and append them before passing to `clf_wd`.

**Current code (lines 311–320):**
```python
X = hstack([X_c, X_d, _csr(X_n), _csr(X_v)])

fa_idx = _bundle["clf_fa"].predict(X)[0]
wd_idx = _bundle["clf_wd"].predict(X)[0]
fa_prob = float(np.max(_bundle["clf_fa"].predict_proba(X)[0]))
wd_prob = float(np.max(_bundle["clf_wd"].predict_proba(X)[0]))
```

**Replace with:**
```python
X = hstack([X_c, X_d, _csr(X_n), _csr(X_v)])   # base features (unchanged)

# FA inference
fa_proba_row = _bundle["clf_fa"].predict_proba(X)[0]   # shape (6,)
fa_idx  = int(np.argmax(fa_proba_row))
fa_prob = float(fa_proba_row[fa_idx])

# Cascade FA probs into warranty input
X_wd = hstack([X, _csr(fa_proba_row.reshape(1, -1))])

# WD inference using cascaded features
wd_proba_row = _bundle["clf_wd"].predict_proba(X_wd)[0]
wd_idx  = int(np.argmax(wd_proba_row))
wd_prob = float(wd_proba_row[wd_idx])
```

> **Important:** The `X_wd` matrix shape at inference must exactly match training.  
> If FIX 2 is also applied, the order must be:  
> `hstack([X_c, X_d, X_n, X_v, X_s, X_m, fa_proba_row])` — FA probs always appended last.

---

<a name="fix-2"></a>
## FIX 2 — Add Supplier and Mileage_km as Features

### Why
The dataset has two completely unused columns (confirmed from CSV schema):
- **`Supplier`** (categorical, ~5–10 values): "Production Failure" claims are tightly linked to specific supplier batches — this is among the highest-signal features for distinguishing production vs customer faults.
- **`Mileage_km`** (numeric, range ~0–150k): Low mileage + failure = production defect. High mileage + failure = wear/customer. This is a textbook warranty signal.

### Expected Gain
**+2–4% warranty accuracy**

### Changes Required

#### In `train_and_save()` — after line 203 (where `X_v` is built)

**Add after line 203:**
```python
# FIX 2: Supplier (categorical OHE) + Mileage (scaled numeric)
ohe_supplier  = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
mileage_scaler = StandardScaler()

X_s = ohe_supplier.fit_transform(df[["Supplier"]])
X_m = mileage_scaler.fit_transform(df[["Mileage_km"]])

from scipy.sparse import csr_matrix
X = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v),
            X_s, csr_matrix(X_m)])                          # ← was without X_s, X_m
```

**Update the existing `X = hstack(...)` line 206 — remove it (replaced above).**

**Update bundle save to include new transformers:**
```python
bundle = dict(
    clf_fa=clf_fa,
    clf_wd=clf_wd,
    le_fa=le_fa,
    le_wd=le_wd,
    ohe=ohe,
    tfidf_d=tfidf_d,
    scaler=scaler,
    ohe_supplier=ohe_supplier,    # NEW
    mileage_scaler=mileage_scaler # NEW
)
```

#### In `run_ml()` — feature assembly block (lines 294–311)

The `features` dict passed to `run_ml()` must carry `supplier` and `mileage_km`. Update the dict building in `predict()` (lines 518–529) and in `run_ml()`.

**In `predict()`, update the fallback features block (lines 518–529):**
```python
if features is None:
    dtc_f = extract_dtc_features(fc)
    features = {
        "customer_complaint": match_complaint(notes),
        "dtc_text":           dtc_f["dtc_text"],
        "dtc_count":          dtc_f["dtc_count"],
        "voltage":            v if v is not None else 12.5,
        "has_P":              dtc_f["has_P"],
        "has_U":              dtc_f["has_U"],
        "has_C":              dtc_f["has_C"],
        "has_B":              dtc_f["has_B"],
        "supplier":           "Unknown",    # NEW — passed from caller if available
        "mileage_km":         50000.0,      # NEW — neutral midpoint default
    }
```

> **Note:** Update all callers of `predict()` to pass `supplier` and `mileage_km`  
> when available. The LLM feature translator (`translate_to_ml_features`) should  
> also be updated to extract/pass these if they appear in the claim context.

**In `run_ml()`, add after `X_v` is built (around line 310):**
```python
# FIX 2: Supplier + Mileage
supplier_val  = features.get("supplier", "Unknown")
mileage_val   = features.get("mileage_km", 50000.0)

X_s = _bundle["ohe_supplier"].transform(
    pd.DataFrame([[supplier_val]], columns=["Supplier"])
)
X_m = _csr(_bundle["mileage_scaler"].transform(
    pd.DataFrame([[mileage_val]], columns=["Mileage_km"])
))

X = hstack([X_c, X_d, _csr(X_n), _csr(X_v), X_s, X_m])   # replaces line 311
```

---

<a name="fix-3"></a>
## FIX 3 — Enable `class_weight="balanced"` on Warranty Classifier

### Why
From the CSV: Customer Failure = 20,451 rows, According to Specification = 15,000, Production Failure = 14,549. This ~1.4x imbalance causes the classifier to under-serve the minority classes. The confusion matrix confirms this — Production Failure F1 is only 0.83 (worst of all three warranty classes).

### Expected Gain
**+0.5–1.5% warranty accuracy, larger improvement on Production Failure recall**

### Changes Required

#### In `train_and_save()` — line 214

**Current:**
```python
clf_wd = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
```

**Replace with:**
```python
clf_wd = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

> **Note:** This is already incorporated into the FIX 1 replacement block above.  
> Only apply separately if implementing FIX 3 without FIX 1.

---

<a name="fix-4"></a>
## FIX 4 — Remove the Artificial 50% Confidence Floor in `run_ml()`

### Why
Line 320 clamps ML confidence to a minimum of 50%:
```python
ml_confidence = round(min(98.0, max(50.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)
```
When both classifiers are uncertain (e.g. fa_prob=0.4, wd_prob=0.35), the true confidence is ~37% but gets reported as 50%. This corrupts `combine_scores()` by making genuinely uncertain predictions look more reliable than they are, preventing correct routing to `"Needs Manual Review"`.

### Expected Gain
**Accuracy-neutral; improves status routing correctness and trust calibration**

### Changes Required

#### In `run_ml()` — Line 320

**Current:**
```python
ml_confidence = round(min(98.0, max(50.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)
```

**Replace with:**
```python
ml_confidence = round(min(98.0, max(0.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)
```

> Optionally use `max(30.0, ...)` instead of `0.0` if downstream consumers  
> of `ml_confidence` expect a non-zero value (e.g. for display purposes).  
> Check `combine_scores()` thresholds `CONFIDENCE_THRESHOLD_FIRM = 85.0` and  
> `CONFIDENCE_THRESHOLD_MANUAL = 65.0` — these are unaffected by this change.

---

<a name="fix-5"></a>
## FIX 5 — Reuse `predict_proba` to Avoid Double Inference

### Why
Lines 313–316 call `.predict()` and `.predict_proba()` separately on both classifiers — this runs the full forest traversal twice per classifier per request (4 inference passes total). `.predict()` output is simply `argmax(predict_proba())`, so the separate call is redundant and wasteful.

### Expected Gain
**~2× inference speedup at prediction time (no accuracy impact)**

### Changes Required

#### In `run_ml()` — Lines 313–316

**Current:**
```python
fa_idx = _bundle["clf_fa"].predict(X)[0]
wd_idx = _bundle["clf_wd"].predict(X)[0]
fa_prob = float(np.max(_bundle["clf_fa"].predict_proba(X)[0]))
wd_prob = float(np.max(_bundle["clf_wd"].predict_proba(X)[0]))
```

**Replace with (also incorporates FIX 1 cascade):**
```python
# FA — single proba call
fa_proba_row = _bundle["clf_fa"].predict_proba(X)[0]
fa_idx       = int(np.argmax(fa_proba_row))
fa_prob      = float(fa_proba_row[fa_idx])

# Cascade FA probs into WD features (FIX 1)
X_wd = hstack([X, _csr(fa_proba_row.reshape(1, -1))])

# WD — single proba call
wd_proba_row = _bundle["clf_wd"].predict_proba(X_wd)[0]
wd_idx       = int(np.argmax(wd_proba_row))
wd_prob      = float(wd_proba_row[wd_idx])
```

---

<a name="fix-6"></a>
## FIX 6 — Increase ML Weight on Rule Disagreement

### Why
When a rule fires but disagrees with the ML model, the current weights are:
```python
RULE_WEIGHT_DISAGREE = 0.6
ML_WEIGHT_DISAGREE   = 0.1   # Lines 338–340
```
The ML signal is nearly silenced (weights sum to 0.7, not 1.0). When the ML model is high-confidence and the disagreeing rule is a weak heuristic (e.g. `b_code` at 76% confidence, `c_code` at 78%), the correct decision is often the ML output. The current weighting discards that signal.

### Expected Gain
**Reduces mis-routed cases in rule-vs-ML disagreement scenarios; improves `"Needs Manual Review"` precision**

### Changes Required

#### Module-level constants — Lines 337–340

**Current:**
```python
RULE_WEIGHT_AGREE    = 0.7
ML_WEIGHT_AGREE      = 0.3
RULE_WEIGHT_DISAGREE = 0.6
ML_WEIGHT_DISAGREE   = 0.1
```

**Replace with:**
```python
RULE_WEIGHT_AGREE    = 0.7
ML_WEIGHT_AGREE      = 0.3
RULE_WEIGHT_DISAGREE = 0.55
ML_WEIGHT_DISAGREE   = 0.35    # ← was 0.1; now ML gets meaningful weight on disagreement
```

> The disagreement weights still don't need to sum to 1.0 — the gap intentionally  
> deflates combined confidence when there is conflict (acting as a disagreement  
> penalty). Keeping the sum at 0.9 preserves that behaviour.

#### No changes needed to `combine_scores()` logic itself — the constants flow through correctly.

---

<a name="fix-7"></a>
## FIX 7 — Add `Year` as a Temporal Feature

### Why
The dataset spans 2019–2024. Production defects often cluster by model year (batch failures, supplier qualification periods). Year is a free integer signal already in the CSV and adds minimal complexity.

### Expected Gain
**+0.5–1% warranty accuracy for production failure recall**

### Changes Required

#### In `train_and_save()` — after `X_v` is built (around line 203)

**Add:**
```python
# FIX 7: Year as scaled temporal feature
year_scaler = StandardScaler()
X_y = csr_matrix(year_scaler.fit_transform(df[["Year"]]))
```

**Update `X = hstack(...)` to include `X_y`:**
```python
X = hstack([X_c, X_d, csr_matrix(X_n), csr_matrix(X_v),
            X_s, csr_matrix(X_m), X_y])
```

**Update bundle save:**
```python
bundle = dict(
    ...,
    year_scaler=year_scaler    # NEW
)
```

#### In `run_ml()` — feature assembly

**Add to `features` fallback in `predict()` (lines 518–529):**
```python
"year": 2024,    # NEW — pass actual claim year if available from intake form
```

**Add to `run_ml()` after `X_m` is built:**
```python
# FIX 7: Year
year_val = features.get("year", 2024)
X_y = _csr(_bundle["year_scaler"].transform(
    pd.DataFrame([[year_val]], columns=["Year"])
))

X = hstack([X_c, X_d, _csr(X_n), _csr(X_v), X_s, X_m, X_y])
```

---

<a name="fix-8"></a>
## FIX 8 — Explicit Binary Flags for High-Value DTC Codes

### Why
Feature importance analysis (evaluation output) shows individual DTC codes already appearing in the top features: `p0300` (0.61%), `p0615` (0.56%), `p0481` (0.47%), `p1682` (0.47%), `p0301` (0.44%), `p0480` (0.43%). These are currently encoded via TF-IDF on `dtc_text`, which is a noisy representation for structured codes. Explicit binary flags are more stable and interpretable.

### Expected Gain
**+0.5–1% for specific DTC-linked failure classes; improves model explainability**

### Changes Required

#### In `extract_dtc_features()` — Lines 141–153

**Current return dict:**
```python
return {
    "dtc_count": len(codes),
    "has_P": int(any(c.startswith("P") for c in codes)),
    "has_U": int(any(c.startswith("U") for c in codes)),
    "has_C": int(any(c.startswith("C") for c in codes)),
    "has_B": int(any(c.startswith("B") for c in codes)),
    "dtc_text": " ".join(codes),
}
```

**Replace with:**
```python
# Top DTC codes from feature importance analysis (evaluation_output_v3)
HIGH_VALUE_DTCS = [
    "P0300", "P0615", "P0481", "P1682", "P0301",
    "P0480", "P0073", "P0304", "P0482"
]

return {
    "dtc_count": len(codes),
    "has_P":     int(any(c.startswith("P") for c in codes)),
    "has_U":     int(any(c.startswith("U") for c in codes)),
    "has_C":     int(any(c.startswith("C") for c in codes)),
    "has_B":     int(any(c.startswith("B") for c in codes)),
    "dtc_text":  " ".join(codes),
    # FIX 8: explicit high-value DTC flags
    **{f"dtc_{d.lower()}": int(d in codes) for d in HIGH_VALUE_DTCS},
}
```

> Move `HIGH_VALUE_DTCS` to module level (near the `KNOWN_COMPLAINTS` list at line 44)  
> so it is accessible outside `extract_dtc_features()` if needed.

#### In `train_and_save()` — update `X_n` construction (line 202)

**Current:**
```python
X_n = dtc_feats[["dtc_count","has_P","has_U","has_C","has_B"]].values
```

**Replace with:**
```python
dtc_flag_cols = (
    ["dtc_count","has_P","has_U","has_C","has_B"] +
    [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
)
X_n = dtc_feats[dtc_flag_cols].values
```

#### In `run_ml()` — update `X_n` construction (line 307)

**Current:**
```python
X_n = df_row[["dtc_count", "has_P", "has_U", "has_C", "has_B"]].values
```

**Replace with:**
```python
dtc_flag_cols = (
    ["dtc_count","has_P","has_U","has_C","has_B"] +
    [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]
)
X_n = df_row[dtc_flag_cols].values
```

> Ensure `df_row` is constructed with the same new DTC flag columns.  
> Update the `pd.DataFrame([{...}])` block at lines 294–302 to include these keys  
> using `features.get(f"dtc_{d.lower()}", 0)` for each DTC in `HIGH_VALUE_DTCS`.

---

<a name="bundle-schema"></a>
## Bundle Schema Changes Summary

After all fixes, the `bundle` dict saved to `trace_models.pkl` should be:

```python
bundle = dict(
    # Existing
    clf_fa         = clf_fa,
    clf_wd         = clf_wd,          # now trained on cascaded X_wd
    le_fa          = le_fa,
    le_wd          = le_wd,
    ohe            = ohe,             # Customer Complaint OHE (unchanged)
    tfidf_d        = tfidf_d,         # DTC TF-IDF (unchanged)
    scaler         = scaler,          # Voltage scaler (unchanged)

    # NEW — FIX 2
    ohe_supplier   = ohe_supplier,
    mileage_scaler = mileage_scaler,

    # NEW — FIX 7
    year_scaler    = year_scaler,
)
```

> **⚠️ Action required:** Delete `trace_models.pkl` before running `train_and_save()`  
> after these changes. The old pickle will have an incompatible bundle schema and  
> `load_models()` will silently use the stale model if the file exists.

---

<a name="run_ml-refactor"></a>
## `run_ml()` Full Refactored Method

Below is the complete replacement for `run_ml()` (currently lines 273–328) incorporating all fixes:

```python
def run_ml(features: dict) -> dict:
    """
    Run ML scoring on extracted features.
    FIX 1: Cascades FA probabilities into WD classifier input.
    FIX 2: Includes Supplier and Mileage_km features.
    FIX 4: Removes artificial 50% confidence floor.
    FIX 5: Single predict_proba call per classifier (no double inference).
    FIX 7: Includes Year as temporal feature.
    FIX 8: Includes explicit high-value DTC binary flags.
    """
    global _bundle
    if _bundle is None:
        _bundle = load_models()

    from scipy.sparse import csr_matrix as _csr

    # ── Base features ─────────────────────────────────────────────────────────
    dtc_flag_cols = (
        ["dtc_count","has_P","has_U","has_C","has_B"] +
        [f"dtc_{d.lower()}" for d in HIGH_VALUE_DTCS]    # FIX 8
    )
    df_row = pd.DataFrame([{
        "Customer Complaint": features.get("customer_complaint", "OBD Light ON"),
        "dtc_text":           features.get("dtc_text", ""),
        **{col: features.get(col, 0) for col in dtc_flag_cols},
    }])

    X_c = _bundle["ohe"].transform(df_row[["Customer Complaint"]])
    X_d = _bundle["tfidf_d"].transform(df_row["dtc_text"])
    X_n = _csr(df_row[dtc_flag_cols].values)

    X_v = _csr(_bundle["scaler"].transform(
        pd.DataFrame([[features.get("voltage", 12.5)]], columns=["Voltage"])
    ))

    # FIX 2: Supplier + Mileage
    X_s = _bundle["ohe_supplier"].transform(
        pd.DataFrame([[features.get("supplier", "Unknown")]], columns=["Supplier"])
    )
    X_m = _csr(_bundle["mileage_scaler"].transform(
        pd.DataFrame([[features.get("mileage_km", 50000.0)]], columns=["Mileage_km"])
    ))

    # FIX 7: Year
    X_y = _csr(_bundle["year_scaler"].transform(
        pd.DataFrame([[features.get("year", 2024)]], columns=["Year"])
    ))

    # Assemble base feature matrix
    X = hstack([X_c, X_d, X_n, X_v, X_s, X_m, X_y])

    # FIX 5 + FIX 1: single proba call for FA, cascade into WD
    fa_proba_row = _bundle["clf_fa"].predict_proba(X)[0]   # shape (n_fa_classes,)
    fa_idx       = int(np.argmax(fa_proba_row))
    fa_prob      = float(fa_proba_row[fa_idx])

    X_wd = hstack([X, _csr(fa_proba_row.reshape(1, -1))])  # FIX 1 cascade

    wd_proba_row = _bundle["clf_wd"].predict_proba(X_wd)[0]
    wd_idx       = int(np.argmax(wd_proba_row))
    wd_prob      = float(wd_proba_row[wd_idx])

    ml_failure_analysis  = _bundle["le_fa"].inverse_transform([fa_idx])[0]
    ml_warranty_decision = _bundle["le_wd"].inverse_transform([wd_idx])[0]

    # FIX 4: remove artificial 50% floor
    ml_confidence = round(min(98.0, max(0.0, (fa_prob * wd_prob) ** 0.5 * 100)), 1)

    return {
        "ml_warranty_decision": ml_warranty_decision,
        "ml_failure_analysis":  ml_failure_analysis,
        "fa_prob":              fa_prob,
        "wd_prob":              wd_prob,
        "ml_confidence":        ml_confidence,
    }
```

---

<a name="expected-gains"></a>
## Expected Cumulative Accuracy Gains

| Fix | Change | Est. Warranty Accuracy Gain |
|-----|--------|-----------------------------|
| FIX 1 — FA cascade into WD | Structural model change | **+3–5%** |
| FIX 2 — Supplier + Mileage features | New training columns | **+2–4%** |
| FIX 3 — `class_weight="balanced"` | RF hyperparameter | **+0.5–1.5%** |
| FIX 4 — Remove 50% confidence floor | Confidence calibration | Routing improvement |
| FIX 5 — Single `predict_proba` call | Code quality / speed | ~2× inference speedup |
| FIX 6 — ML weight on disagreement | Score blending constants | Status routing quality |
| FIX 7 — Year temporal feature | New training column | **+0.5–1%** |
| FIX 8 — Explicit DTC flags | Feature engineering | **+0.5–1%** |
| **TOTAL ESTIMATED** | | **~88% → 94–96%** |

---

## Implementation Order (Recommended)

Apply fixes in this order to allow incremental validation:

```
1. FIX 5  — Safe refactor, zero risk, validates proba logic before cascade
2. FIX 3  — One-liner, immediate class balance improvement
3. FIX 4  — One-liner, fixes confidence calibration
4. FIX 6  — Four constant changes, improves blending immediately
5. FIX 8  — DTC flags in extract_dtc_features() before touching training
6. FIX 2  — Add Supplier + Mileage to training pipeline
7. FIX 7  — Add Year to training pipeline
8. FIX 1  — Cascade (biggest gain, touches the most code — do last)
```

After each step, delete `trace_models.pkl` and re-run:
```bash
python ml_predictor.py
```
Then re-run your evaluation script to measure incremental gains.
