# Warranty Decision Accuracy Improvement Plan
### Target: `ml_predictor.py` — Raise WD accuracy from 0.8939 → ≥ 0.91

---

## Table of Contents
1. [Overview of All Changes](#overview)
2. [Change 1 — Import additions (line 76)](#change-1)
3. [Change 2 — New helper: `voltage_band()` (after line 243)](#change-2)
4. [Change 3 — Feature engineering in `train_and_save()` (after line 268)](#change-3)
5. [Change 4 — New transformers + fit on train slice (after line 281)](#change-4)
6. [Change 5 — Assemble `X_tr` with new features (line 292–293)](#change-5)
7. [Change 6 — Transform test slice with new features (line 302–305)](#change-6)
8. [Change 7 — OOF cascade fix for `fa_probs_tr` (line 311)](#change-7)
9. [Change 8 — Save new transformers into bundle (line 325–328)](#change-8)
10. [Change 9 — Reconstruct new features in `run_ml()` (line 403–427)](#change-9)
11. [Change 10 — Defaults in `predict()` fallback block (line 640–654)](#change-10)
12. [Review Checklist](#review-checklist)
13. [Test Steps](#test-steps)

---

## Overview of All Changes <a name="overview"></a>

| # | Location | What changes | Why |
|---|---|---|---|
| 1 | Line 76 | Add `cross_val_predict` import | Required for OOF cascade fix |
| 2 | After line 243 (after `match_complaint`) | Add `voltage_band()` helper function | Needed by both `train_and_save` and `run_ml` |
| 3 | After line 268 (after `train_test_split`) | Engineer `mileage_bracket`, `claim_age`, `voltage_band` columns on `df_tr` / `df_te` | New features for clf_wd |
| 4 | After line 281 (after `year_scaler = StandardScaler()`) | Declare `ohe_mileage` and `claim_age_scaler`; fit on train slice | Two new transformers |
| 5 | Lines 292–293 (`X_tr = hstack(...)`) | Add `X_mb_tr` and `X_ca_tr` to the hstack | Includes new features in train matrix |
| 6 | Lines 304–305 (`X_te = hstack(...)`) | Add `X_mb_te` and `X_ca_te` to the hstack | Includes new features in test matrix |
| 7 | Line 311 (`fa_probs_tr = clf_fa.predict_proba(X_tr)`) | Replace with `cross_val_predict(…, method="predict_proba")` | Fixes cascade calibration |
| 8 | Lines 325–328 (bundle dict) | Add `ohe_mileage`, `claim_age_scaler` | New transformers must be persisted |
| 9 | Lines 403–427 (`run_ml` feature build) | Build `X_mb` and `X_ca` from features dict; add to `X = hstack(...)` | Runtime must mirror training feature matrix exactly |
| 10 | Lines 640–654 (`predict()` fallback block) | Add `mileage_bracket` and `claim_age` keys to fallback features dict | Provides defaults when LLM is absent |

---

## Change 1 — Import additions (line 76) <a name="change-1"></a>

### Current code (line 76)
```python
from   sklearn.model_selection import train_test_split
```

### New code
```python
from   sklearn.model_selection import train_test_split, cross_val_predict
```

### Why
`cross_val_predict` is the sklearn utility that generates out-of-fold predictions.
It is used in Change 7 to produce unbiased FA probabilities for training `clf_wd`.
Adding it here keeps all sklearn imports co-located at the top of the file.

### Placement rule
This line sits **within the existing import block** (lines 69–79). It is a pure additive
change to an existing `from sklearn.model_selection import …` line.

---

## Change 2 — New helper function `voltage_band()` (after line 243) <a name="change-2"></a>

### Where exactly
After the closing `return` of `match_complaint()` (line 243), before `def train_and_save():` (line 246).
Insert a blank line after line 243, then the function, then two blank lines before `def train_and_save`.

### Code to insert
```python
def voltage_band(v: float) -> str:
    """
    Bucket raw voltage into domain-meaningful bands.
    Mirrors the voltage thresholds already encoded in the rule engine
    (over_voltage rule: v > 16.0, low_voltage rule: v < 11.0) so that
    clf_wd sees the same non-linear boundary the rules exploit.
    """
    if v < 11.0:
        return "under_voltage"
    if v > 16.0:
        return "over_voltage"
    if v < 12.0:
        return "low_normal"
    if v > 14.5:
        return "high_normal"
    return "nominal"
```

### Why a standalone function (not inline lambda)
- `train_and_save()` applies it via `df_tr["voltage_band"] = df_tr["Voltage"].apply(voltage_band)`.
- `run_ml()` applies it to a single scalar: `voltage_band(features.get("voltage", 12.5))`.
- A named function is reusable in both call sites without duplicating the logic.

### Logic correctness check
| Voltage range | Band returned | Rule engine mapping |
|---|---|---|
| v < 11.0 | `under_voltage` | `low_voltage` rule → Production Failure |
| v > 16.0 | `over_voltage` | `over_voltage` rule → Customer Failure / EOS |
| 11.0–12.0 | `low_normal` | No direct rule, but low operating voltage |
| 14.5–16.0 | `high_normal` | No direct rule, charging system stress |
| 12.0–14.5 | `nominal` | Normal operating range |

The boundaries deliberately match the rule thresholds (11.0 and 16.0) so the ML
classifier can learn the same cutoffs the rule engine uses deterministically.

---

## Change 3 — Feature engineering in `train_and_save()` (after line 268) <a name="change-3"></a>

### Where exactly
After the `train_test_split(...)` call that ends on line 268 and before the
`dtc_flag_cols` definition on line 270. Insert the block below at approximately
line 269 (after the split, before flag cols).

**Important:** This block must come AFTER `train_test_split` because it writes
new columns onto `df_tr` and `df_te`, which do not exist until the split is done.
Writing onto the unsplit `df` before splitting would cause index misalignment
between `df_tr`/`df_te` and the `dtc_feats` split halves.

### Code to insert
```python
    # ── Feature Engineering: new warranty-signal columns ─────────────────────
    # Applied AFTER split so df_tr and df_te are independent.
    # 1. mileage_bracket — warranty decisions are threshold-sensitive to mileage;
    #    OHE captures the non-linearity better than a raw scaled value.
    _mileage_bins   = [0, 20_000, 60_000, 100_000, np.inf]
    _mileage_labels = ["low", "mid", "high", "very_high"]
    df_tr["mileage_bracket"] = pd.cut(
        df_tr["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)
    df_te["mileage_bracket"] = pd.cut(
        df_te["Mileage_km"], bins=_mileage_bins, labels=_mileage_labels
    ).astype(str)

    # 2. claim_age — years between vehicle manufacture year and claim date.
    #    A direct warranty-eligibility signal entirely absent from the current
    #    feature set. Raw 'Year' column (importance 0.016) is the vehicle year
    #    alone; claim_age combines it with the claim date for real signal.
    df_tr["claim_age"] = pd.to_datetime(df_tr["Date"]).dt.year - df_tr["Year"]
    df_te["claim_age"] = pd.to_datetime(df_te["Date"]).dt.year - df_te["Year"]

    # 3. voltage_band — OHE version of voltage buckets aligned to rule thresholds.
    df_tr["voltage_band"] = df_tr["Voltage"].apply(voltage_band)
    df_te["voltage_band"] = df_te["Voltage"].apply(voltage_band)
```

### Correctness notes
- `pd.cut` on `df_te` uses the **same** `_mileage_bins` / `_mileage_labels`
  defined once above. It does NOT re-compute bins from `df_te` statistics.
  This is correct: bin edges are fixed domain thresholds, not data-derived.
- `pd.cut` may produce `NaN` if a value lands outside all bins (impossible here
  since `np.inf` is the upper edge) or for null `Mileage_km`. `.astype(str)`
  converts any `NaN` to the string `"nan"`, which the OHE will treat as an
  unknown category and encode as all-zeros (because `handle_unknown="ignore"`
  is set on `ohe_mileage` in Change 4). This is safe.
- `claim_age` is computed as integer years. Negative values are theoretically
  impossible (a claim cannot be filed before the vehicle year) but if the data
  contains bad rows, clipping to 0 is advisable:
  `df_tr["claim_age"] = df_tr["claim_age"].clip(lower=0)` (optional guard).
- `voltage_band` uses the helper defined in Change 2. It is applied with `.apply()`
  which passes each scalar `Voltage` value to the function — correct usage.

---

## Change 4 — New transformers, fit on train slice (after line 281) <a name="change-4"></a>

### Where exactly
After `year_scaler = StandardScaler()` on line 281, before `X_c_tr = ohe.fit_transform(...)` on line 283.
These declarations follow the existing pattern: declare transformer → fit on train → transform test.

### Code to insert
```python
    ohe_mileage      = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    ohe_vband        = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    claim_age_scaler = StandardScaler()
```

Then, after line 289 (`X_y_tr = year_scaler.fit_transform(df_tr[["Year"]])`), add:

```python
    X_mb_tr = ohe_mileage.fit_transform(df_tr[["mileage_bracket"]])
    X_vb_tr = ohe_vband.fit_transform(df_tr[["voltage_band"]])
    X_ca_tr = claim_age_scaler.fit_transform(df_tr[["claim_age"]])
```

### Why these transformer types
| Feature | Transformer | Reason |
|---|---|---|
| `mileage_bracket` | `OneHotEncoder` | Categorical strings ("low", "mid", "high", "very_high") → sparse binary columns |
| `voltage_band` | `OneHotEncoder` | Categorical strings ("nominal", "under_voltage", etc.) → sparse binary columns |
| `claim_age` | `StandardScaler` | Numeric (integer years 0–5 typically); scaling keeps it comparable to other numeric inputs |

### `handle_unknown="ignore"` — why it matters
At inference time, `run_ml()` receives raw user inputs. If an unexpected string
lands in `mileage_bracket` or `voltage_band`, the OHE silently emits a zero
vector rather than raising an exception. This is the same strategy used by the
existing `ohe` (complaint) and `ohe_supplier` transformers in the file.

---

## Change 5 — Assemble `X_tr` with new features (lines 292–293) <a name="change-5"></a>

### Current code (lines 291–293)
```python
    from scipy.sparse import csr_matrix
    X_tr = hstack([X_c_tr, X_d_tr, csr_matrix(X_n_tr), csr_matrix(X_v_tr),
                   X_s_tr, csr_matrix(X_m_tr), csr_matrix(X_y_tr)])
```

### New code
```python
    from scipy.sparse import csr_matrix
    X_tr = hstack([X_c_tr, X_d_tr, csr_matrix(X_n_tr), csr_matrix(X_v_tr),
                   X_s_tr, csr_matrix(X_m_tr), csr_matrix(X_y_tr),
                   X_mb_tr, X_vb_tr, csr_matrix(X_ca_tr)])
```

### Column order rationale
New columns are appended at the **end** of the hstack, after all existing columns.
This means the existing column indices (0 to N-1) are unchanged, preserving
any future debugging that references column positions.

### Type consistency check
`hstack` requires all inputs to be sparse matrices:
- `X_mb_tr` — output of `OneHotEncoder.fit_transform` → already sparse ✓  
- `X_vb_tr` — output of `OneHotEncoder.fit_transform` → already sparse ✓  
- `X_ca_tr` — output of `StandardScaler.fit_transform` → dense numpy array → **must wrap in `csr_matrix()`** ✓ (done above)

This matches how existing dense outputs `X_v_tr`, `X_m_tr`, `X_y_tr` are already wrapped.

---

## Change 6 — Transform test slice with new features (lines 304–305) <a name="change-6"></a>

### Current code (lines 302–305)
```python
    X_m_te = mileage_scaler.transform(df_te[["Mileage_km"]])
    X_y_te = year_scaler.transform(df_te[["Year"]])

    X_te = hstack([X_c_te, X_d_te, csr_matrix(X_n_te), csr_matrix(X_v_te),
                   X_s_te, csr_matrix(X_m_te), csr_matrix(X_y_te)])
```

### New code
```python
    X_m_te = mileage_scaler.transform(df_te[["Mileage_km"]])
    X_y_te = year_scaler.transform(df_te[["Year"]])
    X_mb_te = ohe_mileage.transform(df_te[["mileage_bracket"]])
    X_vb_te = ohe_vband.transform(df_te[["voltage_band"]])
    X_ca_te = claim_age_scaler.transform(df_te[["claim_age"]])

    X_te = hstack([X_c_te, X_d_te, csr_matrix(X_n_te), csr_matrix(X_v_te),
                   X_s_te, csr_matrix(X_m_te), csr_matrix(X_y_te),
                   X_mb_te, X_vb_te, csr_matrix(X_ca_te)])
```

### Critical constraint — column order MUST match Change 5
The hstack column order in `X_te` must be **identical** to `X_tr` from Change 5.
If the order differs, `clf_fa` and `clf_wd` will receive feature columns in the
wrong positions and produce garbage predictions. The order above is:
```
[X_c, X_d, X_n, X_v, X_s, X_m, X_y, X_mb, X_vb, X_ca]
```
This same order must also be reproduced in `run_ml()` (Change 9).

### `transform()` not `fit_transform()` — why
`ohe_mileage`, `ohe_vband`, and `claim_age_scaler` were **fit** on `df_tr` only in Change 4.
Calling `.transform()` on `df_te` applies the learned vocabulary/statistics without
re-fitting — which is the correct leakage-free pattern already used for all
other transformers in this file.

---

## Change 7 — OOF cascade fix for `fa_probs_tr` (line 311) <a name="change-7"></a>

### Current code (lines 311–314)
```python
    fa_probs_tr = clf_fa.predict_proba(X_tr)
    fa_probs_te = clf_fa.predict_proba(X_te)
    X_wd_tr = hstack([X_tr, csr_matrix(fa_probs_tr)])
    X_wd_te = hstack([X_te, csr_matrix(fa_probs_te)])
```

### The problem
`clf_fa.predict_proba(X_tr)` scores the same rows `clf_fa` was trained on.
A Random Forest memorises training rows perfectly (zero training error), so
`fa_probs_tr` is full of near-1.0 confidences — mean 0.9888, std 0.0438
(measured in evaluation output). But at inference time, `clf_fa` sees unseen
data and returns calibrated probabilities — mean 0.9776, std 0.0741.
`clf_wd` is therefore trained on a **different distribution of FA features**
than it will encounter at inference, degrading performance specifically on
ambiguous Customer Failure / Production Failure boundary cases.

### New code
```python
    logger.info("[INIT] Generating OOF FA probabilities for WD cascade (cv=5)...")
    fa_probs_tr = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_tr, yfa_tr,
        cv=5,
        method="predict_proba",
    )
    # Train the final clf_fa on ALL of X_tr (OOF probs used only for clf_wd training)
    clf_fa = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_fa.fit(X_tr, yfa_tr)

    fa_probs_te = clf_fa.predict_proba(X_te)
    X_wd_tr = hstack([X_tr, csr_matrix(fa_probs_tr)])
    X_wd_te = hstack([X_te, csr_matrix(fa_probs_te)])
```

### Step-by-step execution logic
1. `cross_val_predict` splits `X_tr` into 5 folds. For each fold, it trains a
   fresh RF on the other 4 folds and predicts probabilities for the held-out fold.
   The result `fa_probs_tr` has shape `(len(X_tr), n_fa_classes)` — one
   probability row per training sample, but each row was predicted by a model
   that **never saw that row during training**. This mirrors true inference.
2. A **separate** `clf_fa` is then trained on all of `X_tr`. This is the model
   saved into the bundle and used at inference time. The OOF probabilities are
   only used to train `clf_wd` with a realistic input distribution.
3. `fa_probs_te = clf_fa.predict_proba(X_te)` uses the full-data clf_fa, same
   as before. `X_wd_te` and `X_wd_tr` assembly is unchanged.

### Runtime cost
`cross_val_predict` trains 5 RF models (200 estimators each) on ~80% of 40,000
rows. On a modern multi-core CPU with `n_jobs=-1` this takes ~2–4 minutes.
This is a one-time training cost, not an inference cost.

### Why not `cv=3`?
The evaluation report used `cv=3` for speed. For training, `cv=5` is preferred:
more folds = each OOF fold is predicted by a model trained on a larger fraction
of the data = more representative probabilities.

---

## Change 8 — Save new transformers into bundle (lines 325–328) <a name="change-8"></a>

### Current code (lines 325–328)
```python
    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd, le_fa=le_fa, le_wd=le_wd,
                  ohe=ohe, tfidf_d=tfidf_d, scaler=scaler,
                  ohe_supplier=ohe_supplier, mileage_scaler=mileage_scaler,
                  year_scaler=year_scaler)
```

### New code
```python
    bundle = dict(clf_fa=clf_fa, clf_wd=clf_wd, le_fa=le_fa, le_wd=le_wd,
                  ohe=ohe, tfidf_d=tfidf_d, scaler=scaler,
                  ohe_supplier=ohe_supplier, mileage_scaler=mileage_scaler,
                  year_scaler=year_scaler,
                  ohe_mileage=ohe_mileage, ohe_vband=ohe_vband,
                  claim_age_scaler=claim_age_scaler)
```

### Why this matters
`run_ml()` loads `_bundle` from disk via `pickle.load`. If `ohe_mileage`,
`ohe_vband`, and `claim_age_scaler` are not in the bundle, `run_ml()` will raise
a `KeyError` when Change 9 calls `_bundle["ohe_mileage"]`.
The bundle is the single serialisation point — every transformer used in `run_ml`
**must** appear in this dict.

### Impact on existing deployments
Any `.pkl` file generated by the old `train_and_save()` will not contain these
keys. After applying all changes, `train_and_save()` must be re-run to
regenerate `trace_models.pkl`. The old model file should be deleted or renamed
before the new training run to prevent `load_models()` from loading the stale file.

---

## Change 9 — Reconstruct new features in `run_ml()` (lines 403–427) <a name="change-9"></a>

### Where exactly
Inside `run_ml()` (lines 378–450). The existing `X = hstack(...)` on line 427
is the assembly point. New feature extraction and assembly must be inserted
**between** the existing `X_y` block (line 423–425) and the `X = hstack(...)` line.

### Current code (lines 417–427)
```python
    X_s = _bundle["ohe_supplier"].transform(
        pd.DataFrame([[features.get("supplier", "Unknown")]], columns=["Supplier"])
    )
    X_m = _csr(_bundle["mileage_scaler"].transform(
        pd.DataFrame([[features.get("mileage_km", 50000.0)]], columns=["Mileage_km"])
    ))
    X_y = _csr(_bundle["year_scaler"].transform(
        pd.DataFrame([[features.get("year", 2024)]], columns=["Year"])
    ))

    X = hstack([X_c, X_d, _csr(X_n), _csr(X_v), X_s, X_m, X_y])
```

### New code (replace the `X = hstack(...)` line only, keep lines above)
```python
    X_s = _bundle["ohe_supplier"].transform(
        pd.DataFrame([[features.get("supplier", "Unknown")]], columns=["Supplier"])
    )
    X_m = _csr(_bundle["mileage_scaler"].transform(
        pd.DataFrame([[features.get("mileage_km", 50000.0)]], columns=["Mileage_km"])
    ))
    X_y = _csr(_bundle["year_scaler"].transform(
        pd.DataFrame([[features.get("year", 2024)]], columns=["Year"])
    ))

    # New features — must match train_and_save() column order exactly
    _raw_mileage = features.get("mileage_km", 50000.0)
    _mb_val = pd.cut(
        pd.Series([_raw_mileage]),
        bins=[0, 20_000, 60_000, 100_000, np.inf],
        labels=["low", "mid", "high", "very_high"]
    ).astype(str).iloc[0]
    X_mb = _bundle["ohe_mileage"].transform(
        pd.DataFrame([[_mb_val]], columns=["mileage_bracket"])
    )

    _vb_val = voltage_band(features.get("voltage", 12.5))
    X_vb = _bundle["ohe_vband"].transform(
        pd.DataFrame([[_vb_val]], columns=["voltage_band"])
    )

    _ca_val = float(features.get("claim_age", 1))
    X_ca = _csr(_bundle["claim_age_scaler"].transform(
        pd.DataFrame([[_ca_val]], columns=["claim_age"])
    ))

    X = hstack([X_c, X_d, _csr(X_n), _csr(X_v), X_s, X_m, X_y, X_mb, X_vb, X_ca])
```

### Column order verification
```
train_and_save X_tr:  [X_c, X_d, X_n, X_v, X_s, X_m, X_y, X_mb, X_vb, X_ca]
run_ml X:             [X_c, X_d, X_n, X_v, X_s, X_m, X_y, X_mb, X_vb, X_ca]  ← identical ✓
```

### `pd.cut` in inference — why it's safe
`pd.cut` here uses **hardcoded bin edges** (not derived from any data), so it
produces the same bucketing regardless of what data it is called on. The OHE
then maps the string label to the column it learned during training. If
`_mb_val` comes back as `"nan"` (e.g. negative mileage), `handle_unknown="ignore"`
emits a zero row — the model gets no mileage signal for that row and falls back
on other features, which is the correct degraded-gracefully behaviour.

### `claim_age` default at inference
The fallback `features.get("claim_age", 1)` uses 1 year as the default —
representing a typical in-warranty claim. This default is set here and in
Change 10 (predict fallback block). Both must use the same default value.

---

## Change 10 — Defaults in `predict()` fallback block (lines 640–654) <a name="change-10"></a>

### Where exactly
Inside `predict()`, the `if features is None:` block (lines 640–654).
This block runs when neither the LLM path nor the LLM feature translation
succeeded — i.e., it is the pure rule-engine fallback used in production
when the LLM key is absent.

### Current code (lines 640–654)
```python
    if features is None:
        dtc_f = extract_dtc_features(fc)
        features = {
            "customer_complaint": match_complaint(notes),
            "dtc_text": dtc_f["dtc_text"],
            "dtc_count": dtc_f["dtc_count"],
            "voltage": v if v is not None else 12.5,
            "has_P": dtc_f["has_P"],
            "has_U": dtc_f["has_U"],
            "has_C": dtc_f["has_C"],
            "has_B": dtc_f["has_B"],
            "supplier": "Unknown",
            "mileage_km": 50000.0,
            "year": 2024,
        }
```

### New code
```python
    if features is None:
        dtc_f = extract_dtc_features(fc)
        _voltage_val = v if v is not None else 12.5
        features = {
            "customer_complaint": match_complaint(notes),
            "dtc_text": dtc_f["dtc_text"],
            "dtc_count": dtc_f["dtc_count"],
            "voltage": _voltage_val,
            "has_P": dtc_f["has_P"],
            "has_U": dtc_f["has_U"],
            "has_C": dtc_f["has_C"],
            "has_B": dtc_f["has_B"],
            "supplier": "Unknown",
            "mileage_km": 50000.0,
            "year": 2024,
            "claim_age": 1,          # default: assume 1-year-old claim
        }
```

### Why `mileage_bracket` and `voltage_band` are NOT added here
`run_ml()` (Change 9) derives `mileage_bracket` from `features["mileage_km"]`
and `voltage_band` from `features["voltage"]` internally. Those two derived
values are **not** passed through the features dict — they are computed fresh
inside `run_ml` from the raw values. Only `claim_age` must be in the features
dict because it requires knowledge of the claim date that is not otherwise
available in the `predict()` signature.

### Why `claim_age = 1`
The `predict()` function signature is `predict(fault_code, technician_notes, voltage)`.
It has no `date` or `claim_age` parameter. A default of 1 year represents
a standard early-warranty claim. If the calling application can supply a
claim age, the LLM feature translation path (`translate_to_ml_features`) should
be updated to include it in the returned features dict.

---

## Review Checklist <a name="review-checklist"></a>

Work through these checks **in order** after making all changes. Each check
references the specific code location it verifies.

### Syntax checks

- [ ] **R1** — `cross_val_predict` is in the import on line 76 alongside
  `train_test_split`. No comma missing. Run `python -c "from sklearn.model_selection import train_test_split, cross_val_predict"`.

- [ ] **R2** — `voltage_band()` is defined **before** `train_and_save()` in the
  file. Python reads top-to-bottom; if `train_and_save` calls `voltage_band`
  but it is defined after, a `NameError` will occur at runtime. Verify by
  checking line numbers: `voltage_band` definition line < `def train_and_save` line.

- [ ] **R3** — All three new feature transform lines in `train_and_save()` use
  `.fit_transform()` (not `.transform()`):
  ```python
  X_mb_tr = ohe_mileage.fit_transform(df_tr[["mileage_bracket"]])
  X_vb_tr = ohe_vband.fit_transform(df_tr[["voltage_band"]])
  X_ca_tr = claim_age_scaler.fit_transform(df_tr[["claim_age"]])
  ```

- [ ] **R4** — All three new feature transform lines for the test slice use
  `.transform()` (not `.fit_transform()`):
  ```python
  X_mb_te = ohe_mileage.transform(df_te[["mileage_bracket"]])
  X_vb_te = ohe_vband.transform(df_te[["voltage_band"]])
  X_ca_te = claim_age_scaler.transform(df_te[["claim_age"]])
  ```

- [ ] **R5** — The `X_tr` hstack in `train_and_save()` and the `X` hstack in
  `run_ml()` end with the same three new components in the same order:
  `..., X_mb, X_vb, X_ca` / `..., X_mb_tr, X_vb_tr, csr_matrix(X_ca_tr)`.

- [ ] **R6** — `csr_matrix(X_ca_tr)` and `_csr(X_ca)` are wrapped because
  `StandardScaler` outputs dense numpy arrays. Confirm that `X_mb_tr`, `X_vb_tr`
  (OHE outputs) are **not** wrapped — they are already sparse.

- [ ] **R7** — The bundle dict contains all three new keys: `ohe_mileage`,
  `ohe_vband`, `claim_age_scaler`. Count the keys — old: 10 keys, new: 13 keys.

- [ ] **R8** — In `run_ml()`, `_bundle["ohe_mileage"]`, `_bundle["ohe_vband"]`,
  and `_bundle["claim_age_scaler"]` are referenced. These key names must exactly
  match what was saved in the bundle dict (Change 8). No typos.

### Logic checks

- [ ] **R9** — `cross_val_predict` call uses `method="predict_proba"`. Without
  this argument it defaults to `method="predict"` and returns class labels
  instead of probabilities. Verify the keyword argument is present.

- [ ] **R10** — After `cross_val_predict`, `clf_fa` is re-instantiated and
  `.fit(X_tr, yfa_tr)` is called on the full training set. Confirm `clf_fa`
  is defined **after** the `cross_val_predict` block (not before), so the
  final model trained on full data is what gets pickled.

- [ ] **R11** — The `pd.cut` bin edges and labels in `run_ml()` (Change 9)
  exactly match those in `train_and_save()` (Change 3):
  - Bins: `[0, 20_000, 60_000, 100_000, np.inf]`
  - Labels: `["low", "mid", "high", "very_high"]`
  Any mismatch produces different string labels, causing the OHE to emit
  all-zeros (unknown category) for every real sample.

- [ ] **R12** — `voltage_band()` thresholds in Change 2 match the rule engine
  thresholds in `RULES` (lines 110 and 119): `> 16.0` for over-voltage,
  `< 11.0` for under-voltage. Confirm no off-by-one (the rules use `> 16.0`
  and `< 11.0`; the function must use the same strict inequalities).

- [ ] **R13** — `claim_age` default is `1` in both the `predict()` fallback
  block (Change 10) and the `features.get("claim_age", 1)` call in `run_ml()`
  (Change 9). Both must be the same value.

### Sequential execution order check

The final execution order inside `train_and_save()` must be:

```
1. pd.read_csv + fillna                           (lines 248–253)
2. LabelEncoder fit on full df                    (lines 257–258)
3. extract_dtc_features on full df                (line 260)
4. train_test_split                               (lines 266–268)
5. Feature engineering (mileage_bracket,          ← Change 3 — AFTER split
   claim_age, voltage_band on df_tr/df_te)
6. dtc_flag_cols definition                       (lines 270–273)
7. Declare transformers                           (lines 276–281 + Change 4 additions)
8. fit_transform on df_tr slice                   (lines 283–289 + Change 4 fit lines)
9. Assemble X_tr                                  (lines 291–293, Change 5)
10. transform on df_te slice                      (lines 296–302 + Change 6 transform lines)
11. Assemble X_te                                 (lines 304–305, Change 6)
12. Train clf_fa (initial, for OOF)               (← implicitly inside cross_val_predict)
13. cross_val_predict → fa_probs_tr               (Change 7)
14. Train final clf_fa on full X_tr               (Change 7)
15. clf_fa.predict_proba(X_te) → fa_probs_te      (line 312)
16. Assemble X_wd_tr, X_wd_te                     (lines 313–314)
17. Train clf_wd                                   (lines 316–318)
18. Evaluate + log accuracy                       (lines 320–323)
19. Save bundle with all transformers             (lines 325–330, Change 8)
```

Verify that no step uses a variable defined in a later step.

---

## Test Steps <a name="test-steps"></a>

### T1 — Dry-run import test (no training required)
```bash
python -c "
import ml_predictor
print('Import OK')
print('voltage_band(10.5):', ml_predictor.voltage_band(10.5))   # expect: under_voltage
print('voltage_band(12.5):', ml_predictor.voltage_band(12.5))   # expect: nominal
print('voltage_band(17.0):', ml_predictor.voltage_band(17.0))   # expect: over_voltage
print('voltage_band(11.5):', ml_predictor.voltage_band(11.5))   # expect: low_normal
print('voltage_band(15.0):', ml_predictor.voltage_band(15.0))   # expect: high_normal
"
```
**Pass condition:** No `ImportError`, all 5 band labels print as expected.

---

### T2 — Feature engineering shape test (no training required)
```bash
python -c "
import pandas as pd, numpy as np
from ml_predictor import voltage_band

df = pd.DataFrame({'Mileage_km': [10000, 40000, 80000, 150000],
                   'Voltage':    [10.5,  12.5,  14.8,  17.5],
                   'Year':       [2020,  2021,  2022,  2023],
                   'Date':       ['2021-01-01','2023-06-01','2024-03-01','2024-12-01']})

bins   = [0, 20_000, 60_000, 100_000, np.inf]
labels = ['low', 'mid', 'high', 'very_high']
df['mileage_bracket'] = pd.cut(df['Mileage_km'], bins=bins, labels=labels).astype(str)
df['claim_age']       = pd.to_datetime(df['Date']).dt.year - df['Year']
df['voltage_band']    = df['Voltage'].apply(voltage_band)

print(df[['mileage_bracket','claim_age','voltage_band']])
# Expected:
#   mileage_bracket  claim_age  voltage_band
#   low              1          under_voltage
#   mid              2          nominal
#   high             2          high_normal
#   very_high        1          over_voltage
"
```
**Pass condition:** All three derived columns match expected values above.

---

### T3 — Training run with shape assertion
After making all changes, run training and assert that the bundle contains the
new keys and that model input shapes are as expected:

```bash
python -c "
import pickle, os
os.remove('trace_models.pkl') if os.path.exists('trace_models.pkl') else None
from ml_predictor import train_and_save
bundle = train_and_save()

required_keys = [
    'clf_fa', 'clf_wd', 'le_fa', 'le_wd',
    'ohe', 'tfidf_d', 'scaler', 'ohe_supplier',
    'mileage_scaler', 'year_scaler',
    'ohe_mileage', 'ohe_vband', 'claim_age_scaler'   # new
]
missing = [k for k in required_keys if k not in bundle]
assert not missing, f'Missing bundle keys: {missing}'
print('Bundle keys OK:', list(bundle.keys()))

# Verify OHE categories learned
print('mileage OHE categories:', bundle['ohe_mileage'].categories_)
print('vband   OHE categories:', bundle['ohe_vband'].categories_)
"
```
**Pass condition:**
- No assertion error — all 13 keys present.
- `ohe_mileage.categories_` shows `[['high', 'low', 'mid', 'very_high']]` (alphabetical).
- `ohe_vband.categories_` shows all 5 band strings.
- Training completes without exception.

---

### T4 — Accuracy regression test
```bash
python -c "
import pickle, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from ml_predictor import train_and_save

bundle = train_and_save()
# Accuracy is logged during training — check the log output for lines like:
# Warranty Decision accuracy: 0.XXX
# Target: >= 0.91 (up from 0.8939 baseline)
"
```
**Pass condition:** Log output shows `Warranty Decision accuracy` ≥ 0.905.
If the result is between 0.895 and 0.905, training succeeded but gains were
modest — check T5 for the OOF cascade specifically.

---

### T5 — OOF cascade distribution check
Paste this snippet **inside `train_and_save()` temporarily** immediately after
the `fa_probs_tr = cross_val_predict(...)` block to verify calibration:

```python
    print(f"OOF FA probs — mean: {fa_probs_tr.max(axis=1).mean():.4f}  "
          f"std: {fa_probs_tr.max(axis=1).std():.4f}")
```
**Pass condition:** OOF mean top-class probability should be **lower than 0.98**
(the old value was 0.9888 from in-sample scoring) and std should be higher
(wider spread, more realistic). A value around 0.96–0.975 mean with std ~0.07
is expected. This confirms the calibration gap is closed.

---

### T6 — Inference smoke test (end-to-end predict)
```bash
python -c "
from ml_predictor import predict

cases = [
    # (fault_code,   notes,                                    voltage, expected_wd)
    ('P0301',        'Engine overheating, low idle',            14.2,  'Production Failure'),
    ('U0100',        'Communication error on CAN bus',          12.5,  'Production Failure'),
    ('P0301',        'Moisture found inside connector',         12.0,  'Customer Failure'),
    ('',             'No fault found, intermittent complaint',  13.1,  'According to Specification'),
    ('B1234',        'Starting problem, nothing visible',       18.5,  'Customer Failure'),
]
for fc, notes, v, expected_wd in cases:
    r = predict(fc, notes, v)
    status = 'PASS' if r['warranty_decision'] == expected_wd else 'FAIL'
    print(f'[{status}] {r[\"warranty_decision\"]:25s} (expected: {expected_wd}) | conf: {r[\"confidence\"]}%')
"
```
**Pass condition:** All 5 cases print `[PASS]`. These cases correspond to the
smoke tests already in `__main__` at the bottom of `ml_predictor.py` and
should produce deterministic results with the same trained model.

---

### T7 — Bundle round-trip test (pickle load → inference)
```bash
python -c "
# Simulate a fresh process loading the saved model (not re-training)
import ml_predictor
ml_predictor._bundle = None   # force reload from disk
result = ml_predictor.predict('P0301', 'Engine overheating', 14.2)
print('Round-trip inference OK:', result['warranty_decision'], result['confidence'])
assert 'warranty_decision' in result
assert 'failure_analysis' in result
print('All keys present in output dict.')
"
```
**Pass condition:** No `KeyError` on bundle keys, result dict has all expected
fields. This specifically tests that `ohe_mileage`, `ohe_vband`, and
`claim_age_scaler` are correctly loaded from the `.pkl` file in a fresh
`run_ml()` invocation.

---

*End of plan. Total changes: 10 code locations across 3 functions (`train_and_save`, `run_ml`, `predict`) and 1 module-level section (imports + helper function).*
