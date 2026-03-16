---
date: 2026-03-15T18:30:00+05:30
researcher: opencode
git_commit: d05262ed6f308f79eac2b76b4b8687eebe31f3fc
branch: feature/upgrade-dataset-and-ml
repository: TRACE
topic: "Coupling Analysis: synthetic_warranty_claims_v2.csv and ml_predictor.py"
tags: [research, ml, dataset, coupling, warranty-claims, dtc-codes]
status: complete
last_updated: 2026-03-15
last_updated_by: opencode
---

# Research: Dataset-to-ML Model Coupling Analysis

**Date**: 2026-03-15T18:30:00+05:30
**Researcher**: opencode
**Git Commit**: d05262ed6f308f79eac2b76b4b8687eebe31f3fc
**Branch**: feature/upgrade-dataset-and-ml
**Repository**: TRACE (git@github.com:NotTheRealRohit/TRACE.git)

## Research Question

How tightly coupled is `synthetic_warranty_claims_v2.csv` with `ml_predictor.py`, and what changes would be needed to update the dataset to 100K records with new DTC codes?

## Summary

The dataset and ML predictor are **heavily coupled** through multiple layers:

1. **Feature extraction** depends on specific DTC prefixes (P, U, C, B) and a hardcoded list of 9 "high-value" DTCs
2. **Label encoding** requires exact Failure Analysis and Warranty Decision class names
3. **Rule engine** uses voltage thresholds and DTC prefix patterns directly derived from dataset distributions
4. **Customer complaint mapping** uses a fixed 14-type taxonomy
5. **Voltage bands** are hardcoded to match dataset distributions

Updating to 100K records with new DTC codes requires changes in both the dataset generation and ml_predictor.py.

---

## Detailed Findings

### 1. Dataset Column Dependencies

The dataset has these columns:
- `Customer`, `Year`, `Date`, `QC_Number`, `Customer Complaint`, `DTC`, `Voltage`, `Failure Analysis`, `Warranty Decision`, `Supplier`, `Mileage_km`

**Coupling Points** ([ml_predictor.py:264-271](backend/ml_predictor.py#L264-L271)):
```python
df["DTC"]                = df["DTC"].fillna("").replace("none", "")
df["Customer Complaint"] = df["Customer Complaint"].fillna("OBD Light ON")
df["Failure Analysis"]   = df["Failure Analysis"].fillna("NTF")
df["Warranty Decision"]  = df["Warranty Decision"].fillna("According to Specification")
df["Voltage"]            = pd.to_numeric(df["Voltage"], errors="coerce").fillna(12.5)
```

### 2. HIGH_VALUE_DTCS — Hardcoded DTC List

**Location**: [ml_predictor.py:102-105](backend/ml_predictor.py#L102-L105)
```python
HIGH_VALUE_DTCS = [
    "P0300", "P0615", "P0481", "P1682", "P0301",
    "P0480", "P0073", "P0304", "P0482"
]
```

This list creates one-hot encoded features for each DTC. **New DTC codes must be added here** to be recognized as significant.

### 3. KNOWN_COMPLAINTS — Fixed Taxonomy

**Location**: [ml_predictor.py:94-100](backend/ml_predictor.py#L94-L100)
```python
KNOWN_COMPLAINTS = [
    "Engine jerking during acceleration", "Starting Problem",
    "High fuel consumption", "OBD Light ON", "Vehicle not starting",
    "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON",
    "ABS warning light ON", "Battery warning light ON",
    "Engine stalling", "Multiple warning lights ON", "Transmission jerking",
]
```

The dataset generator uses these 14 complaint types. The ML model uses OneHotEncoder on this exact set.

### 4. Rule Engine Voltage Thresholds

**Location**: [ml_predictor.py:107-195](backend/ml_predictor.py#L107-L195)

The rules are derived from dataset voltage distributions:

| Rule ID | Threshold | Dataset Voltage Range | Failure Analysis |
|---------|-----------|----------------------|------------------|
| `over_voltage` | V > 16.0 | Track burnt: 16.1-20.0V | Track burnt due to EOS |
| `low_voltage` | V < 11.0 | Controller: 9.0-11.5V | controller failure due to supplier production failure |

These thresholds are **hardcoded** and directly mirror the synthetic data generation logic in `generate_dataset_v6.py`:
- ASIC: 14.2-16.0V
- Track burnt: 16.1-20.0V  
- Sensor moisture: 11.0-13.5V
- Controller: 9.0-11.5V

### 5. DTC Prefix Detection in Rules

**Location**: [ml_predictor.py:157-194](backend/ml_predictor.py#L157-L194)

```python
# u_code rule - line 157-163
"id": "u_code",
"match": lambda fc, notes, v: bool(re.search(r'\bU[0-9A-Fa-f]{4}\b', fc)),
# ... similar for p_code, c_code, b_code
```

The rules detect DTC prefixes: U (CAN/LIN), P (powertrain), C (chassis), B (body). New DTCs will work if they follow these prefixes.

### 6. extract_dtc_features() Function

**Location**: [ml_predictor.py:198-212](backend/ml_predictor.py#L198-L212)

```python
def extract_dtc_features(dtc_str: str) -> dict:
    codes = [c.strip() for c in s.split(",") if c.strip()]
    return {
        "dtc_count": len(codes),
        "has_P": int(any(c.startswith("P") for c in codes)),
        "has_U": int(any(c.startswith("U") for c in codes)),
        "has_C": int(any(c.startswith("C") for c in codes)),
        "has_B": int(any(c.startswith("B") for c in codes)),
        "dtc_text": " ".join(codes),
        **{f"dtc_{d.lower()}": int(d in codes) for d in HIGH_VALUE_DTCS},
    }
```

This creates:
- DTC count
- Binary flags for P/U/C/B prefixes
- TF-IDF on DTC text (max 40 features)
- One-hot flags for HIGH_VALUE_DTCS

### 7. Label Encoding Classes

**Location**: [ml_predictor.py:275-276](backend/ml_predictor.py#L275-L276)

```python
le_fa = LabelEncoder(); y_fa = le_fa.fit_transform(df["Failure Analysis"])
le_wd = LabelEncoder(); y_wd = le_wd.fit_transform(df["Warranty Decision"])
```

**Failure Analysis classes** (6 classes from dataset):
- NTF
- Track burnt due to EOS
- ASIC CJ327 failure due to EOS
- Sensor short due to moisture
- Connector damage
- controller failure due to supplier production failure

**Warranty Decision classes** (3 classes):
- Production Failure
- Customer Failure
- According to Specification

### 8. voltage_band() Function

**Location**: [ml_predictor.py:246-261](backend/ml_predictor.py#L246-L261)

```python
def voltage_band(v: float) -> str:
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

This buckets voltage to match rule thresholds - tied to dataset voltage distributions.

### 9. match_complaint() Fallback Mapping

**Location**: [ml_predictor.py:215-243](backend/ml_predictor.py#L215-L243)

Maps free-text technician notes to the 14 KNOWN_COMPLAINTS. Uses keyword matching and fuzzy matching.

---

## Code References

- `backend/ml_predictor.py:92` — DATA_PATH points to CSV
- `backend/ml_predictor.py:102-105` — HIGH_VALUE_DTCS list
- `backend/ml_predictor.py:94-100` — KNOWN_COMPLAINTS list  
- `backend/ml_predictor.py:107-195` — RULES with voltage thresholds and DTC prefix patterns
- `backend/ml_predictor.py:198-212` — extract_dtc_features()
- `backend/ml_predictor.py:215-243` — match_complaint()
- `backend/ml_predictor.py:246-261` — voltage_band()
- `backend/ml_predictor.py:264-394` — train_and_save()
- `backend/dataset_gen/generate_dataset_v6.py` — Dataset generation (50K rows)
- `backend/evaluate_model.py` — Model evaluation script

---

## Architecture Insights

### Tight Coupling Design

The system is intentionally designed with tight coupling for **domain consistency**:

1. **Voltage-based rules** directly mirror synthetic data generation voltage ranges
2. **DTC prefix detection** (P/U/C/B) matches the synthetic data's structured DTC pools
3. **One-hot encoded features** for HIGH_VALUE_DTCS create ML signal for specific codes
4. **Customer complaint taxonomy** is shared between generation and ML

### What's Good About This Design

- Strong ML signal from synthetic data (designed correlations)
- Rule engine provides deterministic fallback for edge cases
- TF-IDF on DTC text captures general DTC patterns
- Cascaded ML (FA → WD) improves warranty decision accuracy

### What Will Break With New DTC Codes

1. **New DTCs not in HIGH_VALUE_DTCS** — Won't get one-hot feature; rely only on TF-IDF
2. **New DTC prefixes** (e.g., "M" for manufacturer-specific) — Won't be detected by rules
3. **New complaint types** — Won't be OneHotEncoded properly; will default to "OBD Light ON"
4. **New Failure Analysis types** — LabelEncoder won't know the class; will error

---

## Dataset Generation Analysis

The current dataset generator (`generate_dataset_v6.py`) produces:

- **50,000 rows** (but AGENTS.md mentions 50K, user wants 100K)
- **14 customer complaint types**
- **6 Failure Analysis types**
- **3 Warranty Decision types**
- **Specific voltage ranges per failure class**
- **DTC pools** that map to failure types

### DTC Pools in Generator (v6)

| Pool | DTCs | Failure Type |
|------|------|--------------|
| DTC_ASIC | P0601-P0613, P0562, P0563 | ASIC CJ327 failure |
| DTC_TRACK | P0562, P0563, P0300-P0304, P0480-P0482, P1682, P0615, P0620, U0001, U0100, U0101, U0155 | Track burnt |
| DTC_SENSOR_MOISTURE | P0113-P0197, P0072-P0073, P0038, P0054, P0131, P0135, P0069 | Sensor short |
| DTC_CONNECTOR | C0031-C0051, C0265, C0460, C0550, B1234, B1031, B1045, B2960, B3055 | Connector damage |
| DTC_CONTROLLER | U0073-U0184 | Controller failure |
| DTC_NTF_MILD | (mostly empty), P0455, P0456, P0171, P0174, P0340, P0325 | NTF |

---

## Recommendations for 100K Records + New DTC Codes

### 1. Update HIGH_VALUE_DTCS in ml_predictor.py

Add new significant DTCs to the one-hot feature list:
```python
HIGH_VALUE_DTCS = [
    "P0300", "P0615", "P0481", "P1682", "P0301",
    "P0480", "P0073", "P0304", "P0482",
    # Add new high-signal DTCs here
]
```

### 2. Regenerate the Model

After updating dataset, delete the pickle file to force retraining:
```bash
rm backend/trace_models.pkl
```

The model auto-trains on startup.

### 3. Add New Complaint Types (if needed)

If new complaint types are added to dataset:
1. Update `KNOWN_COMPLAINTS` list
2. Update dataset generator's `ALL_COMPLAINTS`
3. Re-train model (delete pickle)

### 4. Add New Failure Analysis Types (if needed)

This requires code changes:
1. Add new class to LabelEncoder (will happen automatically on fit)
2. Consider adding new rule if distinct pattern exists
3. Re-train model

### 5. Scale to 100K

Simply increase the target in dataset generator:
```python
TARGET = 100_000  # was 50_000
```

The ML model should handle this without changes (RandomForest scales reasonably).

---

## Historical Context

The codebase has multiple dataset generator versions (v3-v6 in `/backend/dataset_gen/`). The current v6 is the most sophisticated with:
- DTC→Complaint semantic biasing
- Seasonal distributions
- Voltage zones with clean class separation
- Realistic NTF complaint diversity
- Label noise injection (~1.5%)

---

## Open Questions

1. **What new DTC codes are planned?** — Need to know specific codes to update HIGH_VALUE_DTCS
2. **Will failure analysis taxonomy change?** — New classes would require rule + ML updates
3. **Will complaint taxonomy change?** — Would require KNOWN_COMPLAINTS update
4. **Is the voltage distribution changing?** — Rule thresholds may need adjustment
5. **What is the target ratio of 50K→100K?** — Simply doubling or new data distribution?

---

## Related Research

No prior research documents found in `thoughts/shared/research/` for this topic.

---

## Conclusion

The dataset and ML predictor are tightly coupled by design. Updating to 100K records with new DTC codes requires:

1. **Low risk**: Simply regenerating dataset and retraining (delete pickle)
2. **Medium risk**: Adding new DTCs to HIGH_VALUE_DTCS list
3. **High risk**: Changing Failure Analysis taxonomy (requires rule + code changes)

The system was designed for synthetic data with specific correlations - it will work with new data but may lose some signal if the new DTCs don't follow the existing prefix patterns (P/U/C/B).
