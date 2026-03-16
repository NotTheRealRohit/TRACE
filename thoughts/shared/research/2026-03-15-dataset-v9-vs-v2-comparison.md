---
date: 2026-03-15T15:30:00+00:00
researcher: Claude (opencode)
git_commit: d05262ed6f308f79eac2b76b4b8687eebe31f3fc
branch: feature/upgrade-dataset-and-ml
repository: capProj-2
topic: "Dataset comparison: synthetic_warranty_claims_v9 vs v2 for ml_predictor.py"
tags: [research, dataset, ml, warranty-claims, feature-engineering]
status: complete
last_updated: 2026-03-15
last_updated_by: Claude (opencode)
---

# Research: Dataset Comparison - synthetic_warranty_claims_v9 vs v2

**Date**: 2026-03-15T15:30:00+00:00  
**Researcher**: Claude (opencode)  
**Git Commit**: d05262ed6f308f79eac2b76b4b8687eebe31f3fc  
**Branch**: feature/upgrade-dataset-and-ml  
**Repository**: capProj-2

## Research Question

How different is the new dataset (`synthetic_warranty_claims_v9.csv`) compared to the current implementation's dataset (`synthetic_warranty_claims_v2.csv`) for the `ml_predictor.py`? What compatibility issues might arise when switching datasets?

## Summary

The v9 dataset has **significant differences** from v2 that will impact both the rule-based engine and ML model:

| Aspect | v2 | v9 | Impact |
|--------|----|----|-------|
| **Row Count** | 50,000 | 100,000 | ✓ Double size |
| **Date Range** | 2019-2024 | 2019-2025 | ⚠️ New 2025 data |
| **High Voltage → Customer Failure** | 100% | 93.7% | ⚠️ Pattern weakened |
| **Low Voltage → Production Failure** | 100% | 95.4% | ⚠️ Pattern weakened |
| **U-Code → Production Failure** | 49% | 56.5% | ⚠️ Pattern changed |
| **NTF → According to Spec** | 100% | 95.7% | ⚠️ Pattern weakened |

**Key Finding**: v9 has **noisier, less deterministic patterns** compared to v2's clean signal. The current ml_predictor.py trained on v2 will experience reduced accuracy when processing v9 data because:
1. Rule-based patterns no longer have 100% correlation
2. ML model was trained on cleaner, more deterministic data
3. More edge cases and ambiguous classifications exist in v9

## Detailed Findings

### 1. Dataset Size and Structure

- **v2**: 50,000 rows, columns: `Customer, Year, Date, QC_Number, Customer Complaint, DTC, Voltage, Failure Analysis, Warranty Decision, Supplier, Mileage_km`
- **v9**: 100,000 rows, **same columns** - technically schema-compatible
- Both datasets have same unique values for categorical columns (Warranty Decision, Failure Analysis, Customer Complaints, Suppliers, Customers)

### 2. Date Range Expansion

- **v2**: 2019-01-01 to 2024-12-31
- **v9**: 2019-01-01 to 2025-12-31 (includes full 2025)

Year distribution shows v9 has more recent data:
```
v2: 2019=4,069, 2020=5,055, 2021=6,920, 2022=8,957, 2023=10,939, 2024=14,060
v9: 2019=3,970, 2020=6,994, 2021=9,894, 2022=13,003, 2023=18,237, 2024=22,922, 2025=24,980
```

### 3. Critical Pattern Degradation

The rule-based engine relies on deterministic patterns. v9 has introduced **noise** into these patterns:

| Rule Pattern | v2 Correlation | v9 Correlation | Rule Impact |
|-------------|---------------|----------------|-------------|
| High Voltage (>16V) → Customer Failure | **100%** | 93.7% | `over_voltage` rule weakened |
| Low Voltage (<11V) → Production Failure | **100%** | 95.4% | `low_voltage` rule weakened |
| U-Code → Production Failure | 49% | 56.5% | `u_code` rule more accurate |
| C-Code → Production Failure | 78.8% | 80.4% | `c_code` rule slightly better |
| B-Code → Production Failure | 60% | 80.7% | `b_code` rule improved |
| NTF → According to Spec | **100%** | 95.7% | NTF pattern weakened |

### 4. Voltage Distribution

Both datasets have similar voltage ranges but v9 has slightly more extreme values:
- **v2**: min=9.00V, max=20.00V, mean=14.03V
- **v9**: min=8.51V, max=21.00V, mean=13.98V

Voltage band breakdown:
```
v2: normal=35,418, over_voltage=10,000, under_voltage=4,582
v9: normal=72,924, over_voltage=18,857, under_voltage=8,219
```

### 5. DTC Code Distribution

v9 has more DTC codes overall, with increased variety:
- **v2**: ~56% have DTCs, ~25% have none
- **v9**: ~76% have DTCs, ~24% have none

DTC prefix patterns are similar but v9 has more multi-prefix codes.

### 6. Warranty Decision Distribution

Both datasets have balanced classes:
```
v2: Customer Failure=20,427, According to Spec=15,000, Production Failure=14,573
v9: Customer Failure=40,114, According to Spec=30,111, Production Failure=29,775
```

### 7. Failure Analysis Distribution

v2 had perfect deterministic mapping:
- NTF → According to Specification (100%)
- Track burnt due to EOS → Customer Failure (100%)
- Connector damage → Production Failure (100%)
- Sensor short due to moisture → Customer Failure (100%)
- ASIC CJ327 failure due to EOS → Customer Failure (100%)
- controller failure due to supplier production failure → Production Failure (100%)

v9 has **noisy mappings**:
- NTF → 95.7% According to Spec, 3.2% Customer Failure, 1% Production Failure
- Track burnt due to EOS → 95.8% Customer Failure, 4.2% other
- And similar noise in other categories

## Code References

- `backend/ml_predictor.py:92` - Current DATA_PATH points to v2: `DATA_PATH = os.path.join(BASE_DIR, "synthetic_warranty_claims_v2.csv")`
- `backend/ml_predictor.py:107-195` - RULES list with pattern matching logic
- `backend/ml_predictor.py:264-394` - `train_and_save()` function that trains on DATA_PATH

## Architecture Insights

1. **Rule Engine Sensitivity**: The rule-based engine (lines 107-195) relies on deterministic patterns. With v9's noisy data, rules will fire correctly ~95% of the time instead of 100%.

2. **ML Model Retraining Required**: The RandomForest classifiers were trained on v2's clean patterns. When using v9:
   - Feature distributions will differ
   - Target class balance remains similar
   - Accuracy will likely decrease unless model is retrained

3. **Compatibility**: Schema is compatible - same columns, same categorical values. Only the data distribution has changed.

## Recommendations

1. **Retrain ML Models**: Delete `trace_models.pkl` and restart the server to trigger retraining on v9
2. **Update DATA_PATH**: Change line 92 in `ml_predictor.py` from `synthetic_warranty_claims_v2.csv` to `synthetic_warranty_claims_v9.csv`
3. **Expect Lower Base Accuracy**: v9's noisier patterns mean the model won't achieve the same clean accuracy as v2
4. **Consider Rule Confidence Review**: Rule confidences in RULES list may need recalibration for v9's distributions

## Open Questions

- Should rule confidences be recalibrated for v9's weaker patterns?
- Is the 6% noise in high-voltage patterns acceptable for production?
- Should v9 be considered a more realistic (noisy) dataset vs v2's idealized patterns?
