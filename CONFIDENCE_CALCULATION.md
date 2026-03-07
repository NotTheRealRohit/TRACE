# TRACE Warranty Claim Analysis - Confidence Calculation Logic

## Overview

This document explains how the TRACE system calculates the **confidence score** for warranty claim decisions. The confidence score indicates how certain the system is about its prediction (0-100%, capped at 98%).

The system uses a **hybrid approach** combining:
1. Rule-based engine (domain knowledge)
2. Machine Learning (RandomForest models)
3. Optional LLM enhancement (when available)

---

## Confidence Calculation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLAIM INPUT                                           │
│  fault_code: "P0562"  │  technician_notes: "Engine overheating"  │  voltage │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: RULE ENGINE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Check against 9 predefined rules                                     │    │
│  │  - Voltage thresholds (over/under voltage)                         │    │
│  │  - Keyword detection (moisture, physical damage, NTF)               │    │
│  │  - DTC code prefix analysis (P, U, C, B)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                          │
│                    Rule Matches? ──┼── No ──► rule_fired = False             │
│                         │          │                                          │
│                        Yes         │                                          │
│                         ▼          │                                          │
│              rule_confidence = [Predefined per rule]                        │
│              e.g., over_voltage → 94.0                                        │
│              e.g., moisture → 91.0                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ML MODEL SCORING                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Extract features:                                                   │    │
│  │   - customer_complaint (from notes)                                │    │
│  │   - DTC text/count/prefixes (P,U,C,B)                              │    │
│  │   - voltage                                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Run RandomForest classifiers:                                       │    │
│  │   - clf_fa: Failure Analysis (root cause)                          │    │
│  │   - clf_wd: Warranty Decision (Production/Customer/Spec)         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                          │
│                                    ▼                                          │
│  Get probability distributions:                                              │
│     fa_prob = max probability for Failure Analysis prediction              │
│     wd_prob = max probability for Warranty Decision prediction            │
│                                                                             │
│  ml_confidence = geometric_mean(fa_prob, wd_prob) × 100                    │
│                = √(fa_prob × wd_prob) × 100                                  │
│                                                                             │
│  Capped between 50% - 98%                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: COMBINE SCORES                                                     │
│                                                                             │
│  Three scenarios:                                                            │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ SCENARIO A: Rule fired AND agrees with ML                          │    │
│  │                                                                    │    │
│  │ agreement = (rule_warranty_decision == ml_warranty_decision)      │    │
│  │                                                                    │    │
│  │ combined_confidence =                                              │    │
│  │     (0.7 × rule_conf) + (0.3 × ml_conf) + 5.0  (agreement bonus) │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ SCENARIO B: Rule fired BUT disagrees with ML                     │    │
│  │                                                                    │    │
│  │ combined_confidence =                                              │    │
│  │     (0.6 × rule_conf) + (0.1 × ml_conf)                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ SCENARIO C: No rule fired                                         │    │
│  │                                                                    │    │
│  │ combined_confidence = ml_conf                                      │    │
│  │                                                                    │    │
│  │ + If LLM categorized as "other" and no rule: ×0.85 penalty       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Final cap: min(98%, max(0%, combined_confidence))                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: DETERMINE STATUS                                                  │
│                                                                             │
│  Using thresholds:                                                           │
│     CONFIDENCE_THRESHOLD_FIRM   = 85.0                                      │
│     CONFIDENCE_THRESHOLD_MANUAL = 65.0                                      │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ If confidence ≥ 85%:  → APPROVED or REJECTED (firm decision)      │    │
│  │ If confidence ≥ 65%:  → APPROVED or REJECTED (needs review)       │    │
│  │ If confidence < 65%:  → NEEDS MANUAL REVIEW                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Mapping:                                                                   │
│     "Production Failure"         → Approved                                 │
│     "According to Specification" → Approved                                 │
│     "Customer Failure"          → Rejected                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINAL OUTPUT                                        │
│  {                                                                           │
│    status: "Approved" | "Rejected" | "Needs Manual Review",                 │
│    warranty_decision: "Production Failure" | "Customer Failure" | ...,      │
│    confidence: 85.0,                                                        │
│    failure_analysis: "...",                                                 │
│    decision_engine: "Rule+ML" | "LLM+Rule+ML" | "ML"                        │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Explanation

### 1. Rule-Based Confidence

Each rule has a **hardcoded confidence** based on domain expertise:

| Rule ID | Condition | Confidence | Reason |
|---------|-----------|------------|--------|
| `over_voltage` | voltage > 16V | 94.0% | Clear customer-side fault (EOS) |
| `moisture` | Keywords: water, moisture, wet... | 91.0% | Environmental contamination |
| `physical_damage` | Keywords: crack, broken, impact... | 88.5% | Obvious customer misuse |
| `u_code` | DTC starts with U | 85.0% | CAN communication = production defect |
| `low_voltage` | voltage < 11V | 83.0% | Faulty voltage regulator = production |
| `ntf` | Keywords: no fault, NTF... | 82.0% | Vehicle within spec |
| `p_code_engine` | P0xxx + engine symptoms | 80.5% | ECU-level production fault |
| `c_code` | DTC starts with C | 78.0% | Chassis connector issue |
| `b_code` | DTC starts with B | 76.0% | Body electronics wiring |

**Why predefined?** These rules represent domain knowledge where the automotive industry has established causation between symptoms and root causes.

---

### 2. ML Model Confidence

The ML confidence uses a **geometric mean** of two probabilities:

```python
ml_confidence = √(fa_prob × wd_prob) × 100
```

**Why geometric mean?**
- If the model is 90% confident in failure analysis but only 50% confident in warranty decision, we shouldn't be overly confident
- Geometric mean penalizes cases where one probability is low
- Arithmetic mean would give 70% (too optimistic); geometric mean gives 67% (more realistic)

**Why cap at 50-98%?**
- **Floor (50%)**: Better than random chance for a 3-class problem
- **Ceiling (98%)**: The dataset is synthetically balanced with near-random correlations, so 100% confidence is unrealistic

---

### 3. Combining Rule + ML Scores

The system uses **weighted averaging** with different weights based on agreement:

#### Scenario A: Rule Agrees with ML
```
combined = 0.7 × rule_conf + 0.3 × ml_conf + 5.0 (bonus)
```

**Rationale:**
- Rule-based gets higher weight (0.7) because it represents domain certainty
- ML gets lower weight (0.3) as supporting evidence
- +5.0 bonus rewards agreement → pushes confidence higher when both systems align

#### Scenario B: Rule Disagrees with ML
```
combined = 0.6 × rule_conf + 0.1 × ml_conf
```

**Rationale:**
- Rule still wins (0.6) but less than agreement case
- ML almost ignored (0.1) because it contradicts the rule
- No bonus (obviously)

#### Scenario C: No Rule Fired
```
combined = ml_conf
```

**Rationale:**
- Falls back entirely to ML prediction
- Additional penalty if LLM categorized input as "other" (unclear category)

---

### 4. Status Determination

| Confidence | Status |
|------------|--------|
| ≥ 85% | **Approved** or **Rejected** (firm decision) |
| 65% - 84% | **Approved** or **Rejected** (needs review) |
| < 65% | **Needs Manual Review** |

The system maps warranty decisions to statuses:
- `Production Failure` → Approved
- `According to Specification` → Approved  
- `Customer Failure` → Rejected

---

## Example Walkthrough

### Example 1: Over-voltage detection
```
Input: fault_code="", notes="", voltage=17.5

Rule Check:
  - over_voltage rule matches (17.5 > 16.0)
  - rule_confidence = 94.0%
  - rule_warranty_decision = "Customer Failure"

ML Prediction:
  - ml_warranty_decision = "Customer Failure" (let's say)
  - fa_prob = 0.85, wd_prob = 0.88
  - ml_confidence = √(0.85 × 0.88) × 100 = 86.5%

Combine (Agreement = True):
  combined = 0.7 × 94.0 + 0.3 × 86.5 + 5.0
           = 65.8 + 25.95 + 5.0
           = 96.75% → capped at 96.8%

Status: confidence 96.8% ≥ 85% → REJECTED
```

### Example 2: No rule, low ML confidence
```
Input: fault_code="P1234", notes="strange noise", voltage=12.5

Rule Check:
  - No rule matches
  - rule_fired = False

ML Prediction:
  - ml_warranty_decision = "Production Failure"
  - fa_prob = 0.45, wd_prob = 0.52
  - ml_confidence = √(0.45 × 0.52) × 100 = 48.4% → capped at 50%

Combine (No rule):
  combined = 50.0%

Status: confidence 50% < 65% → NEEDS MANUAL REVIEW
```

---

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Rules have predefined confidence | Domain expertise is more reliable than ML on synthetic data |
| Geometric mean for ML confidence | Prevents overconfidence when one model is uncertain |
| Agreement bonus (+5%) | Rewards consensus between rule and ML |
| Rule weight > ML weight when agreeing | Rules represent established automotive causation |
| ML almost ignored when disagreeing | Rule expertise trumps ML when there's conflict |
| 85% threshold for firm decision | High bar for automatic approval/rejection |
| 65% threshold for manual review | Ensures uncertain cases get human attention |

---

## Constants Reference

```python
# Thresholds
CONFIDENCE_THRESHOLD_FIRM   = 85.0   # Auto-approve/reject
CONFIDENCE_THRESHOLD_MANUAL = 65.0   # Human review needed

# Weights for agreement
RULE_WEIGHT_AGREE   = 0.7
ML_WEIGHT_AGREE     = 0.3
AGREEMENT_BONUS     = 5.0

# Weights for disagreement
RULE_WEIGHT_DISAGREE   = 0.6
ML_WEIGHT_DISAGREE     = 0.1

# Penalties
DISAGREEMENT_GAP_THRESHOLD = 20.0   # Gap triggers manual review
WEAK_INPUT_PENALTY         = 0.85    # Applied to unclear LLM categories
```
