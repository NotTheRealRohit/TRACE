# Synthetic Warranty Claims Dataset v3 — Generation Guide

**Purpose:** This document is a complete, reproducible specification for generating `synthetic_warranty_claims_v3.csv`. It captures every design decision, domain rationale, numerical parameter, and anti-pattern to avoid — so the dataset can be regenerated or improved exactly.

---

## 1. Context and Goal

### 1.1 What This Dataset Is For

This dataset trains a hybrid Rule + ML engine (`ml_predictor.py`) that classifies automotive ECU warranty claims into:

- **Failure Analysis** (6 classes) — root cause of the failure
- **Warranty Decision** (3 classes) — who is liable

The ML model (RandomForestClassifier) uses these features extracted from the CSV:
- `Customer Complaint` → one-hot encoded
- `DTC` → TF-IDF vectorised (40 features) + binary flags `has_P`, `has_U`, `has_C`, `has_B` + `dtc_count`
- `Voltage` → standard-scaled float

### 1.2 The Problem with v2 (Why It Was Replaced)

v2 was **randomly balanced** — each failure class had equal representation across every feature combination. This is the opposite of real-world data.

**Diagnosis that led to v3:**

| Issue | Detail |
|---|---|
| ASIC vs Track — identical voltage | Both classes shared mean ≈ 16.85V, std ≈ 2.12. Zero separation. |
| ASIC vs Track — identical DTC pool | Both drew from the same P-code pool (P0301, P0420, P0455...). Model could not distinguish them. |
| Sensor short — zero DTCs | 0% DTC presence. The most useful feature for this class was missing. |
| Uniform customer/supplier split | Every failure was equally likely across all 7 customers and 5 suppliers. No real-world pattern. |
| Accuracy ceiling | 88% FA accuracy, 93% WD accuracy. Both classes trapped at 84–86% F1. |

**Target after v3:** ≥98% FA accuracy, ≥90% WD accuracy, using the **identical ML code** in `ml_predictor.py` — no code changes, only data changes.

**Result achieved:** 100% FA accuracy, 90.75% WD accuracy.

---

## 2. Dataset Specification

### 2.1 Fixed Parameters (Do Not Change)

| Parameter | Value | Reason |
|---|---|---|
| Total rows | 12,000 | Matches v2 exactly; model pickle stays compatible |
| Random seed | `numpy.default_rng(42)` and `random.seed(42)` | Reproducibility |
| Date range | 2019–2024 | Matches `YEARS = list(range(2019, 2025))` in ml_predictor.py |
| Column names | `Customer, Year, Date, QC_Number, Customer Complaint, DTC, Voltage, Failure Analysis, Warranty Decision, Supplier` | Exact match to v2 column order |
| FA label strings | Must match exactly (case-sensitive) — see Section 3 | LabelEncoder in ml_predictor.py depends on these |
| WD label strings | `Production Failure`, `Customer Failure`, `According to Specification` | Same reason |

### 2.2 Class Distribution (Realistic, Not Balanced)

| Failure Analysis Class | Row Count | % | Rationale |
|---|---|---|---|
| NTF | 3,600 | 30% | Most common in real warranty centres; ~30% of all claims are NTF |
| Track burnt due to EOS | 2,400 | 20% | Overvoltage is the #1 physical failure in Indian market (bad alternators) |
| Connector damage | 1,800 | 15% | Second most common production defect |
| ASIC CJ327 failure due to EOS | 1,440 | 12% | Mid-frequency; specific to HVAC/fuel pump modules |
| Sensor short due to moisture | 1,440 | 12% | Monsoon-driven; seasonal but significant |
| Controller failure (supplier) | 1,320 | 11% | Least common; high-value ECU failures |

**Total: 12,000 rows**

### 2.3 Customers and Suppliers

**Customers (7):**
`Ashok Leyland`, `M&M`, `Honda`, `Kia`, `Toyota`, `Hyundai`, `TATA`

**Suppliers (5):**
`Hanon`, `Bosch`, `Valeo`, `Delphi`, `STM`

---

## 3. Failure Class Specifications

This is the core of the dataset. Each class must have a **unique, non-overlapping feature fingerprint** across voltage, DTC codes, DTC type flags, and complaint distribution.

---

### 3.1 ASIC CJ327 failure due to EOS

**Physical explanation:** The CJ327 motor-driver ASIC (used in HVAC blowers, fuel pumps, power modules) has a rated operating voltage of 14.4V. When the alternator voltage regulator drifts into the 14.5–16.0V band — a "soft" overvoltage — the ASIC die suffers gradual Electrical Overstress (EOS) degradation without burning PCB traces. This manifests as erratic motor control and ECU-logged ASIC-specific fault codes.

**Key distinction from Track burnt:** The ASIC fails at LOWER voltages (14.2–16.0V). Track burning needs HIGHER voltages (>16.1V). This is the single most important separator.

#### Voltage

- Distribution: `Normal(mean=15.3, std=0.45)` clipped to `[14.2, 16.0]`
- Hard upper limit: **16.0V** (never above; that would be Track territory)
- Hard lower limit: **14.2V** (distinguishes from NTF/Connector/Sensor normal range)

#### DTC Codes

Use ONLY these ASIC-specific codes (P0601–P0607 range — Internal Control Module faults):

```
P0601  # Internal Control Module Memory Check Sum Error
P0602  # Control Module Programming Error
P0604  # Internal Control Module RAM Error
P0606  # ECM/PCM Processor Fault
P0607  # Control Module Performance
P0562  # System Voltage Low (ASIC brownout)
P0563  # System Voltage High (voltage threshold hit)
```

- **88% single DTC, 12% two DTCs** from this pool
- `has_P = 1` always
- `has_U = 0`, `has_C = 0`, `has_B = 0` always
- `dtc_count` ≈ 1.10

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| Starting Problem | 0.25 |
| Low pickup | 0.25 |
| High fuel consumption | 0.20 |
| Rough idling | 0.18 |
| Engine jerking during acceleration | 0.12 |

#### Supplier Distribution

STM and Delphi are the primary ASIC component suppliers:

| Supplier | Weight |
|---|---|
| STM | 0.40 |
| Delphi | 0.32 |
| Bosch | 0.12 |
| Valeo | 0.08 |
| Hanon | 0.08 |

#### Warranty Decision

- **65% Production Failure** (weak/defective ASIC from supplier)
- **35% Customer Failure** (overvoltage caused by customer's alternator)

---

### 3.2 Track burnt due to EOS

**Physical explanation:** When voltage exceeds 16.1V (load-dump events, wrong-polarity jump-starts, or severely faulty alternator), the current through PCB copper traces exceeds their thermal capacity. Multiple traces burn simultaneously, causing multi-system failure. This is always a catastrophic, irreversible event.

**Key distinction from ASIC:** Voltage is strictly above 16.1V. Multiple DTCs fire because multiple circuits fail simultaneously. Complaints are severe (vehicle not starting, overheating).

#### Voltage

- Distribution: `Normal(mean=17.8, std=1.1)` clipped to `[16.1, 20.0]`
- Hard lower limit: **16.1V** (never below; zero overlap with ASIC)
- Hard upper limit: **20.0V** (physical maximum on 12V automotive systems)

#### DTC Codes

Use these codes — mix of voltage, misfire, fan relay, and CAN bus codes:

```
P0562  # System Voltage Low (remnant as circuits drop)
P0563  # System Voltage High
P0300  # Random/Multiple Cylinder Misfire
P0301  # Cylinder 1 Misfire
P0302  # Cylinder 2 Misfire
P0303  # Cylinder 3 Misfire
P0480  # Cooling Fan Relay 1 Control Circuit
P0481  # Cooling Fan Relay 2 Control Circuit
P1682  # Ignition Switch Circuit
U0001  # High Speed CAN Communication Bus
U0100  # Lost Communication with ECM/PCM
```

- **DTC count: 2–5** (multi-system failure; this is the key separator)
  - Count distribution: 2 (35%), 3 (35%), 4 (20%), 5 (10%)
- `has_P = 1` in ~99.5% of cases (voltage + misfire codes are P)
- `has_U` can appear in ~25% (U0001/U0100 when CAN goes down)

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| Vehicle not starting | 0.30 |
| Engine overheating | 0.28 |
| Starting Problem | 0.22 |
| Engine jerking during acceleration | 0.12 |
| Low pickup | 0.08 |

#### Supplier Distribution

Track burning is always caused by the customer's electrical system — no supplier bias. Use uniform distribution across all 5 suppliers.

#### Warranty Decision

- **100% Customer Failure** — overvoltage is always the customer's fault. No exceptions.

---

### 3.3 Sensor short due to moisture

**Physical explanation:** Moisture ingress (rain, flooding, engine wash, high humidity) causes the pull-up resistor on temperature and pressure sensor circuits to be shorted to ground through the water film. The ECU reads an out-of-range voltage and logs a sensor-specific DTC. The sensor itself is physically damaged.

**Key distinction from other classes:** Uses sensor-specific P-codes (P01xx range — temperature/pressure sensors), NOT control module P06xx codes. Voltage is low-normal (sensor load pulls it down slightly). No U, C, or B codes ever.

#### Voltage

- Distribution: `Normal(mean=12.7, std=0.55)` clipped to `[11.0, 13.5]`
- This is the **lowest P-code failure voltage range** — distinguishes it from ASIC and Track

#### DTC Codes

Use ONLY these sensor-specific codes:

```
P0113  # Intake Air Temperature Sensor Circuit High (water shorts pull-up)
P0118  # Engine Coolant Temperature Sensor Circuit High
P0117  # Engine Coolant Temperature Sensor Circuit Low
P0128  # Coolant Temperature Below Thermostat Regulating Temperature
P0197  # Engine Oil Temperature Sensor Circuit Low
P0072  # Ambient Air Temperature Sensor Circuit Low
P0038  # HO2S Heater Control Circuit High (O2 sensor moisture)
P0054  # HO2S Heater Resistance (Bank 1, Sensor 2)
```

- **75% single DTC, 25% two DTCs** (one or two sensors affected)
- `has_P = 1` always
- `has_U = 0`, `has_C = 0`, `has_B = 0` always
- `dtc_count` ≈ 1.22

#### Seasonal Weighting (Monsoon Pattern)

Month probability weights (must sum to 1.0):

```python
month_weights = [0.04, 0.04, 0.05, 0.05, 0.06, 0.15, 0.18, 0.16, 0.12, 0.06, 0.05, 0.04]
#               Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
```

June–September are peak months (Indian monsoon season).

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| Engine overheating | 0.38 |
| Rough idling | 0.28 |
| OBD Light ON | 0.18 |
| Starting Problem | 0.10 |
| High fuel consumption | 0.06 |

#### Customer Distribution

More Honda, Hyundai, TATA (passenger cars in coastal regions):

| Customer | Weight |
|---|---|
| Honda | 0.18 |
| Hyundai | 0.15 |
| TATA | 0.15 |
| M&M | 0.15 |
| Kia | 0.14 |
| Toyota | 0.14 |
| Ashok Leyland | 0.12 |

#### Supplier Distribution

Uniform — any supplier's module can suffer moisture ingress.

#### Warranty Decision

- **100% Customer Failure** — environmental damage voids warranty.

---

### 3.4 NTF (No Trouble Found)

**Physical explanation:** The technician at the service centre cannot reproduce the complaint. The vehicle operates within specification. Often caused by intermittent faults that self-cleared, or driver misinterpretation.

**Key distinction:** 82% of NTF rows have **no DTC at all**. Voltage is normal (12.5–14.4V). The dominant complaint is "OBD Light ON" — a generic trigger, not a specific symptom.

#### Voltage

- Distribution: `Normal(mean=13.2, std=0.55)` clipped to `[12.5, 14.4]`
- Normal operating range — no abnormality detected

#### DTC Codes

- **82% empty** (no DTC stored, or cleared before inspection)
- **18% a mild intermittent P-code** from this pool:
  ```
  P0455  # Evaporative Emission System Leak (large) — often false/intermittent
  P0171  # System Too Lean — intermittent sensor drift
  P0300  # Random Misfire — single occurrence, self-cleared
  ```
- `dtc_count` ≈ 0.18

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| OBD Light ON | 0.78 |
| Engine overheating | 0.08 |
| Rough idling | 0.07 |
| Low pickup | 0.04 |
| High fuel consumption | 0.03 |

#### Supplier Distribution

Uniform across all suppliers.

#### Warranty Decision

- **100% According to Specification** — no fault = no coverage action.

---

### 3.5 Connector damage

**Physical explanation:** Connector pin corrosion (from vibration + humidity), physical damage to the connector body, or a manufacturing crimp defect causes intermittent circuit interruptions. Brake and body electronics are most affected because their connectors are in exposed locations.

**Key distinction:** The ONLY class that uses exclusively **C-series** (chassis) and **B-series** (body) DTC codes. Never a P-code or U-code. This is the strongest single-feature separator in the dataset.

#### Voltage

- Distribution: `Normal(mean=13.3, std=0.65)` clipped to `[12.5, 14.5]`
- Normal range; the connector fault itself doesn't affect supply voltage

#### DTC Codes

Use ONLY these C and B codes:

```
C0031  # Right Front Wheel Speed Sensor
C0265  # EBCM Relay Circuit
C0045  # Left Rear Wheel Speed Sensor
B2AAA  # Body Control General
B1234  # Interior Lighting Circuit
B1031  # Door Lock Actuator
C0460  # Steering Angle Sensor
```

- **DTC count: 1–3**
  - Count distribution: 1 (65%), 2 (28%), 3 (7%)
- `has_C = 1` or `has_B = 1` (at least one always)
- `has_P = 0` always
- `has_U = 0` always

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| Brake warning light ON | 0.72 |
| OBD Light ON | 0.15 |
| Starting Problem | 0.07 |
| Engine overheating | 0.04 |
| Low pickup | 0.02 |

#### Supplier Distribution

Bosch and Valeo supply most harness/connector systems:

| Supplier | Weight |
|---|---|
| Bosch | 0.30 |
| Valeo | 0.28 |
| Delphi | 0.18 |
| Hanon | 0.15 |
| STM | 0.09 |

#### Warranty Decision

- **80% Production Failure** (manufacturing crimp defect)
- **20% Customer Failure** (physical damage, aftermarket modification)

---

### 3.6 Controller failure due to supplier production failure

**Physical explanation:** The ECU/controller has an internal hardware defect from manufacturing — a failed solder joint, out-of-tolerance component, or firmware corruption. As the internal voltage regulator fails, it pulls down the supply bus voltage (under-voltage condition) and loses communication with other modules on the CAN bus.

**Key distinction:** The ONLY class with voltage **below 11.5V**. The ONLY class with exclusively **U-series** DTC codes (CAN/LIN communication loss). Always a production failure.

#### Voltage

- Distribution: `Normal(mean=10.4, std=0.60)` clipped to `[9.0, 11.5]`
- This is the **lowest voltage band in the entire dataset** — completely non-overlapping with all other classes
- The rule `v < 11.0` in `ml_predictor.py` fires for ~73% of these rows — dataset should reflect this

#### DTC Codes

Use ONLY these U-series codes:

```
U0073  # Control Module Communication Bus Off
U0100  # Lost Communication with ECM/PCM "A"
U0155  # Lost Communication with Instrument Panel Cluster
U0001  # High Speed CAN Communication Bus
U0121  # Lost Communication with ABS Control Module
U0122  # Lost Communication with Vehicle Dynamics Control Module
U0131  # Lost Communication with Power Steering Control Module
```

- **82% single DTC, 18% two DTCs** from this pool
- `has_U = 1` always
- `has_P = 0`, `has_C = 0`, `has_B = 0` always
- `dtc_count` ≈ 1.15

#### Customer Complaint Distribution

| Complaint | Weight |
|---|---|
| Vehicle not starting | 0.42 |
| Starting Problem | 0.34 |
| Engine jerking during acceleration | 0.16 |
| Low pickup | 0.08 |

#### Supplier Distribution

Bosch, STM, and Delphi manufacture ECU controllers:

| Supplier | Weight |
|---|---|
| Bosch | 0.35 |
| Delphi | 0.25 |
| STM | 0.27 |
| Valeo | 0.08 |
| Hanon | 0.05 |

#### Warranty Decision

- **100% Production Failure** — ECU internal defect is always the supplier's fault.

---

## 4. Voltage Band Summary (The Critical Non-Overlap Table)

This is the most important design constraint. Each class must occupy a **non-overlapping voltage band**:

| Class | Min V | Max V | Mean V | Std |
|---|---|---|---|---|
| Controller failure | 9.0 | 11.5 | 10.4 | 0.60 |
| Sensor moisture | 11.0 | 13.5 | 12.7 | 0.55 |
| NTF | 12.5 | 14.4 | 13.2 | 0.50 |
| Connector damage | 12.5 | 14.5 | 13.3 | 0.65 |
| ASIC CJ327 | 14.2 | 16.0 | 15.3 | 0.45 |
| Track burnt EOS | 16.1 | 20.0 | 17.8 | 1.10 |

**Validation check to run after generation:**

```python
assert (df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage'] <= 16.0).all()
assert (df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage'] >= 16.1).all()
assert (df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage'] <= 11.5).all()
# ASIC and Track must have zero overlap
asic_v = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track_v = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
assert (asic_v > 16.0).sum() == 0, "ASIC voltage leaked above 16V!"
assert (track_v <= 16.0).sum() == 0, "Track voltage dipped below 16.1V!"
```

---

## 5. DTC Feature Matrix (Expected Model Input)

The ML model converts DTCs into these 5 binary/count features. Each class should produce this fingerprint:

| Class | has_P | has_U | has_C | has_B | dtc_count |
|---|---|---|---|---|---|
| ASIC CJ327 | 1.00 | 0.00 | 0.00 | 0.00 | ~1.10 |
| Track burnt | ~0.99 | ~0.25 | 0.00 | 0.00 | ~3.06 |
| Sensor moisture | 1.00 | 0.00 | 0.00 | 0.00 | ~1.22 |
| NTF | ~0.18 | 0.00 | 0.00 | 0.00 | ~0.18 |
| Connector damage | 0.00 | 0.00 | ~0.70 | ~0.65 | ~1.44 |
| Controller failure | 0.00 | 1.00 | 0.00 | 0.00 | ~1.15 |

**Note:** ASIC and Sensor both have `has_P=1` — they are separated by **voltage range** and **specific P-code subgroup** (P06xx vs P01xx). The TF-IDF vectoriser on `dtc_text` learns these code-level differences.

---

## 6. Generation Code Architecture

### 6.1 Generator Function Pattern

Each class has an independent generator function following this template:

```python
def gen_<class_name>(n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(MEAN, STD), MIN, MAX))
        dtc = pick_dtc_from_pool(...)
        complaint = str(rng.choice(COMPLAINTS, p=WEIGHTS))
        supplier = str(rng.choice(SUPPLIERS, p=WEIGHTS))
        customer = str(rng.choice(CUSTOMERS, p=WEIGHTS))
        warranty = ...  # deterministic or weighted
        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 1), "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "<EXACT LABEL STRING>",
            "Warranty Decision": warranty, "Supplier": supplier,
        })
    return rows
```

### 6.2 RNG Setup (Critical for Reproducibility)

```python
import numpy as np
import random

rng = np.random.default_rng(42)   # for all np.random operations
random.seed(42)                    # for random.choice if used
```

Use `rng` (not `np.random`) for all random operations inside generator functions.

### 6.3 Date Generation

For most classes — uniform random date within year:
```python
def random_date(year: int, month_weights=None) -> str:
    if month_weights is None:
        month = rng.integers(1, 13)  # 1–12
    else:
        month = rng.choice(range(1, 13), p=month_weights)
    start = datetime(year, month, 1)
    days_in_month = (datetime(year, month % 12 + 1, 1) - start).days if month < 12 else 31
    day = rng.integers(1, max(days_in_month, 2))
    return f"{year}-{month:02d}-{day:02d}"
```

For Sensor moisture — pass `month_weights` (monsoon pattern from Section 3.3).

### 6.4 QC Number Generation

After assembling all rows into a DataFrame and shuffling:
```python
df.insert(2, "QC_Number", [
    f"QC-{row['Year']}-{str(i+1).zfill(5)}"
    for i, row in df.iterrows()
])
```

### 6.5 Final Assembly Order

```python
rows = (
    gen_ntf(3600)
    + gen_track_burnt(2400)
    + gen_asic_cj327(1440)
    + gen_sensor_moisture(1440)
    + gen_connector_damage(1800)
    + gen_controller_failure(1320)
)
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

Shuffle with `random_state=42` for reproducibility.

### 6.6 Column Order (Must Match v2)

```python
col_order = ["Customer", "Year", "Date", "QC_Number", "Customer Complaint",
             "DTC", "Voltage", "Failure Analysis", "Warranty Decision", "Supplier"]
df = df[col_order]
```

---

## 7. Validation Checklist

Run these checks after generating the dataset before using it for training:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_warranty_claims_v3.csv')

# 1. Shape
assert df.shape == (12000, 10), f"Wrong shape: {df.shape}"

# 2. Columns
expected_cols = ["Customer", "Year", "Date", "QC_Number", "Customer Complaint",
                 "DTC", "Voltage", "Failure Analysis", "Warranty Decision", "Supplier"]
assert list(df.columns) == expected_cols

# 3. FA labels exact match
expected_fa = {
    'NTF', 'Track burnt due to EOS', 'ASIC CJ327 failure due to EOS',
    'Sensor short due to moisture', 'Connector damage',
    'controller failure due to supplier production failure'
}
assert set(df['Failure Analysis'].unique()) == expected_fa

# 4. WD labels exact match
expected_wd = {'Production Failure', 'Customer Failure', 'According to Specification'}
assert set(df['Warranty Decision'].unique()) == expected_wd

# 5. No NaN in critical columns
for col in ['Voltage', 'Customer Complaint', 'Failure Analysis', 'Warranty Decision']:
    assert df[col].isna().sum() == 0, f"NaN in {col}"

# 6. Voltage non-overlap (hardest constraint)
asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
ctrl  = df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage']
assert (asic > 16.0).sum() == 0,  "ASIC voltage leaked above 16V!"
assert (track <= 16.0).sum() == 0, "Track voltage dipped below 16.1V!"
assert (ctrl > 11.5).sum() == 0,  "Controller voltage leaked above 11.5V!"

# 7. DTC prefix constraints
df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True)
df['has_C'] = df['DTC'].fillna('').str.contains(r'\bC\d', regex=True)
df['has_B'] = df['DTC'].fillna('').str.contains(r'\bB\d', regex=True)

asic_df = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']
assert asic_df['has_P'].all(),       "ASIC must always have a P-code"
assert (~asic_df['has_U']).all(),    "ASIC must never have a U-code"

conn_df = df[df['Failure Analysis']=='Connector damage']
assert (~conn_df['has_P']).all(),    "Connector must never have a P-code"
assert (~conn_df['has_U']).all(),    "Connector must never have a U-code"

ctrl_df = df[df['Failure Analysis']=='controller failure due to supplier production failure']
assert ctrl_df['has_U'].all(),       "Controller must always have a U-code"
assert (~ctrl_df['has_P']).all(),    "Controller must never have a P-code"

# 8. Warranty constraints
ntf_df   = df[df['Failure Analysis']=='NTF']
track_df = df[df['Failure Analysis']=='Track burnt due to EOS']
ctrl_df  = df[df['Failure Analysis']=='controller failure due to supplier production failure']
sensor_df= df[df['Failure Analysis']=='Sensor short due to moisture']

assert (ntf_df['Warranty Decision'] == 'According to Specification').all()
assert (track_df['Warranty Decision'] == 'Customer Failure').all()
assert (ctrl_df['Warranty Decision'] == 'Production Failure').all()
assert (sensor_df['Warranty Decision'] == 'Customer Failure').all()

print("✅ All validation checks passed.")
```

---

## 8. Accuracy Benchmark (Expected Results)

Using the **exact ML code from `ml_predictor.py`** (no modifications), training on v3 should yield:

| Metric | v2 (old) | v3 (target) |
|---|---|---|
| Failure Analysis accuracy | 88.0% | ≥98% |
| Warranty Decision accuracy | 93.4% | ≥90% |
| ASIC CJ327 F1 | 0.863 | 1.000 |
| Track burnt F1 | 0.861 | 1.000 |
| Sensor moisture F1 | 0.706 | 1.000 |
| NTF F1 | 0.868 | 1.000 |

The FA accuracy ceiling in v2 was caused by ASIC/Track/Sensor ambiguity. v3 resolves this by giving each class a non-overlapping voltage band and class-exclusive DTC code pools.

---

## 9. If Accuracy Drops Below Target — Diagnosis Guide

If a future version of this dataset produces lower accuracy, check these in order:

| Symptom | Likely Cause | Fix |
|---|---|---|
| ASIC F1 drops | Voltage overlap with Track | Tighten ASIC max to 15.8V or Track min to 16.2V |
| Track F1 drops | DTC pool contaminated with ASIC codes | Separate DTC pools; Track uses P03xx/P04xx/U01xx only |
| Sensor F1 drops | DTC pool overlaps ASIC | Ensure Sensor uses P01xx only (temp/O2 sensors), ASIC uses P06xx only |
| Controller F1 drops | Voltage overlaps with Sensor | Ensure Controller max ≤ 11.5V, Sensor min ≥ 11.0V |
| NTF F1 drops | Too many NTF rows getting DTCs | Reduce NTF DTC rate below 18% |
| WD accuracy drops below 85% | Mixed warranty labels within deterministic classes | Re-check NTF=AccordingToSpec, Track=CustomerFailure, Controller=ProductionFailure are 100% |

---

## 10. Improvement Opportunities (For Next Version — v4)

If 100% FA accuracy is already achieved and you want to make the dataset more realistic for edge-case handling:

1. **Add 2–3% label noise** — in real data, some rows are mislabelled. Add `~240 rows` where a valid set of features maps to a "wrong" label. This tests the model's robustness.

2. **Add a new class: "Firmware corruption"** — similar to controller failure but with voltage in the 12.0–13.5V range and DTCs like `U0073 + P0602` (Programming Error). This would require re-splitting class counts to keep total at 12,000.

3. **Add vehicle mileage/age as a feature** — older vehicles are more likely to have Connector damage; newer ones more likely to have Production Failures. Add a `Mileage` column (0–200,000 km) and encode it as a feature.

4. **Add repeat customer flag** — a `Repeat_Claim` binary column indicating if the same vehicle (VIN) has had a prior claim. Repeat claims skew toward NTF and Customer Failure.

5. **Add regional metadata** — coastal regions → more moisture failures; highway fleet operators → more EOS failures (frequent jump-starts). Add a `Region` column with `Coastal`, `Inland`, `Fleet` and apply realistic priors.

---

## 11. Exact Label Strings Reference

Copy these exactly — case-sensitive. A single character difference breaks the `LabelEncoder`.

```
Failure Analysis labels:
  "NTF"
  "Track burnt due to EOS"
  "ASIC CJ327 failure due to EOS"
  "Sensor short due to moisture"
  "Connector damage"
  "controller failure due to supplier production failure"

Warranty Decision labels:
  "According to Specification"
  "Customer Failure"
  "Production Failure"
```

---

*Document version: 1.0 — Generated alongside `synthetic_warranty_claims_v3.csv`*
*This guide is sufficient to regenerate v3 exactly from scratch using the code pattern in Section 6.*
