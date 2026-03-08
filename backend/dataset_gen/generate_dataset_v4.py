"""
Realistic Warranty Claims Dataset Generator — v4
=================================================
Changes from v3:
  [SCALE]
  - Dataset size: 12,000 → 50,000 rows
  - Proportions kept identical (NTF 30%, Track 20%, Connector 15%, etc.)

  [REALISM — what makes real user inputs work better]
  1. Expanded DTC pools per class (more code variety → TF-IDF learns concepts,
     not just specific tokens; model generalises to unseen-but-similar codes)
  2. Temporal weighting — 2022–2024 have ~3× more claims than 2019 (realistic
     warranty centre growth; avoids year as an unintentional discriminating feature)
  3. ASIC warranty decision correlated with voltage — higher voltage within the
     14.2–16.0V ASIC band → higher probability of Customer Failure
  4. Connector damage: 6% of cases add a secondary P-code (mixed harness/powertrain
     fault; real connectors sometimes cause P-codes via broken sensor circuits)
  5. Sensor moisture: 4% of cases have no DTC stored (code cleared before workshop;
     real-world scenario where technician notes are the only signal)
  6. NTF: voltage ceiling raised slightly to 14.6V (some NTF units had a brief
     high-alt event but no damage; tests the model's voltage boundary)
  7. Expanded NTF DTC pool — 8 intermittent codes instead of 3
  8. Mileage column added (0–250,000 km) — not used by current ML but makes
     dataset more realistic and future-ready
  9. ~1.5% warranty decision label noise (~750 rows) — realistic technician
     mislabelling at the Production Failure / Customer Failure boundary
 10. Voltage precision increased to 2 decimal places (real DVOM readings)
 11. Fixed date generator (previous version could skip last day of month)
 12. Year distribution weighted toward recent years
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random

rng = np.random.default_rng(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Domain constants
# ─────────────────────────────────────────────────────────────────────────────

CUSTOMERS = ["Ashok Leyland", "M&M", "Honda", "Kia", "Toyota", "Hyundai", "TATA"]
SUPPLIERS  = ["Hanon", "Bosch", "Valeo", "Delphi", "STM"]
YEARS      = list(range(2019, 2025))

# Temporal weighting — claims grow over time (warranty centre ramp-up + fleet growth)
# 2019: low volume, 2024: peak
YEAR_WEIGHTS = [0.08, 0.10, 0.14, 0.18, 0.22, 0.28]  # sums to 1.0

# ─────────────────────────────────────────────────────────────────────────────
# DTC pools — EXPANDED from v3 for better TF-IDF generalisation
# ─────────────────────────────────────────────────────────────────────────────

# ASIC CJ327: P06xx = Internal ECM faults, P05xx = system voltage faults
DTC_ASIC = [
    "P0601",   # Internal Control Module Memory Check Sum Error
    "P0602",   # Control Module Programming Error
    "P0604",   # Internal Control Module RAM Error
    "P0605",   # Internal Control Module Read Only Memory Error
    "P0606",   # ECM/PCM Processor Fault
    "P0607",   # Control Module Performance
    "P0608",   # Control Module VSS Output "A" Malfunction
    "P0610",   # Control Module Vehicle Options Error
    "P0562",   # System Voltage Low (ASIC brownout)
    "P0563",   # System Voltage High (voltage threshold hit)
    "P0611",   # Fuel Injector Control Module Performance
    "P0613",   # TCM Processor Fault
]

# Track burnt: high-voltage event causes multi-system simultaneous failure
DTC_TRACK = [
    "P0562",   # System Voltage Low (remnant as circuits drop)
    "P0563",   # System Voltage High
    "P0300",   # Random/Multiple Cylinder Misfire
    "P0301",   # Cylinder 1 Misfire
    "P0302",   # Cylinder 2 Misfire
    "P0303",   # Cylinder 3 Misfire
    "P0304",   # Cylinder 4 Misfire
    "P0480",   # Cooling Fan Relay 1 Control Circuit
    "P0481",   # Cooling Fan Relay 2 Control Circuit
    "P0482",   # Cooling Fan Relay 3 Control Circuit
    "P1682",   # Ignition Switch Circuit
    "P0615",   # Starter Relay Circuit
    "P0620",   # Generator Control Circuit Malfunction
    "U0001",   # High Speed CAN Communication Bus
    "U0100",   # Lost Communication with ECM/PCM
    "U0101",   # Lost Communication with TCM
    "U0155",   # Lost Communication with Instrument Panel Cluster
]

# Sensor moisture: P01xx = temperature/pressure sensor circuits
DTC_SENSOR_MOISTURE = [
    "P0113",   # Intake Air Temperature Sensor Circuit High
    "P0112",   # Intake Air Temperature Sensor Circuit Low
    "P0118",   # Engine Coolant Temperature Sensor Circuit High
    "P0117",   # Engine Coolant Temperature Sensor Circuit Low
    "P0128",   # Coolant Temperature Below Thermostat Regulating Temperature
    "P0197",   # Engine Oil Temperature Sensor Circuit Low
    "P0196",   # Engine Oil Temperature Sensor Range/Performance
    "P0072",   # Ambient Air Temperature Sensor Circuit Low
    "P0073",   # Ambient Air Temperature Sensor Circuit High
    "P0038",   # HO2S Heater Control Circuit High Bank 1 Sensor 2
    "P0054",   # HO2S Heater Resistance Bank 1 Sensor 2
    "P0131",   # O2 Sensor Circuit Low Voltage Bank 1 Sensor 1
    "P0135",   # O2 Sensor Heater Circuit Bank 1 Sensor 1
    "P0069",   # MAP/BARO Pressure Correlation
]

# Connector damage: exclusively C/B codes (chassis/body)
DTC_CONNECTOR = [
    "C0031",   # Right Front Wheel Speed Sensor
    "C0265",   # EBCM Relay Circuit
    "C0045",   # Left Rear Wheel Speed Sensor
    "C0036",   # Right Rear Wheel Speed Sensor
    "C0051",   # Steering Wheel Position Sensor Circuit
    "C0460",   # Steering Angle Sensor
    "C0550",   # ECU Internal Failure
    "B2AAA",   # Body Control General
    "B1234",   # Interior Lighting Circuit
    "B1031",   # Door Lock Actuator
    "B1045",   # Driver Power Seat Circuit
    "B2960",   # Key In Ignition Switch Circuit
    "B3055",   # Fuel Pump Relay Control Circuit
]

# Secondary P-codes that can appear in connector damage (6% mixed fault scenario)
# Real: broken harness → intermittent sensor signal → P-code logged alongside C/B code
DTC_CONNECTOR_SECONDARY_P = [
    "P0340",   # Camshaft Position Sensor Circuit Malfunction
    "P0325",   # Knock Sensor Circuit Malfunction
    "P0500",   # Vehicle Speed Sensor Malfunction
    "P0720",   # Output Shaft Speed Sensor Circuit Malfunction
]

# Controller failure: U-codes (CAN/LIN communication loss)
DTC_CONTROLLER = [
    "U0073",   # Control Module Communication Bus Off
    "U0100",   # Lost Communication with ECM/PCM "A"
    "U0101",   # Lost Communication with TCM
    "U0103",   # Lost Communication with Gear Shift Control Module
    "U0121",   # Lost Communication with ABS Control Module
    "U0122",   # Lost Communication with Vehicle Dynamics Control Module
    "U0131",   # Lost Communication with Power Steering Control Module
    "U0140",   # Lost Communication with Body Control Module
    "U0155",   # Lost Communication with Instrument Panel Cluster
    "U0001",   # High Speed CAN Communication Bus
    "U0164",   # Lost Communication with HVAC Control Module
    "U0184",   # Lost Communication with Radio
]

# NTF: expanded pool of mild/intermittent P-codes
DTC_NTF_MILD = [
    "",        # No DTC stored (most common)
    "P0455",   # Evaporative Emission System Leak (large) — often false
    "P0456",   # Evaporative Emission System Leak (small) — often false
    "P0171",   # System Too Lean — intermittent sensor drift
    "P0174",   # System Too Lean Bank 2
    "P0300",   # Random Misfire — single occurrence, self-cleared
    "P0316",   # Misfire Detected on Startup — clears at warmup
    "P0420",   # Catalyst System Efficiency Below Threshold — borderline cat
    "P0430",   # Catalyst System Efficiency Below Threshold Bank 2
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def pick_year() -> int:
    """Pick year with temporal weighting — more recent years have more claims."""
    return int(rng.choice(YEARS, p=YEAR_WEIGHTS))


def random_date(year: int, month_weights=None) -> str:
    """Generate a random date. Fixed: correctly handles all months including December."""
    if month_weights is None:
        month = int(rng.integers(1, 13))
    else:
        month = int(rng.choice(range(1, 13), p=month_weights))

    # Compute days in month correctly
    if month == 12:
        days_in_month = 31
    else:
        days_in_month = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

    day = int(rng.integers(1, days_in_month + 1))  # inclusive upper bound now
    return f"{year}-{month:02d}-{day:02d}"


def gen_mileage_km(fa_class: str) -> int:
    """
    Generate realistic mileage for a claim based on failure class.
    
    - Controller failure (production defect): typically early life, low mileage
    - ASIC EOS: can happen at any mileage but peaks at 30–80k (alternator wear)
    - Track burnt (EOS): sudden event, any mileage
    - Connector damage: higher mileage (corrosion builds over time) 
    - Sensor moisture: any mileage (environment-driven)
    - NTF: any mileage
    """
    if fa_class == "controller failure due to supplier production failure":
        km = rng.normal(22000, 18000)
        return int(np.clip(km, 500, 80000))
    elif fa_class == "ASIC CJ327 failure due to EOS":
        km = rng.normal(55000, 30000)
        return int(np.clip(km, 5000, 180000))
    elif fa_class == "Track burnt due to EOS":
        km = rng.normal(70000, 40000)
        return int(np.clip(km, 1000, 220000))
    elif fa_class == "Connector damage":
        km = rng.normal(85000, 35000)
        return int(np.clip(km, 15000, 230000))
    elif fa_class == "Sensor short due to moisture":
        km = rng.normal(60000, 35000)
        return int(np.clip(km, 2000, 200000))
    else:  # NTF
        km = rng.normal(50000, 35000)
        return int(np.clip(km, 500, 220000))


# ─────────────────────────────────────────────────────────────────────────────
# Failure-mode generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_asic_cj327(n: int) -> list[dict]:
    """
    ASIC CJ327 failure due to EOS.
    
    Voltage: 14.2–16.0V (soft overvoltage damages ASIC die without burning traces)
    DTC: P06xx ASIC-specific internal control module faults, 1–2 codes
    
    v4 change: Warranty decision is now voltage-correlated.
      - Low end of ASIC band (14.2–14.9V): supplier defect more likely (65% PF)
      - High end of ASIC band (15.0–16.0V): customer's alternator pushed it there (55% CF)
    """
    rows = []
    complaints = ["Starting Problem", "Low pickup", "High fuel consumption",
                  "Rough idling", "Engine jerking during acceleration"]
    complaint_weights = [0.25, 0.25, 0.20, 0.18, 0.12]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(15.3, 0.45), 14.2, 16.0))

        dtc_count = 1 if rng.random() < 0.88 else 2
        primary = str(rng.choice(DTC_ASIC[:10]))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_ASIC))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS, p=[0.08, 0.12, 0.08, 0.32, 0.40]))
        customer  = str(rng.choice(CUSTOMERS, p=[0.20, 0.15, 0.12, 0.12, 0.13, 0.13, 0.15]))

        # Voltage-correlated warranty: high voltage → more likely customer's alternator fault
        if voltage >= 15.0:
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.42, 0.58]))
        else:
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.72, 0.28]))

        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "ASIC CJ327 failure due to EOS",
            "Warranty Decision": warranty, "Supplier": supplier,
            "Mileage_km": gen_mileage_km("ASIC CJ327 failure due to EOS"),
        })
    return rows


def gen_track_burnt(n: int) -> list[dict]:
    """
    Track burnt due to EOS.
    
    Voltage: 16.1–20.0V (catastrophic PCB trace burn from sustained overvoltage)
    DTC: 2–5 codes — mixed voltage, misfire, fan relay, CAN codes (systemic failure)
    Warranty: 100% Customer Failure
    """
    rows = []
    complaints = ["Vehicle not starting", "Engine overheating",
                  "Starting Problem", "Engine jerking during acceleration", "Low pickup"]
    complaint_weights = [0.30, 0.28, 0.22, 0.12, 0.08]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(17.8, 1.1), 16.1, 20.0))

        dtc_count = int(rng.choice([2, 3, 4, 5], p=[0.35, 0.35, 0.20, 0.10]))
        dtc_codes = list(rng.choice(DTC_TRACK, size=min(dtc_count, len(DTC_TRACK)), replace=False))
        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "Track burnt due to EOS",
            "Warranty Decision": "Customer Failure", "Supplier": supplier,
            "Mileage_km": gen_mileage_km("Track burnt due to EOS"),
        })
    return rows


def gen_sensor_moisture(n: int) -> list[dict]:
    """
    Sensor short due to moisture.
    
    Voltage: 11.0–13.5V (sensor load slightly depresses rail)
    DTC: P01xx/P00xx sensor-specific codes, 75% single / 25% two codes
    Season: weighted Jun–Sep (Indian monsoon)
    Warranty: 100% Customer Failure (environmental damage)
    
    v4 change: 4% of cases have no DTC stored (cleared pre-inspection)
    """
    rows = []
    complaints = ["Engine overheating", "Rough idling", "OBD Light ON",
                  "Starting Problem", "High fuel consumption"]
    complaint_weights = [0.38, 0.28, 0.18, 0.10, 0.06]
    month_weights = [0.04, 0.04, 0.05, 0.05, 0.06, 0.15, 0.18, 0.16, 0.12, 0.06, 0.05, 0.04]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(12.7, 0.55), 11.0, 13.5))

        # 4% chance DTC was cleared before workshop saw the vehicle
        if rng.random() < 0.04:
            dtc = ""
        else:
            dtc_count = 1 if rng.random() < 0.75 else 2
            primary = str(rng.choice(DTC_SENSOR_MOISTURE))
            if dtc_count == 2:
                secondary = str(rng.choice(DTC_SENSOR_MOISTURE))
                dtc = f"{primary}, {secondary}" if secondary != primary else primary
            else:
                dtc = primary

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))
        customer  = str(rng.choice(CUSTOMERS, p=[0.12, 0.15, 0.18, 0.14, 0.14, 0.15, 0.12]))

        rows.append({
            "Customer": customer, "Year": year,
            "Date": random_date(year, month_weights),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "Sensor short due to moisture",
            "Warranty Decision": "Customer Failure", "Supplier": supplier,
            "Mileage_km": gen_mileage_km("Sensor short due to moisture"),
        })
    return rows


def gen_ntf(n: int) -> list[dict]:
    """
    NTF — No Trouble Found.
    
    Voltage: 12.5–14.6V (normal operating range; v4 raises ceiling by 0.2V
             to represent cases with brief high-alt event but no trace damage)
    DTC: 82% empty; 18% mild intermittent code from expanded 8-code pool
    Warranty: 100% According to Specification
    """
    rows = []
    complaints = ["OBD Light ON", "Engine overheating", "Rough idling",
                  "Low pickup", "High fuel consumption"]
    complaint_weights = [0.78, 0.08, 0.07, 0.04, 0.03]

    for _ in range(n):
        year = pick_year()
        # Slightly raised ceiling vs v3 (14.4→14.6V) for realism
        voltage = float(np.clip(rng.normal(13.2, 0.55), 12.5, 14.6))

        if rng.random() < 0.82:
            dtc = ""
        else:
            dtc = str(rng.choice([c for c in DTC_NTF_MILD if c]))  # exclude empty

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "NTF",
            "Warranty Decision": "According to Specification", "Supplier": supplier,
            "Mileage_km": gen_mileage_km("NTF"),
        })
    return rows


def gen_connector_damage(n: int) -> list[dict]:
    """
    Connector damage.
    
    Voltage: 12.5–14.5V (normal)
    DTC: C/B-codes exclusively; 65% 1-code, 28% 2-codes, 7% 3-codes
    Warranty: 80% Production Failure (crimp defect), 20% Customer Failure
    
    v4 change: 6% of cases have an additional secondary P-code logged
    (broken harness intermittently opens a sensor circuit → P-code fires too).
    This is a realistic mixed-fault scenario.
    """
    rows = []
    complaints = ["Brake warning light ON", "OBD Light ON", "Starting Problem",
                  "Engine overheating", "Low pickup"]
    complaint_weights = [0.72, 0.15, 0.07, 0.04, 0.02]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(13.3, 0.65), 12.5, 14.5))

        dtc_count = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        dtc_codes = list(rng.choice(DTC_CONNECTOR, size=min(dtc_count, len(DTC_CONNECTOR)), replace=False))

        # 6% chance: secondary P-code from broken sensor circuit
        if rng.random() < 0.06:
            secondary_p = str(rng.choice(DTC_CONNECTOR_SECONDARY_P))
            dtc_codes.append(secondary_p)

        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS, p=[0.15, 0.30, 0.28, 0.18, 0.09]))
        customer  = str(rng.choice(CUSTOMERS))
        warranty  = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.80, 0.20]))

        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "Connector damage",
            "Warranty Decision": warranty, "Supplier": supplier,
            "Mileage_km": gen_mileage_km("Connector damage"),
        })
    return rows


def gen_controller_failure(n: int) -> list[dict]:
    """
    Controller failure due to supplier production failure.
    
    Voltage: 9.0–11.5V (under-voltage from failing internal regulator)
    DTC: U-codes only (CAN/LIN communication loss), 1–2 codes
    Warranty: 100% Production Failure
    """
    rows = []
    complaints = ["Vehicle not starting", "Starting Problem",
                  "Engine jerking during acceleration", "Low pickup"]
    complaint_weights = [0.42, 0.34, 0.16, 0.08]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(10.4, 0.60), 9.0, 11.5))

        dtc_count = 1 if rng.random() < 0.82 else 2
        primary = str(rng.choice(DTC_CONTROLLER))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_CONTROLLER))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS, p=[0.05, 0.35, 0.08, 0.25, 0.27]))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer, "Year": year, "Date": random_date(year),
            "Voltage": round(voltage, 2),
            "DTC": dtc, "Customer Complaint": complaint,
            "Failure Analysis": "controller failure due to supplier production failure",
            "Warranty Decision": "Production Failure", "Supplier": supplier,
            "Mileage_km": gen_mileage_km("controller failure due to supplier production failure"),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Label noise injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_warranty_label_noise(df: pd.DataFrame, noise_rate: float = 0.015) -> pd.DataFrame:
    """
    Add ~1.5% warranty decision label noise to simulate real-world technician
    mislabelling at the Production Failure / Customer Failure boundary.
    
    Only affects rows where ambiguity is physically plausible:
    - ASIC cases near the voltage boundary (14.8–15.2V): PF↔CF are both defensible
    - Connector damage: PF↔CF (difficult to prove customer abuse vs crimp defect)
    
    NTF, Track, Controller are never flipped (their labels are deterministic in
    real warranty centres — NTF=AccordingToSpec, Track=CustomerFailure are unambiguous).
    """
    df = df.copy()
    
    fa_col = "Failure Analysis"
    wd_col = "Warranty Decision"
    
    # ASIC: flip at voltage boundary zone
    asic_mask = (
        (df[fa_col] == "ASIC CJ327 failure due to EOS") &
        (df["Voltage"].between(14.8, 15.2)) &
        (df[wd_col].isin(["Production Failure", "Customer Failure"]))
    )
    asic_candidates = df[asic_mask].index
    n_asic_flip = int(len(asic_candidates) * noise_rate * 3)  # higher noise in boundary zone
    if n_asic_flip > 0:
        flip_idx = rng.choice(asic_candidates, size=min(n_asic_flip, len(asic_candidates)), replace=False)
        for idx in flip_idx:
            current = df.at[idx, wd_col]
            df.at[idx, wd_col] = "Customer Failure" if current == "Production Failure" else "Production Failure"

    # Connector damage: ~2% label flip (customer abuse vs manufacturing crimp)
    conn_mask = (
        (df[fa_col] == "Connector damage") &
        (df[wd_col].isin(["Production Failure", "Customer Failure"]))
    )
    conn_candidates = df[conn_mask].index
    n_conn_flip = int(len(conn_candidates) * noise_rate)
    if n_conn_flip > 0:
        flip_idx = rng.choice(conn_candidates, size=min(n_conn_flip, len(conn_candidates)), replace=False)
        for idx in flip_idx:
            current = df.at[idx, wd_col]
            df.at[idx, wd_col] = "Customer Failure" if current == "Production Failure" else "Production Failure"

    total_flipped = n_asic_flip + n_conn_flip
    print(f"  Label noise injected: {total_flipped} rows flipped "
          f"({total_flipped/len(df)*100:.2f}% of dataset)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

TARGET = 50_000

counts = {
    "ntf":        15_000,   # 30%
    "track":      10_000,   # 20%
    "connector":   7_500,   # 15%
    "asic":        6_000,   # 12%
    "moisture":    6_000,   # 12%
    "controller":  5_500,   # 11%
}
assert sum(counts.values()) == TARGET, f"Total mismatch: {sum(counts.values())}"

print("Generating rows...")
rows = (
    gen_ntf(counts["ntf"])
    + gen_track_burnt(counts["track"])
    + gen_asic_cj327(counts["asic"])
    + gen_sensor_moisture(counts["moisture"])
    + gen_connector_damage(counts["connector"])
    + gen_controller_failure(counts["controller"])
)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Inject label noise BEFORE generating QC numbers (so shuffled index doesn't matter)
print("\nInjecting label noise...")
df = inject_warranty_label_noise(df, noise_rate=0.015)

# Generate QC numbers
df.insert(2, "QC_Number", [
    f"QC-{row['Year']}-{str(i+1).zfill(6)}"
    for i, row in df.iterrows()
])

# Final column order — matches v3 + new Mileage_km column at end
col_order = [
    "Customer", "Year", "Date", "QC_Number", "Customer Complaint",
    "DTC", "Voltage", "Failure Analysis", "Warranty Decision", "Supplier",
    "Mileage_km",
]
df = df[col_order]

print(f"\nGenerated {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== FAILURE ANALYSIS DISTRIBUTION ===")
fa_counts = df['Failure Analysis'].value_counts()
print(fa_counts)
print(fa_counts / len(df) * 100)

print("\n=== WARRANTY DECISION DISTRIBUTION ===")
print(df['Warranty Decision'].value_counts())

print("\n=== VOLTAGE STATS BY FAILURE ANALYSIS ===")
print(df.groupby('Failure Analysis')['Voltage'].agg(['mean','std','min','max']).round(2))

print("\n=== YEAR DISTRIBUTION ===")
print(df['Year'].value_counts().sort_index())

print("\n=== MILEAGE STATS BY FAILURE ANALYSIS ===")
print(df.groupby('Failure Analysis')['Mileage_km'].agg(['mean','min','max']).round(0))

df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True).astype(int)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True).astype(int)
df['has_C'] = df['DTC'].fillna('').str.contains(r'\bC\d', regex=True).astype(int)
df['has_B'] = df['DTC'].fillna('').str.contains(r'\bB\d', regex=True).astype(int)
df['dtc_count'] = df['DTC'].fillna('').apply(
    lambda x: len([c for c in x.split(',') if c.strip()])
)
print("\n=== DTC FEATURE STATS BY FAILURE ANALYSIS ===")
print(df.groupby('Failure Analysis')[['has_P','has_U','has_C','has_B','dtc_count']].mean().round(3))

print("\n=== VOLTAGE SEPARATION: ASIC vs TRACK (must not overlap) ===")
asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
ctrl  = df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage']
print(f"ASIC  voltage: {asic.min():.2f}–{asic.max():.2f}V  mean={asic.mean():.2f}")
print(f"Track voltage: {track.min():.2f}–{track.max():.2f}V  mean={track.mean():.2f}")
print(f"Ctrl  voltage: {ctrl.min():.2f}–{ctrl.max():.2f}V  mean={ctrl.mean():.2f}")
print(f"ASIC overlap >16V:  {(asic > 16.0).sum()} rows")
print(f"Track overlap <=16V: {(track <= 16.0).sum()} rows")
print(f"Ctrl overlap >11.5V: {(ctrl > 11.5).sum()} rows")

print("\n=== CONNECTOR: rows with secondary P-code ===")
conn_df = df[df['Failure Analysis'] == 'Connector damage']
has_p_conn = conn_df['has_P'].sum()
print(f"  {has_p_conn} / {len(conn_df)} connector rows have a P-code ({has_p_conn/len(conn_df)*100:.1f}%)")

print("\n=== SENSOR MOISTURE: rows with no DTC ===")
sensor_df = df[df['Failure Analysis'] == 'Sensor short due to moisture']
no_dtc = (sensor_df['DTC'].fillna('') == '').sum()
print(f"  {no_dtc} / {len(sensor_df)} sensor rows have no DTC ({no_dtc/len(sensor_df)*100:.1f}%)")

# Remove helper columns before saving
df.drop(columns=['has_P','has_U','has_C','has_B','dtc_count'], inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# Validation assertions
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== RUNNING VALIDATION ASSERTIONS ===")

assert df.shape[0] == TARGET, f"Wrong row count: {df.shape[0]}"
assert df.shape[1] == 11, f"Wrong column count: {df.shape[1]}"

expected_fa = {
    'NTF', 'Track burnt due to EOS', 'ASIC CJ327 failure due to EOS',
    'Sensor short due to moisture', 'Connector damage',
    'controller failure due to supplier production failure'
}
assert set(df['Failure Analysis'].unique()) == expected_fa

expected_wd = {'Production Failure', 'Customer Failure', 'According to Specification'}
assert set(df['Warranty Decision'].unique()) == expected_wd

for col in ['Voltage', 'Customer Complaint', 'Failure Analysis', 'Warranty Decision']:
    assert df[col].isna().sum() == 0, f"NaN in {col}"

# Voltage hard boundaries (must never overlap)
asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
ctrl  = df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage']
assert (asic  > 16.0).sum() == 0,  "ASIC voltage leaked above 16V!"
assert (track <= 16.0).sum() == 0,  "Track voltage dipped below 16.1V!"
assert (ctrl  > 11.5).sum() == 0,  "Controller voltage leaked above 11.5V!"

# DTC prefix constraints
df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True)

asic_df = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']
assert asic_df['has_P'].all(),        "ASIC must always have a P-code"
assert (~asic_df['has_U']).all(),     "ASIC must never have a U-code"

ctrl_df = df[df['Failure Analysis']=='controller failure due to supplier production failure']
assert ctrl_df['has_U'].all(),        "Controller must always have a U-code"
assert (~ctrl_df['has_P']).all(),     "Controller must never have a P-code"

# Warranty constraints for deterministic classes
ntf_df   = df[df['Failure Analysis']=='NTF']
track_df = df[df['Failure Analysis']=='Track burnt due to EOS']

assert (ntf_df['Warranty Decision']   == 'According to Specification').all()
assert (track_df['Warranty Decision'] == 'Customer Failure').all()
assert (ctrl_df['Warranty Decision']  == 'Production Failure').all()

df.drop(columns=['has_P', 'has_U'], inplace=True)

print("✅ All validation assertions passed.")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

out_path = "/mnt/user-data/outputs/synthetic_warranty_claims_v4.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Saved to {out_path}")
print(f"   Shape: {df.shape}")
