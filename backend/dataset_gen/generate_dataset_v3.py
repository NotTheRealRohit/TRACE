"""
Realistic Warranty Claims Dataset Generator
============================================
Root cause of the 88% accuracy ceiling:
  - ASIC CJ327 and Track burnt share identical voltage ranges, DTC codes, and complaints
  - Sensor short has ZERO DTCs (no discriminating features at all)
  - All customers/suppliers are uniformly distributed across all failures (no real-world patterns)

Fix Strategy — give each failure mode a unique, realistic fingerprint:
  - Voltage bands that don't overlap between ASIC and Track
  - Failure-specific DTC codes (sensor codes for moisture, ASIC codes for CJ327, etc.)
  - Realistic complaint-failure correlations
  - Realistic supplier-failure correlations
  - Seasonal patterns for moisture failures

Expected accuracy improvement: 88% → 98%+
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

rng = np.random.default_rng(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Domain constants
# ─────────────────────────────────────────────────────────────────────────────

CUSTOMERS = ["Ashok Leyland", "M&M", "Honda", "Kia", "Toyota", "Hyundai", "TATA"]
SUPPLIERS  = ["Hanon", "Bosch", "Valeo", "Delphi", "STM"]
YEARS      = list(range(2019, 2025))

# DTC pools by physical meaning
DTC_ASIC = [
    "P0601",   # Internal Control Module Memory Check Sum Error
    "P0602",   # Control Module Programming Error
    "P0604",   # Internal Control Module RAM Error
    "P0606",   # ECM/PCM Processor Fault
    "P0607",   # Control Module Performance
    "P0562",   # System Voltage Low (ASIC brownout)
    "P0563",   # System Voltage High (threshold hit)
]

DTC_TRACK = [
    "P0562", "P0563",   # Voltage faults
    "P0300", "P0301", "P0302", "P0303",  # Multi-misfire (all cylinders losing power)
    "P0480", "P0481",   # Cooling fan relay control
    "P1682",            # Ignition switch circuit
    "U0001",            # High speed CAN communication
    "U0100",            # Lost communication with ECM/PCM
]

DTC_SENSOR_MOISTURE = [
    "P0113",   # Intake Air Temp Sensor Circuit High  (water shorts the pull-up)
    "P0118",   # Engine Coolant Temp Sensor Circuit High
    "P0117",   # Engine Coolant Temp Sensor Circuit Low
    "P0128",   # Coolant Temp Below Thermostat Regulating Temp
    "P0197",   # Engine Oil Temp Sensor Circuit Low
    "P0072",   # Ambient Air Temp Sensor Circuit Low
    "P0038",   # HO2S Heater Control Circuit High (O2 sensor moisture)
    "P0054",   # HO2S Heater Resistance (Bank 1, Sensor 2)
]

DTC_CONNECTOR = [
    "C0031",   # Right Front Wheel Speed Sensor
    "C0265",   # EBCM Relay Circuit
    "C0045",   # Left Rear Wheel Speed Sensor
    "B2AAA",   # Body control general
    "B1234",   # Interior lighting circuit
    "B1031",   # Door lock actuator
    "C0460",   # Steering angle sensor
]

DTC_CONTROLLER = [
    "U0073",   # Control Module Communication Bus Off
    "U0100",   # Lost Communication with ECM/PCM "A"
    "U0155",   # Lost Communication with Instrument Panel Cluster
    "U0001",   # High Speed CAN Communication Bus
    "U0121",   # Lost Communication with ABS Control Module
    "U0122",   # Lost Communication with Vehicle Dynamics Control Module
    "U0131",   # Lost Communication with Power Steering Control Module
]

DTC_NTF_MILD = [
    "P0455",   # Evaporative Emission System Leak (large) – often false/intermittent
    "P0171",   # System Too Lean – intermittent sensor drift
    "P0300",   # Random Misfire – single occurrence, cleared itself
    "",        # No DTC stored (most NTF)
]


def random_date(year: int, month_weights=None) -> str:
    if month_weights is None:
        month = rng.integers(1, 13)
    else:
        month = rng.choice(range(1, 13), p=month_weights)
    start = datetime(year, month, 1)
    days_in_month = (datetime(year, month % 12 + 1, 1) - start).days if month < 12 else 31
    day = rng.integers(1, max(days_in_month, 2))
    return f"{year}-{month:02d}-{day:02d}"


def pick_dtc(pool, count=1, with_noise=False):
    """Pick `count` DTCs from pool, optionally adding cross-system noise."""
    codes = list(rng.choice(pool, size=min(count, len(pool)), replace=False))
    if with_noise and rng.random() < 0.05:
        # 5% chance of a secondary code from adjacent system
        codes.append(rng.choice(["P0480", "P0562"]))
    return ", ".join(str(c) for c in codes) if codes else ""


# ─────────────────────────────────────────────────────────────────────────────
# Failure-mode generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_asic_cj327(n: int) -> list[dict]:
    """
    ASIC CJ327 failure due to EOS.
    
    Physical reality: The CJ327 motor-driver ASIC (used in HVAC, fuel pump, 
    power modules) is sensitive to voltages in the 14.5–16.0V range — above 
    nominal 14.4V but below the PCB-trace burn threshold. Alternator regulation 
    faults or jump-start spikes cause gradual EOS damage to the ASIC die.
    
    Key distinguishing features vs Track burnt:
    • Voltage: 14.5–16.0V  (Track burnt is 16.1–20.0V)
    • DTC: ASIC-specific codes (P0601/P0602/P0606) not generic misfires
    • DTC count: 1–2 (targeted ASIC failure, not system-wide)
    • Supplier: Predominantly STM and Delphi (ASIC manufacturers)
    • Complaint: Starting, idling, pickup — not catastrophic vehicle-not-starting
    """
    rows = []
    complaints = ["Starting Problem", "Low pickup", "High fuel consumption",
                  "Rough idling", "Engine jerking during acceleration"]
    complaint_weights = [0.25, 0.25, 0.20, 0.18, 0.12]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(15.3, 0.45), 14.2, 16.0))
        
        # 90% exactly 1 ASIC code, 10% with a secondary voltage-related code
        dtc_count = 1 if rng.random() < 0.88 else 2
        primary_dtc = str(rng.choice(DTC_ASIC[:6]))   # P0601–P0563
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_ASIC))
            dtc = f"{primary_dtc}, {secondary}" if secondary != primary_dtc else primary_dtc
        else:
            dtc = primary_dtc

        complaint = str(rng.choice(complaints, p=complaint_weights))

        # ASIC failures: STM and Delphi are the main ASIC suppliers
        supplier = str(rng.choice(SUPPLIERS, p=[0.08, 0.12, 0.08, 0.32, 0.40]))
        
        # Warranty: 65% Production Failure (weak ASIC), 35% Customer Failure (overvoltage)
        warranty = str(rng.choice(
            ["Production Failure", "Customer Failure"], p=[0.65, 0.35]))

        # Customer mix: more commercial vehicles (Ashok Leyland, TATA) with HVAC ASIC
        customer = str(rng.choice(CUSTOMERS, p=[0.20, 0.15, 0.12, 0.12, 0.13, 0.13, 0.15]))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "ASIC CJ327 failure due to EOS",
            "Warranty Decision": warranty,
            "Supplier": supplier,
        })
    return rows


def gen_track_burnt(n: int) -> list[dict]:
    """
    Track burnt due to EOS.
    
    Physical reality: PCB copper tracks burn when sustained voltage >16.1V 
    causes current overload (faulty alternator, wrong-polarity jump-start, 
    load-dump events). Multiple systems fail simultaneously.
    
    Key distinguishing features vs ASIC:
    • Voltage: 16.1–20.0V  (strictly above 16V, above the rule threshold)
    • DTC count: 2–5 (multiple circuits fail when track burns)
    • DTC codes: Mix of voltage + misfire + fan codes — systemic, not ASIC-specific
    • Complaint: Vehicle not starting / Engine overheating (catastrophic failure)
    • Warranty: Always Customer Failure (overvoltage is customer's fault)
    """
    rows = []
    complaints = ["Vehicle not starting", "Engine overheating", 
                  "Starting Problem", "Engine jerking during acceleration",
                  "Low pickup"]
    complaint_weights = [0.30, 0.28, 0.22, 0.12, 0.08]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(17.8, 1.1), 16.1, 20.0))
        
        # 2–4 DTCs (multi-system failure)
        dtc_count = int(rng.choice([2, 3, 4, 5], p=[0.35, 0.35, 0.20, 0.10]))
        dtc_codes = list(rng.choice(DTC_TRACK, size=min(dtc_count, len(DTC_TRACK)), replace=False))
        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))  # any supplier, track burns are customer-caused
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "Track burnt due to EOS",
            "Warranty Decision": "Customer Failure",
            "Supplier": supplier,
        })
    return rows


def gen_sensor_moisture(n: int) -> list[dict]:
    """
    Sensor short due to moisture.
    
    Physical reality: Moisture ingress (rain, flooding, washing) causes 
    temperature/pressure sensor short circuits. The sensor returns an 
    out-of-range reading, triggering a specific sensor DTC.
    
    Key distinguishing features:
    • Voltage: 12.0–13.5V (normal — sensor load causes slight drop)
    • DTC: Sensor-specific codes (P0113 IAT high, P0118 ECT high, P0128, etc.)
    • DTC count: 1–2 (specific sensor(s) affected)
    • Season: skewed June–September (Indian monsoon season)
    • Complaint: Engine overheating (false sensor reading), Rough idling
    • Warranty: Always Customer Failure
    """
    rows = []
    complaints = ["Engine overheating", "Rough idling", "OBD Light ON",
                  "Starting Problem", "High fuel consumption"]
    complaint_weights = [0.38, 0.28, 0.18, 0.10, 0.06]

    # Monsoon season weight (higher in Jun–Sep)
    month_weights = [0.04, 0.04, 0.05, 0.05, 0.06, 0.15, 0.18, 0.16, 0.12, 0.06, 0.05, 0.04]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(12.7, 0.55), 11.0, 13.5))

        # 1 or 2 sensor-specific DTCs
        dtc_count = 1 if rng.random() < 0.75 else 2
        primary = str(rng.choice(DTC_SENSOR_MOISTURE))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_SENSOR_MOISTURE))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))
        # Coastal/humid regions — more Honda, Hyundai, TATA (passenger cars)
        customer  = str(rng.choice(CUSTOMERS, p=[0.12, 0.15, 0.18, 0.14, 0.14, 0.15, 0.12]))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year, month_weights),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "Sensor short due to moisture",
            "Warranty Decision": "Customer Failure",
            "Supplier": supplier,
        })
    return rows


def gen_ntf(n: int) -> list[dict]:
    """
    NTF — No Trouble Found.
    
    Physical reality: Complaint logged but no fault reproduced in workshop.
    Often intermittent DTCs that self-cleared.
    
    Key distinguishing features:
    • Voltage: 12.5–14.4V (normal operating range)
    • DTC: 82% empty, 12% mild intermittent P-codes (not sensor/ASIC specific)
    • Complaint: Predominantly generic "OBD Light ON"
    • Warranty: Always According to Specification
    """
    rows = []
    complaints = ["OBD Light ON", "Engine overheating", "Rough idling",
                  "Low pickup", "High fuel consumption"]
    complaint_weights = [0.78, 0.08, 0.07, 0.04, 0.03]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(13.2, 0.55), 12.5, 14.4))

        # 82% no DTC, 18% a mild intermittent code
        if rng.random() < 0.82:
            dtc = ""
        else:
            dtc = str(rng.choice(DTC_NTF_MILD[:-1]))  # exclude empty string

        complaint = str(rng.choice(complaints, p=complaint_weights))
        supplier  = str(rng.choice(SUPPLIERS))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "NTF",
            "Warranty Decision": "According to Specification",
            "Supplier": supplier,
        })
    return rows


def gen_connector_damage(n: int) -> list[dict]:
    """
    Connector damage.
    
    Physical reality: Connector pin corrosion or physical damage causes 
    intermittent circuit faults in brake/body systems. Characterised by 
    C-series (chassis) and B-series (body) DTCs.
    
    Key distinguishing features:
    • Voltage: 12.5–14.5V (normal)
    • DTC: C-codes or B-codes — NEVER pure P or U codes
    • Complaint: Brake warning light ON (dominant), OBD Light ON
    • Warranty: 80% Production Failure (manufacturing crimp issue), 20% Customer
    """
    rows = []
    complaints = ["Brake warning light ON", "OBD Light ON", "Starting Problem",
                  "Engine overheating", "Low pickup"]
    complaint_weights = [0.72, 0.15, 0.07, 0.04, 0.02]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        voltage = float(np.clip(rng.normal(13.3, 0.65), 12.5, 14.5))

        dtc_count = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        dtc_codes = list(rng.choice(DTC_CONNECTOR, size=min(dtc_count, len(DTC_CONNECTOR)), replace=False))
        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = str(rng.choice(complaints, p=complaint_weights))
        # Connector issues: any supplier; Valeo/Bosch most common for harness supply
        supplier  = str(rng.choice(SUPPLIERS, p=[0.15, 0.30, 0.28, 0.18, 0.09]))
        customer  = str(rng.choice(CUSTOMERS))
        warranty  = str(rng.choice(
            ["Production Failure", "Customer Failure"], p=[0.80, 0.20]))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "Connector damage",
            "Warranty Decision": warranty,
            "Supplier": supplier,
        })
    return rows


def gen_controller_failure(n: int) -> list[dict]:
    """
    Controller failure due to supplier production failure.
    
    Physical reality: ECU/controller has an internal hardware defect from 
    manufacturing (solder joint, component tolerance, firmware bug). 
    The controller under-performs, causing low bus voltage as protection 
    circuits activate.
    
    Key distinguishing features:
    • Voltage: 9.5–11.5V  (under-voltage — controller's internal regulator failing)
    • DTC: U-series codes (CAN/LIN communication loss — controller stops talking)
    • DTC count: 1–2 (communication bus goes down)
    • Complaint: Vehicle not starting, Starting Problem
    • Warranty: Always Production Failure
    """
    rows = []
    complaints = ["Vehicle not starting", "Starting Problem",
                  "Engine jerking during acceleration", "Low pickup"]
    complaint_weights = [0.42, 0.34, 0.16, 0.08]

    for _ in range(n):
        year = int(rng.choice(YEARS))
        # Under-voltage from failing internal regulator
        voltage = float(np.clip(rng.normal(10.4, 0.60), 9.0, 11.5))

        dtc_count = 1 if rng.random() < 0.82 else 2
        primary = str(rng.choice(DTC_CONTROLLER))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_CONTROLLER))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary

        complaint = str(rng.choice(complaints, p=complaint_weights))
        # Controller failures: Bosch, STM, Delphi supply ECUs
        supplier  = str(rng.choice(SUPPLIERS, p=[0.05, 0.35, 0.08, 0.25, 0.27]))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer,
            "Year": year,
            "Date": random_date(year),
            "Voltage": round(voltage, 1),
            "DTC": dtc,
            "Customer Complaint": complaint,
            "Failure Analysis": "controller failure due to supplier production failure",
            "Warranty Decision": "Production Failure",
            "Supplier": supplier,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

TARGET = 12_000

counts = {
    "ntf":         3600,   # 30% — most common in real warranty centres
    "track":       2400,   # 20%
    "asic":        1440,   # 12%
    "moisture":    1440,   # 12%
    "connector":   1800,   # 15%
    "controller":  1320,   # 11%
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

# Generate QC numbers
df.insert(2, "QC_Number", [
    f"QC-{row['Year']}-{str(i+1).zfill(5)}"
    for i, row in df.iterrows()
])

# Reorder to match original column order
col_order = ["Customer", "Year", "Date", "QC_Number", "Customer Complaint",
             "DTC", "Voltage", "Failure Analysis", "Warranty Decision", "Supplier"]
df = df[col_order]

print(f"\nGenerated {len(df):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== FAILURE ANALYSIS DISTRIBUTION ===")
print(df['Failure Analysis'].value_counts())

print("\n=== VOLTAGE STATS BY FAILURE ANALYSIS ===")
voltage_stats = df.groupby('Failure Analysis')['Voltage'].agg(['mean','std','min','max'])
print(voltage_stats.round(2))

print("\n=== DTC FEATURE STATS BY FAILURE ANALYSIS ===")
df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True).astype(int)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True).astype(int)
df['has_C'] = df['DTC'].fillna('').str.contains(r'\bC\d', regex=True).astype(int)
df['has_B'] = df['DTC'].fillna('').str.contains(r'\bB\d', regex=True).astype(int)
df['dtc_count'] = df['DTC'].fillna('').apply(lambda x: len([c for c in x.split(',') if c.strip()]))
print(df.groupby('Failure Analysis')[['has_P','has_U','has_C','has_B','dtc_count']].mean().round(3))

print("\n=== VOLTAGE SEPARATION: ASIC vs TRACK ===")
asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
print(f"ASIC  voltage: {asic.min():.1f}–{asic.max():.1f}V  mean={asic.mean():.2f}")
print(f"Track voltage: {track.min():.1f}–{track.max():.1f}V  mean={track.mean():.2f}")
print(f"Any ASIC overlap >16V: {(asic > 16.0).sum()} rows ({(asic > 16.0).mean()*100:.1f}%)")
print(f"Any Track overlap <=16V: {(track <= 16.0).sum()} rows ({(track <= 16.0).mean()*100:.1f}%)")

# Remove helper columns before saving
df.drop(columns=['has_P','has_U','has_C','has_B','dtc_count'], inplace=True)

out_path = "/mnt/user-data/outputs/synthetic_warranty_claims_v3.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Saved to {out_path}")
