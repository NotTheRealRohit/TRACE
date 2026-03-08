"""
Realistic Warranty Claims Dataset Generator — v6
=================================================
Changes from v5:

  [DATASET QUALITY — Realism & ML signal improvements]

  Improvement 1 — Expanded complaint vocabulary (9 → 14 categories)
    v5 NTF was 78% "OBD Light ON" — a single complaint dominating a 15k-row class
    creates a fragile OHE feature. v6 distributes NTF realistically across 7 complaints
    reflecting true workshop intake patterns (intermittent, self-clearing faults).
    New complaint labels added:
      • "ABS warning light ON"        — C-code wheel speed sensor failures
      • "Battery warning light ON"    — Voltage/alternator related faults
      • "Multiple warning lights ON"  — CAN bus / controller failures (U-codes)
      • "Engine stalling"             — Sensor dropout / moisture faults
      • "Transmission jerking"        — TCM communication loss (U0101)

  Improvement 2 — DTC-driven complaint biasing (dtc_biased_complaint)
    Previously, complaint was drawn blindly from a pool independent of the DTC
    that was already selected. Now each DTC carries a semantic bias:
      P0480/P0481/P0482 (cooling fan)     → 80% "Engine overheating"
      P0300-P0304 (misfire)               → weighted "Rough idling" / "Engine jerking"
      P0562/P0563 (voltage)               → 60% "Battery warning light ON"
      C0031/C0036/C0045/C0051 (wheel spd) → 70% "ABS warning light ON"
      U-codes (CAN loss)                  → 60% "Multiple warning lights ON"
      P0420/P0430/P0455/P0456 (cat/EVAP)  → 80% "OBD Light ON"
      P0601-P0613 (ECM internal)          → 45% "Multiple warning lights ON"
    This creates a meaningful DTC↔complaint correlation that the ML TF-IDF and
    OHE features can jointly learn from, without introducing data leakage.

  Improvement 3 — Connector damage complaint rebalancing
    v5: 72% Brake warning light ON (C-codes include ABS wheel speed sensors)
    v6: 30% Brake warning light ON / 28% ABS warning light ON / 15% OBD / 12% Multi
    Rationale: C0031/C0036/C0045 are ABS wheel speed sensor circuits — they trigger
    the ABS warning lamp specifically, not the general brake warning. This also
    creates a stronger differentiator vs sensor moisture class.

  Improvement 4 — Seasonal distribution for Track burnt
    High-EOS events peak in hot months (Apr–Jul) when alternator load is highest
    due to A/C usage. Previously Track burnt had uniform month distribution.
    month_weights now peak Apr–Jun (summer AC load stress on regulator).

  Improvement 5 — B2AAA code removed (was fabricated, not in DTC sheet)
    Replaced with additional weight on real B-codes from the supplied DTC sheet.
    All DTC codes in all pools now correspond exactly to the DTC sheet.

  Improvement 6 — NTF DTC pool expanded with intermittent codes
    Added P0340 (camshaft sensor — intermittent false-trigger) and P0325
    (knock sensor circuit — can trip on rough roads) as realistic NTF DTCs.
    These appear in real workshops as single-occurrence codes that cannot be
    reproduced during inspection.

  Improvement 7 — Sensor moisture complaint variety (5 → 6 types)
    "Engine stalling" added as a realistic sensor-dropout symptom.
    "Engine overheating" reduced from 38% → 25% (was overfit to one code type).
    Distribution now reflects the diversity of sensor codes (O2, MAP, IAT, coolant).

  Improvement 8 — Controller failure complaint variety (4 → 5 types)
    "Multiple warning lights ON" added as primary complaint (35%) — the most
    common real-world presentation of a CAN bus communication failure.
    "Transmission jerking" added for U0101 (TCM lost comm) scenarios.

  [RETAINED from v5]
  - 50,000 rows
  - Temporal year weighting (2024 has 3× more claims than 2019)
  - Voltage zones with clean separation between failure classes
  - Two-zone ASIC warranty binary split (CF/PF by voltage band)
  - Mileage_km column with failure-class-appropriate distributions
  - Monsoon peak weighting for sensor moisture (Jun–Sep)
  - Connector damage 6% secondary P-code (mixed harness/powertrain fault)
  - ~0.4% label noise at ASIC boundary and connector damage
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
YEAR_WEIGHTS = [0.08, 0.10, 0.14, 0.18, 0.22, 0.28]  # sums to 1.0

# ─────────────────────────────────────────────────────────────────────────────
# ALL COMPLAINTS (14 categories — expanded from 9 in v5)
# ─────────────────────────────────────────────────────────────────────────────
#
# New in v6: ABS warning light ON, Battery warning light ON,
#             Multiple warning lights ON, Engine stalling, Transmission jerking
#
ALL_COMPLAINTS = [
    "OBD Light ON",                        # Generic check-engine / emission warning
    "Brake warning light ON",              # Hydraulic brake system warning
    "ABS warning light ON",                # ABS / wheel speed sensor specific      [NEW]
    "Battery warning light ON",            # Charging system / alternator warning    [NEW]
    "Multiple warning lights ON",          # Simultaneous multi-system failure       [NEW]
    "Vehicle not starting",                # Complete no-start
    "Starting Problem",                    # Intermittent / slow crank
    "Engine overheating",                  # Coolant temp high
    "Rough idling",                        # Unstable idle RPM
    "Engine jerking during acceleration",  # Misfire / hesitation
    "Engine stalling",                     # Sudden engine cut-out                   [NEW]
    "Low pickup",                          # Poor throttle response
    "High fuel consumption",               # Excessive fuel use
    "Transmission jerking",                # Hard shift / jerk in gear change        [NEW]
]

# ─────────────────────────────────────────────────────────────────────────────
# DTC pools — all codes verified against the supplied DTC sheet
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

# Connector damage: exclusively C/B codes (chassis/body) — B2AAA removed (v6: was fabricated)
# All codes verified against DTC sheet. Weights added to reflect field frequency:
# wheel speed sensor codes (C0031/C0036/C0045/C0051) are the most common connector failures.
DTC_CONNECTOR = [
    "C0031",   # Right Front Wheel Speed Sensor Circuit
    "C0036",   # Right Rear Wheel Speed Sensor Circuit
    "C0045",   # Left Rear Wheel Speed Sensor Circuit (intermittent)
    "C0051",   # Left Rear Wheel Speed Sensor Circuit Fault
    "C0265",   # Steering Angle Sensor — Range/Performance
    "C0460",   # Pressure Control Solenoid A (Transmission Hydraulic)
    "C0550",   # Brake Switch A/B Correlation — Brake Light Switch Mismatch
    "B1234",   # Manufacturer-specific Body Fault (interior lighting circuit)
    "B1031",   # Body Control Module Fault — Door Lock Actuator
    "B1045",   # Airbag/Restraint System Diagnostics (seat sensor)
    "B2960",   # Airbag Restraint System Fault — Deployment/Sensor Fault
    "B3055",   # Occupant Detection / Seat Sensor Fault
]

# Connector DTC weights — wheel speed sensors most prevalent in field
DTC_CONNECTOR_WEIGHTS = [
    0.12,   # C0031 — very common (front wheel bearing area, exposed)
    0.12,   # C0036
    0.11,   # C0045
    0.11,   # C0051
    0.09,   # C0265
    0.09,   # C0460
    0.09,   # C0550
    0.07,   # B1234
    0.07,   # B1031
    0.07,   # B1045
    0.08,   # B2960
    0.08,   # B3055  (weights sum to ~1.10 due to rounding; will normalise)
]
# Normalise
_w = np.array(DTC_CONNECTOR_WEIGHTS, dtype=float)
DTC_CONNECTOR_WEIGHTS = (_w / _w.sum()).tolist()

# Secondary P-codes that can appear in connector damage (6% mixed fault scenario)
DTC_CONNECTOR_SECONDARY_P = [
    "P0340",   # Camshaft Position Sensor Circuit Malfunction
    "P0325",   # Knock Sensor Circuit Malfunction
    "P0500",   # Vehicle Speed Sensor Malfunction
    "P0720",   # Output Shaft Speed Sensor Circuit Malfunction
]

# Controller failure: U-codes (CAN/LIN communication loss)
# Added U0164 and U0184 from DTC sheet (v6)
DTC_CONTROLLER = [
    "U0073",   # Control Module Communication Bus Off
    "U0100",   # Lost Communication with ECM/PCM "A"
    "U0101",   # Lost Communication with TCM
    "U0103",   # Lost Communication with Gear Shift Control Module
    "U0121",   # Lost Communication with ABS Control Module
    "U0122",   # Lost Communication with Vehicle Dynamics Control Module
    "U0131",   # Lost Communication with Power Steering Control Module / BCM
    "U0140",   # Lost Communication with Body Control Module / SRS
    "U0155",   # Lost Communication with Instrument Panel Cluster
    "U0001",   # High Speed CAN Communication Bus
    "U0164",   # Invalid Data Received From Instrument Panel Cluster  [NEW in v6]
    "U0184",   # Invalid Data Received From Body Control Module        [NEW in v6]
]

# NTF: mild/intermittent P-codes — expanded with camshaft and knock sensor in v6
DTC_NTF_MILD = [
    "",        # No DTC stored (most common for NTF)
    "P0455",   # Evaporative Emission System Leak (large) — often false
    "P0456",   # Evaporative Emission System Leak (small) — often false
    "P0171",   # System Too Lean — intermittent sensor drift
    "P0174",   # System Too Lean Bank 2
    "P0300",   # Random Misfire — single occurrence, self-cleared
    "P0316",   # Misfire Detected on Startup — clears at warmup
    "P0420",   # Catalyst System Efficiency Below Threshold — borderline cat
    "P0430",   # Catalyst System Efficiency Below Threshold Bank 2
    "P0340",   # Camshaft Position Sensor Circuit — intermittent false-trigger [NEW]
    "P0325",   # Knock Sensor Circuit — trips on rough roads / intermittent     [NEW]
]

# ─────────────────────────────────────────────────────────────────────────────
# DTC → Complaint semantic bias map
# ─────────────────────────────────────────────────────────────────────────────
# Maps a DTC code to a (complaint, probability) tuple.
# When the DTC is selected, that complaint is chosen with the given probability
# and replaced by a class-default complaint otherwise. This creates a realistic
# DTC↔symptom correlation without deterministic leakage.

DTC_COMPLAINT_BIAS = {
    # Cooling fan codes → engine overheating
    "P0480": ("Engine overheating",          0.80),
    "P0481": ("Engine overheating",          0.80),
    "P0482": ("Engine overheating",          0.75),

    # Misfire codes → rough idle / engine jerking
    "P0300": ("Rough idling",                0.45),
    "P0301": ("Engine jerking during acceleration", 0.50),
    "P0302": ("Engine jerking during acceleration", 0.50),
    "P0303": ("Rough idling",                0.50),
    "P0304": ("Rough idling",                0.50),
    "P0316": ("Rough idling",                0.55),

    # Voltage codes → battery warning
    "P0562": ("Battery warning light ON",    0.60),
    "P0563": ("Battery warning light ON",    0.55),
    "P0620": ("Battery warning light ON",    0.50),
    "P1682": ("Vehicle not starting",        0.55),
    "P0615": ("Vehicle not starting",        0.55),

    # Catalyst / EVAP → OBD light
    "P0420": ("OBD Light ON",                0.80),
    "P0430": ("OBD Light ON",                0.80),
    "P0455": ("OBD Light ON",                0.82),
    "P0456": ("OBD Light ON",                0.85),

    # Lean codes → rough idle / high fuel consumption
    "P0171": ("High fuel consumption",       0.45),
    "P0174": ("High fuel consumption",       0.45),

    # ECM internal faults → multiple warning lights
    "P0601": ("Multiple warning lights ON",  0.40),
    "P0602": ("Multiple warning lights ON",  0.45),
    "P0604": ("Multiple warning lights ON",  0.40),
    "P0605": ("Multiple warning lights ON",  0.40),
    "P0606": ("Multiple warning lights ON",  0.45),
    "P0607": ("Multiple warning lights ON",  0.40),
    "P0608": ("Low pickup",                  0.40),
    "P0610": ("Multiple warning lights ON",  0.40),
    "P0611": ("Rough idling",                0.45),
    "P0613": ("Transmission jerking",        0.45),

    # Coolant temp sensor → overheating
    "P0117": ("Engine overheating",          0.55),
    "P0118": ("Engine overheating",          0.55),
    "P0128": ("Engine overheating",          0.45),

    # O2 / air sensor → rough idling / high fuel
    "P0131": ("Rough idling",                0.45),
    "P0135": ("High fuel consumption",       0.40),
    "P0038": ("Engine stalling",             0.40),
    "P0054": ("Engine stalling",             0.40),
    "P0112": ("Starting Problem",            0.45),
    "P0113": ("Starting Problem",            0.45),
    "P0196": ("Rough idling",                0.45),
    "P0197": ("Engine stalling",             0.40),
    "P0072": ("Engine stalling",             0.40),
    "P0073": ("Engine stalling",             0.40),
    "P0069": ("Rough idling",                0.45),

    # Wheel speed sensors → ABS warning
    "C0031": ("ABS warning light ON",        0.70),
    "C0036": ("ABS warning light ON",        0.70),
    "C0045": ("ABS warning light ON",        0.70),
    "C0051": ("ABS warning light ON",        0.70),
    "C0265": ("Brake warning light ON",      0.55),
    "C0550": ("Brake warning light ON",      0.60),
    "C0460": ("Brake warning light ON",      0.45),

    # Body codes → OBD or multiple warning lights
    "B1234": ("OBD Light ON",                0.45),
    "B1031": ("OBD Light ON",                0.40),
    "B1045": ("Multiple warning lights ON",  0.40),
    "B2960": ("Starting Problem",            0.45),
    "B3055": ("Multiple warning lights ON",  0.35),

    # CAN / U-codes → multiple warning lights
    "U0001": ("Multiple warning lights ON",  0.65),
    "U0073": ("Multiple warning lights ON",  0.65),
    "U0100": ("Multiple warning lights ON",  0.60),
    "U0101": ("Transmission jerking",        0.50),
    "U0103": ("Multiple warning lights ON",  0.55),
    "U0121": ("ABS warning light ON",        0.55),
    "U0122": ("Multiple warning lights ON",  0.55),
    "U0131": ("Multiple warning lights ON",  0.55),
    "U0140": ("Multiple warning lights ON",  0.55),
    "U0155": ("Multiple warning lights ON",  0.55),
    "U0164": ("Multiple warning lights ON",  0.50),
    "U0184": ("Multiple warning lights ON",  0.50),

    # Camshaft / knock sensor → rough idling (intermittent, often NTF)
    "P0340": ("Rough idling",                0.50),
    "P0325": ("Rough idling",                0.45),
    "P0500": ("Low pickup",                  0.45),
    "P0720": ("Transmission jerking",        0.50),
}


def pick_complaint_with_dtc_bias(dtc_string: str, fallback_pool: list, fallback_weights: list) -> str:
    """
    Select a customer complaint that is semantically consistent with the DTC(s) logged.

    For each DTC code in the string, check if it has a bias entry.  The first
    (primary) DTC's bias is applied.  If the random draw succeeds, the biased
    complaint is returned; otherwise fall through to the class-specific pool.

    Args:
        dtc_string:       Comma-separated DTC string, e.g. "P0480, P0562"
        fallback_pool:    List of complaint strings for this failure class
        fallback_weights: Corresponding probabilities for fallback_pool (must sum to 1)

    Returns:
        A single complaint string.
    """
    if dtc_string:
        primary_dtc = dtc_string.split(",")[0].strip().upper()
        if primary_dtc in DTC_COMPLAINT_BIAS:
            biased_complaint, bias_prob = DTC_COMPLAINT_BIAS[primary_dtc]
            # Only apply bias if the complaint exists in our master list
            if rng.random() < bias_prob and biased_complaint in ALL_COMPLAINTS:
                return biased_complaint
    # Fallback: class-specific distribution
    return str(rng.choice(fallback_pool, p=fallback_weights))


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def pick_year() -> int:
    """Pick year with temporal weighting — more recent years have more claims."""
    return int(rng.choice(YEARS, p=YEAR_WEIGHTS))


def random_date(year: int, month_weights=None) -> str:
    """Generate a random date. Correctly handles all months including December."""
    if month_weights is None:
        month = int(rng.integers(1, 13))
    else:
        month = int(rng.choice(range(1, 13), p=month_weights))

    if month == 12:
        days_in_month = 31
    else:
        days_in_month = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

    day = int(rng.integers(1, days_in_month + 1))
    return f"{year}-{month:02d}-{day:02d}"


def gen_mileage_km(fa_class: str) -> int:
    """
    Generate realistic mileage for a claim based on failure class.

    - Controller failure (production defect): typically early life, low mileage
    - ASIC EOS: peaks at 30-80k (alternator wear)
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

    Complaint distribution (v6):
      "Battery warning light ON" added as a primary symptom for high-V ASIC failures
      where the charging system warning fires before the ECM faults become visible.
      "Multiple warning lights ON" added for cases with dual ECM faults.

    Warranty: voltage-correlated two-zone binary split (retained from v5).
    """
    rows = []
    # v6: Added "Battery warning light ON" (22%) — common first symptom of ASIC
    # overvoltage before ECM codes fully manifest. "Multiple warning lights ON"
    # appears when two P06xx codes are logged simultaneously.
    fallback_complaints = [
        "Starting Problem",
        "Battery warning light ON",
        "Low pickup",
        "High fuel consumption",
        "Rough idling",
        "Engine jerking during acceleration",
        "Multiple warning lights ON",
    ]
    fallback_weights = [0.20, 0.22, 0.18, 0.14, 0.12, 0.08, 0.06]

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

        complaint = pick_complaint_with_dtc_bias(dtc, fallback_complaints, fallback_weights)
        supplier  = str(rng.choice(SUPPLIERS, p=[0.08, 0.12, 0.08, 0.32, 0.40]))
        customer  = str(rng.choice(CUSTOMERS, p=[0.20, 0.15, 0.12, 0.12, 0.13, 0.13, 0.15]))

        # Two-zone binary warranty split (retained from v5)
        if voltage <= 14.7:
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.78, 0.22]))
        elif voltage >= 15.4:
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.38, 0.62]))
        else:
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.60, 0.40]))

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

    v6 changes:
      - Seasonal peak in summer months (Apr–Jul): highest alternator stress from
        air conditioning, consistent with real-world EOS field data.
      - "Multiple warning lights ON" (20%) added — catastrophic failure triggers
        simultaneous CAN dropout and multiple module warnings.
      - "Battery warning light ON" (12%) added — alternator overvoltage often
        shows as a battery/charging warning before total failure.
      - "Engine stalling" (5%) added — voltage spike can cause instant ECU reset
        presenting as a stall event.
    """
    rows = []
    # v6: expanded complaint pool for track burnt (catastrophic multi-system event)
    fallback_complaints = [
        "Vehicle not starting",
        "Multiple warning lights ON",
        "Engine overheating",
        "Starting Problem",
        "Battery warning light ON",
        "Engine jerking during acceleration",
        "Engine stalling",
    ]
    fallback_weights = [0.27, 0.20, 0.18, 0.15, 0.12, 0.05, 0.03]

    # v6: summer peak — high AC load stresses alternator voltage regulator
    # Apr–Jul: 0.12 each (peak), other months lower
    month_weights = [0.04, 0.04, 0.07, 0.12, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07, 0.07]
    assert abs(sum(month_weights) - 1.0) < 1e-9, "month_weights must sum to 1"

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(17.8, 1.1), 16.1, 20.0))

        dtc_count = int(rng.choice([2, 3, 4, 5], p=[0.35, 0.35, 0.20, 0.10]))
        dtc_codes = list(rng.choice(DTC_TRACK, size=min(dtc_count, len(DTC_TRACK)), replace=False))
        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = pick_complaint_with_dtc_bias(dtc, fallback_complaints, fallback_weights)
        supplier  = str(rng.choice(SUPPLIERS))
        customer  = str(rng.choice(CUSTOMERS))

        rows.append({
            "Customer": customer, "Year": year,
            "Date": random_date(year, month_weights),
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

    v6 changes:
      - "Engine stalling" added (15%) — moisture-induced sensor dropout can cause
        ECU to command fuel cut or misfire → sudden stall event.
      - "Engine overheating" reduced 38% → 25% — was overfit to coolant sensor codes
        only; the pool also includes O2, MAP, IAT sensors with different symptoms.
      - "Rough idling" increased 28% → 20% to reflect O2/IAT sensor faults.
      - DTC bias ensures coolant sensor codes (P0117/P0118) bias toward overheating,
        and O2/air codes bias toward stalling/rough idle.
    """
    rows = []
    fallback_complaints = [
        "Engine overheating",
        "Rough idling",
        "OBD Light ON",
        "Engine stalling",
        "High fuel consumption",
        "Starting Problem",
    ]
    fallback_weights = [0.25, 0.20, 0.18, 0.18, 0.12, 0.07]

    # Indian monsoon peak: Jun–Sep
    month_weights = [0.04, 0.04, 0.05, 0.05, 0.06, 0.15, 0.18, 0.16, 0.12, 0.06, 0.05, 0.04]
    assert abs(sum(month_weights) - 1.0) < 1e-9

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(12.7, 0.55), 11.0, 13.5))

        dtc_count = 1 if rng.random() < 0.75 else 2
        primary = str(rng.choice(DTC_SENSOR_MOISTURE))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_SENSOR_MOISTURE))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary

        complaint = pick_complaint_with_dtc_bias(dtc, fallback_complaints, fallback_weights)
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

    Voltage: 12.5–14.4V (normal operating range)
    DTC: 82% empty; 18% mild intermittent code from pool (expanded in v6)
    Warranty: 100% According to Specification

    v6 changes:
      - NTF complaint distribution is now spread across 7 complaints (was 78% OBD).
      - "OBD Light ON" reduced 78% → 38% — more realistic; NTF presentations are
        diverse (the customer's symptom cannot be reproduced, not that the OBD lamp
        is always the trigger).
      - "Engine stalling" (12%) added — common NTF: customer reports intermittent
        stall, technician cannot reproduce on inspection.
      - "Starting Problem" (8%) added — intermittent hard-start that clears on warm-up.
      - DTC pool expanded with P0340 (cam sensor) and P0325 (knock sensor) for the
        18% NTF rows that have a mild stored code.

    Note: NTF rows intentionally use the DTC bias sparingly (the intermittent code
    often doesn't match the complaint because the fault wasn't present when the
    complaint occurred — this mild mismatch is physically realistic for NTF).
    """
    rows = []
    # v6: realistic NTF complaint distribution — tech can't reproduce any of these
    fallback_complaints = [
        "OBD Light ON",
        "Engine stalling",
        "Rough idling",
        "Low pickup",
        "High fuel consumption",
        "Engine overheating",
        "Starting Problem",
    ]
    # OBD still most common NTF complaint but no longer dominant
    fallback_weights = [0.38, 0.13, 0.14, 0.11, 0.09, 0.07, 0.08]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(13.2, 0.55), 12.5, 14.4))

        if rng.random() < 0.82:
            dtc = ""
        else:
            dtc = str(rng.choice([c for c in DTC_NTF_MILD if c]))  # exclude empty

        # For NTF: apply DTC bias at a lower probability (50% of normal bias strength)
        # because the stored code is intermittent and may not match the presenting
        # complaint — this realistic mismatch is characteristic of NTF cases.
        if dtc and dtc in DTC_COMPLAINT_BIAS:
            biased_complaint, bias_prob = DTC_COMPLAINT_BIAS[dtc]
            if rng.random() < bias_prob * 0.50 and biased_complaint in ALL_COMPLAINTS:
                complaint = biased_complaint
            else:
                complaint = str(rng.choice(fallback_complaints, p=fallback_weights))
        else:
            complaint = str(rng.choice(fallback_complaints, p=fallback_weights))

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
    DTC: C/B-codes (weighted toward wheel speed sensors); 65% 1-code, 28% 2-codes, 7% 3-codes
    Warranty: 80% Production Failure (crimp defect), 20% Customer Failure
    6% of cases have an additional secondary P-code (broken harness → sensor P-code)

    v6 changes:
      - "ABS warning light ON" added (28%) — wheel speed sensor codes (C0031, C0036,
        C0045, C0051) specifically trigger the ABS warning, not the general brake warning.
        Separating these creates a much stronger ML discriminator for connector damage.
      - "Brake warning light ON" reduced 72% → 30% — retained as the primary for
        steering-angle/brake-switch connector failures (C0265, C0550).
      - "Multiple warning lights ON" added (12%) — multi-code connector failures.
      - "Battery warning light ON" added (7%) — body code connector failures
        (B2960 key/ignition circuit) can trigger charging system warning display.
      - DTC_CONNECTOR now weighted toward wheel speed sensors (most common in field).
      - B2AAA removed (was fabricated); all codes verified against DTC sheet.
    """
    rows = []
    fallback_complaints = [
        "Brake warning light ON",
        "ABS warning light ON",
        "OBD Light ON",
        "Multiple warning lights ON",
        "Starting Problem",
        "Battery warning light ON",
    ]
    fallback_weights = [0.30, 0.28, 0.14, 0.12, 0.09, 0.07]

    for _ in range(n):
        year = pick_year()
        voltage = float(np.clip(rng.normal(13.3, 0.65), 12.5, 14.5))

        dtc_count = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        dtc_codes = list(rng.choice(
            DTC_CONNECTOR,
            size=min(dtc_count, len(DTC_CONNECTOR)),
            replace=False,
            p=DTC_CONNECTOR_WEIGHTS,
        ))

        # 6% chance: secondary P-code from broken sensor circuit
        if rng.random() < 0.06:
            secondary_p = str(rng.choice(DTC_CONNECTOR_SECONDARY_P))
            dtc_codes.append(secondary_p)

        dtc = ", ".join(str(c) for c in dtc_codes)

        complaint = pick_complaint_with_dtc_bias(dtc, fallback_complaints, fallback_weights)
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

    v6 changes:
      - "Multiple warning lights ON" added as primary complaint (35%) — the most
        common real-world presentation of a CAN bus communication failure. Multiple
        modules going offline simultaneously triggers dashboard lamp cluster.
      - "Transmission jerking" added (12%) — U0101 (TCM lost comm) often presents
        as gear shift hesitation or jerking before full failure.
      - "Vehicle not starting" reduced 42% → 28%, "Starting Problem" reduced 34% → 18%
        (still significant but no longer the only symptoms).
    """
    rows = []
    fallback_complaints = [
        "Multiple warning lights ON",
        "Vehicle not starting",
        "Starting Problem",
        "Transmission jerking",
        "Engine jerking during acceleration",
    ]
    fallback_weights = [0.35, 0.28, 0.18, 0.12, 0.07]

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

        complaint = pick_complaint_with_dtc_bias(dtc, fallback_complaints, fallback_weights)
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

    # ASIC: flip at voltage boundary zone (reduced noise multiplier from v5 retained)
    asic_mask = (
        (df[fa_col] == "ASIC CJ327 failure due to EOS") &
        (df["Voltage"].between(14.8, 15.2)) &
        (df[wd_col].isin(["Production Failure", "Customer Failure"]))
    )
    asic_candidates = df[asic_mask].index
    n_asic_flip = int(len(asic_candidates) * noise_rate * 1.2)
    if n_asic_flip > 0:
        flip_idx = rng.choice(asic_candidates, size=min(n_asic_flip, len(asic_candidates)), replace=False)
        for idx in flip_idx:
            current = df.at[idx, wd_col]
            df.at[idx, wd_col] = "Customer Failure" if current == "Production Failure" else "Production Failure"

    # Connector damage: ~1.5% label flip (customer abuse vs manufacturing crimp)
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

# Inject label noise BEFORE generating QC numbers
print("\nInjecting label noise...")
df = inject_warranty_label_noise(df, noise_rate=0.015)

# Generate QC numbers
df.insert(2, "QC_Number", [
    f"QC-{row['Year']}-{str(i+1).zfill(6)}"
    for i, row in df.iterrows()
])

# Final column order
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
print((fa_counts / len(df) * 100).round(1))

print("\n=== WARRANTY DECISION DISTRIBUTION ===")
print(df['Warranty Decision'].value_counts())

print("\n=== VOLTAGE STATS BY FAILURE ANALYSIS ===")
print(df.groupby('Failure Analysis')['Voltage'].agg(['mean','std','min','max']).round(2))

print("\n=== YEAR DISTRIBUTION ===")
print(df['Year'].value_counts().sort_index())

print("\n=== COMPLAINT DISTRIBUTION BY FAILURE TYPE ===")
for fa in sorted(df['Failure Analysis'].unique()):
    sub = df[df['Failure Analysis'] == fa]
    print(f"\n{fa} (n={len(sub)}):")
    print(sub['Customer Complaint'].value_counts(normalize=True).round(3).to_string())

print("\n=== COMPLAINT DIVERSITY CHECK ===")
print("Total unique complaint types:", df['Customer Complaint'].nunique())
print("All complaint types present:")
for c in sorted(df['Customer Complaint'].unique()):
    print(f"  {c}: {(df['Customer Complaint']==c).sum():,}")

df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True).astype(int)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True).astype(int)
df['has_C'] = df['DTC'].fillna('').str.contains(r'\bC\d', regex=True).astype(int)
df['has_B'] = df['DTC'].fillna('').str.contains(r'\bB\d', regex=True).astype(int)
df['dtc_count'] = df['DTC'].fillna('').apply(
    lambda x: len([c for c in x.split(',') if c.strip()])
)
print("\n=== DTC FEATURE STATS BY FAILURE ANALYSIS ===")
print(df.groupby('Failure Analysis')[['has_P','has_U','has_C','has_B','dtc_count']].mean().round(3))

print("\n=== VOLTAGE SEPARATION ===")
asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
ctrl  = df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage']
print(f"ASIC  voltage: {asic.min():.2f}–{asic.max():.2f}V  mean={asic.mean():.2f}")
print(f"Track voltage: {track.min():.2f}–{track.max():.2f}V  mean={track.mean():.2f}")
print(f"Ctrl  voltage: {ctrl.min():.2f}–{ctrl.max():.2f}V  mean={ctrl.mean():.2f}")
print(f"ASIC  overlap >16V:  {(asic > 16.0).sum()} rows")
print(f"Track overlap <=16V: {(track <= 16.0).sum()} rows")
print(f"Ctrl  overlap >11.5V: {(ctrl > 11.5).sum()} rows")

print("\n=== CONNECTOR: rows with secondary P-code ===")
conn_df = df[df['Failure Analysis'] == 'Connector damage']
has_p_conn = conn_df['has_P'].sum()
print(f"  {has_p_conn} / {len(conn_df)} connector rows have a P-code ({has_p_conn/len(conn_df)*100:.1f}%)")

print("\n=== SENSOR MOISTURE: rows with no DTC ===")
sensor_df = df[df['Failure Analysis'] == 'Sensor short due to moisture']
no_dtc = (sensor_df['DTC'].fillna('') == '').sum()
print(f"  {no_dtc} / {len(sensor_df)} sensor rows have no DTC ({no_dtc/len(sensor_df)*100:.1f}%)")

print("\n=== TRACK BURNT: seasonal distribution ===")
track_df = df[df['Failure Analysis'] == 'Track burnt due to EOS'].copy()
track_df['Month'] = pd.to_datetime(track_df['Date']).dt.month
print(track_df['Month'].value_counts().sort_index())

print("\n=== DTC-COMPLAINT CORRELATION SANITY CHECK ===")
# Check that wheel speed sensor DTCs bias toward ABS warning
ws_dtcs = ["C0031", "C0036", "C0045", "C0051"]
for d in ws_dtcs:
    sub = df[df['DTC'].str.contains(d, na=False)]
    if len(sub) > 0:
        abs_pct = (sub['Customer Complaint'] == 'ABS warning light ON').mean()
        print(f"  {d}: {len(sub)} rows, ABS warning: {abs_pct:.1%}")

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

# All complaints must be in master list
unknown_complaints = set(df['Customer Complaint'].unique()) - set(ALL_COMPLAINTS)
assert len(unknown_complaints) == 0, f"Unknown complaints: {unknown_complaints}"

# All 14 complaint types must appear in the dataset
for c in ALL_COMPLAINTS:
    count = (df['Customer Complaint'] == c).sum()
    assert count > 0, f"Complaint '{c}' never appears — check weights"

# Voltage hard boundaries
df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True)

asic  = df[df['Failure Analysis']=='ASIC CJ327 failure due to EOS']['Voltage']
track = df[df['Failure Analysis']=='Track burnt due to EOS']['Voltage']
ctrl  = df[df['Failure Analysis']=='controller failure due to supplier production failure']['Voltage']
assert (asic  > 16.0).sum() == 0,  "ASIC voltage leaked above 16V!"
assert (track <= 16.0).sum() == 0,  "Track voltage dipped below 16.1V!"
assert (ctrl  > 11.5).sum() == 0,  "Controller voltage leaked above 11.5V!"

# DTC prefix constraints
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

out_path = "/mnt/user-data/outputs/synthetic_warranty_claims_v6.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Saved to {out_path}")
print(f"   Shape: {df.shape}")
