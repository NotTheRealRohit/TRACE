"""
Realistic Warranty Claims Dataset Generator — v9

Changes from v8:
  C4 — Probabilistic WD for Sensor Moisture (was 100% deterministic CF)
  C5 — NTF supplier draw now uses SUPPLIER_WEIGHTS_DEFAULT (was uniform)
  C6 — ASIC customer weights aligned to CUSTOMER_WEIGHTS (Option A)
  W5 — Voltage x mileage EOS wear correlation (Track burnt + ASIC)
  W6 — Bimodal mileage for Connector Damage (assembly-defect early population)
  W7 — Mileage-boundary label noise (CF<->PF in 88k-112k band)
  DTC1 — ASIC primary DTC pool expanded from 10 to 12 codes (named constants)
  DTC2 — Cross-FA injection pre-filters native FA pool (true foreign codes only)
  DTC3 — Companion DTC logic for Track, ASIC, Controller (mechanistic co-occurrence)
  DTC4 — Directional secondary DTC pool for Sensor Moisture
"""
import numpy as np
import pandas as pd
from datetime import datetime
import random
from scipy.stats import truncnorm

rng = np.random.default_rng(42)
random.seed(42)

# ── Core reference data ────────────────────────────────────────────────────────

CUSTOMERS = ["Ashok Leyland", "M&M", "Honda", "Kia", "Toyota", "Hyundai", "TATA"]
SUPPLIERS  = ["Hanon", "Bosch", "Valeo", "Delphi", "STM"]
YEARS      = list(range(2019, 2026))
YEAR_WEIGHTS = [0.04, 0.07, 0.10, 0.13, 0.18, 0.23, 0.25]

# W1 — Realistic OEM volume weights (TATA + Hyundai dominant, Kia smallest)
CUSTOMER_WEIGHTS = [0.10, 0.13, 0.14, 0.12, 0.18, 0.20, 0.13]

# W1 — Realistic supplier weights: Bosch dominant
SUPPLIER_WEIGHTS_DEFAULT = [0.10, 0.35, 0.25, 0.20, 0.10]

# W4 — DTCs that are genuinely ambiguous across multiple failure modes
DTC_AMBIGUOUS_CROSS = [
    "P0300",   # random misfire
    "P0171",   # system lean
    "P0325",   # knock sensor
    "P0340",   # cam sensor
    "U0100",   # lost comm CAN
    "U0155",   # lost comm instrument cluster
    "P0500",   # VSS
    "P0420",   # catalyst efficiency
]
CROSS_FA_INJECT_RATE = 0.04

ALL_COMPLAINTS = [
    "OBD Light ON","Brake warning light ON","ABS warning light ON",
    "Battery warning light ON","Multiple warning lights ON","Vehicle not starting",
    "Starting Problem","Engine overheating","Rough idling",
    "Engine jerking during acceleration","Engine stalling","Low pickup",
    "High fuel consumption","Transmission jerking",
]

# DTC1 — ASIC pool split into primary (12 codes) and secondary (4 codes)
DTC_ASIC_PRIMARY = ["P0601","P0602","P0604","P0605","P0606","P0607","P0608","P0610",
                    "P0562","P0563","P0611","P0613"]
DTC_ASIC_SECONDARY = ["P0634","P068A","P068B","P0686"]
DTC_ASIC = DTC_ASIC_PRIMARY + DTC_ASIC_SECONDARY

DTC_TRACK = ["P0562","P0563","P0300","P0301","P0302","P0303","P0304","P0480","P0481","P0482",
             "P1682","P0615","P0620","U0001","U0100","U0101","U0155",
             "P0351","P0352","P0353","P0354","P0355","P0356",
             "P0691","P0692","P0693","P0694","P0560","P0616","P0617","P0305","P0306"]

DTC_SENSOR_MOISTURE = ["P0113","P0112","P0118","P0117","P0128","P0197","P0196","P0072","P0073",
                       "P0038","P0054","P0131","P0135","P0069",
                       "P0116","P0111","P0101","P0096","P0097","P0098"]

# DTC4 — Directional secondary pool for sensor moisture
DTC_SENSOR_MOISTURE_SECONDARY = [
    "P0171","P0174","P0131","P0141","P0161",   # cross-pool downstream consequences
    "P0112","P0113","P0117","P0118","P0096","P0097",   # same-pool logical pairs
]

DTC_CONNECTOR = ["C0031","C0036","C0045","C0051","C0265","C0460","C0550",
                 "B1234","B1031","B1045","B2960","B3055","C0082"]

_w_conn = np.array([0.11,0.11,0.10,0.10,0.08,0.08,0.08,0.07,0.06,0.06,0.07,0.07,0.01], dtype=float)
DTC_CONNECTOR_WEIGHTS = (_w_conn / _w_conn.sum()).tolist()

DTC_CONNECTOR_SECONDARY_P = ["P0340","P0325","P0500","P0720","P0571","P0572","P0573","P0501","P0502"]

DTC_CONTROLLER = ["U0073","U0100","U0101","U0103","U0121","U0122","U0131","U0140","U0155",
                  "U0001","U0164","U0184",
                  "U0002","U0028","U0029","U0037","U0038","U0401","U0402","U0422","U0128","U0114"]

DTC_NTF_MILD = ["","P0455","P0456","P0171","P0174","P0300","P0316","P0420","P0430","P0340","P0325",
                "P0341","P0342","P0343","P0327","P0328","P0457","P0104"]

# DTC3 — Companion pairs: mechanistically linked codes
DTC_COMPANIONS = {
    "P0562": ("P0563", 0.55), "P0563": ("P0562", 0.55),
    "P0691": ("P0692", 0.60), "P0692": ("P0691", 0.60),
    "P0616": ("P0617", 0.50), "P0617": ("P0616", 0.50),
    "P0351": ("P0352", 0.45), "P0352": ("P0351", 0.45),
    "P0601": ("P0604", 0.50), "P0604": ("P0601", 0.50),
    "P0602": ("P0606", 0.45), "P0606": ("P0602", 0.45),
    "P0605": ("P0607", 0.40), "P0607": ("P0605", 0.40),
    "U0100": ("U0101", 0.60), "U0101": ("U0100", 0.60),
    "U0073": ("U0001", 0.55), "U0001": ("U0073", 0.55),
    "U0121": ("U0140", 0.45), "U0140": ("U0121", 0.45),
}

# DTC2 — Native DTC pool per FA class
FA_NATIVE_DTC_POOLS = {
    "NTF":                                                   set(DTC_NTF_MILD),
    "Track burnt due to EOS":                                set(DTC_TRACK),
    "ASIC CJ327 failure due to EOS":                         set(DTC_ASIC),
    "Sensor short due to moisture":                          set(DTC_SENSOR_MOISTURE),
    "Connector damage":                                      set(DTC_CONNECTOR) | set(DTC_CONNECTOR_SECONDARY_P),
    "controller failure due to supplier production failure": set(DTC_CONTROLLER),
}

DTC_COMPLAINT_BIAS = {
    "P0480":("Engine overheating",0.80),"P0481":("Engine overheating",0.80),"P0482":("Engine overheating",0.75),
    "P0300":("Rough idling",0.45),"P0301":("Engine jerking during acceleration",0.50),
    "P0302":("Engine jerking during acceleration",0.50),"P0303":("Rough idling",0.50),
    "P0304":("Rough idling",0.50),"P0316":("Rough idling",0.55),
    "P0562":("Battery warning light ON",0.60),"P0563":("Battery warning light ON",0.55),
    "P0620":("Battery warning light ON",0.50),"P1682":("Vehicle not starting",0.55),
    "P0615":("Vehicle not starting",0.55),"P0420":("OBD Light ON",0.80),"P0430":("OBD Light ON",0.80),
    "P0455":("OBD Light ON",0.82),"P0456":("OBD Light ON",0.85),
    "P0171":("High fuel consumption",0.45),"P0174":("High fuel consumption",0.45),
    "P0601":("Multiple warning lights ON",0.40),"P0602":("Multiple warning lights ON",0.45),
    "P0604":("Multiple warning lights ON",0.40),"P0605":("Multiple warning lights ON",0.40),
    "P0606":("Multiple warning lights ON",0.45),"P0607":("Multiple warning lights ON",0.40),
    "P0608":("Low pickup",0.40),"P0610":("Multiple warning lights ON",0.40),
    "P0611":("Rough idling",0.45),"P0613":("Transmission jerking",0.45),
    "P0117":("Engine overheating",0.55),"P0118":("Engine overheating",0.55),
    "P0128":("Engine overheating",0.45),"P0131":("Rough idling",0.45),
    "P0135":("High fuel consumption",0.40),"P0038":("Engine stalling",0.40),
    "P0054":("Engine stalling",0.40),"P0112":("Starting Problem",0.45),
    "P0113":("Starting Problem",0.45),"P0196":("Rough idling",0.45),
    "P0197":("Engine stalling",0.40),"P0072":("Engine stalling",0.40),
    "P0073":("Engine stalling",0.40),"P0069":("Rough idling",0.45),
    "C0031":("ABS warning light ON",0.70),"C0036":("ABS warning light ON",0.70),
    "C0045":("ABS warning light ON",0.70),"C0051":("ABS warning light ON",0.70),
    "C0265":("Brake warning light ON",0.55),"C0550":("Brake warning light ON",0.60),
    "C0460":("Brake warning light ON",0.45),"B1234":("OBD Light ON",0.45),
    "B1031":("OBD Light ON",0.40),"B1045":("Multiple warning lights ON",0.40),
    "B2960":("Starting Problem",0.45),"B3055":("Multiple warning lights ON",0.35),
    "U0001":("Multiple warning lights ON",0.65),"U0073":("Multiple warning lights ON",0.65),
    "U0100":("Multiple warning lights ON",0.60),"U0101":("Transmission jerking",0.50),
    "U0103":("Multiple warning lights ON",0.55),"U0121":("ABS warning light ON",0.55),
    "U0122":("Multiple warning lights ON",0.55),"U0131":("Multiple warning lights ON",0.55),
    "U0140":("Multiple warning lights ON",0.55),"U0155":("Multiple warning lights ON",0.55),
    "U0164":("Multiple warning lights ON",0.50),"U0184":("Multiple warning lights ON",0.50),
    "P0340":("Rough idling",0.50),"P0325":("Rough idling",0.45),
    "P0500":("Low pickup",0.45),"P0720":("Transmission jerking",0.50),
    "P0351":("Engine jerking during acceleration",0.55),"P0352":("Engine jerking during acceleration",0.55),
    "P0353":("Rough idling",0.50),"P0354":("Rough idling",0.50),
    "P0355":("Engine jerking during acceleration",0.50),"P0356":("Engine jerking during acceleration",0.50),
    "P0305":("Engine jerking during acceleration",0.50),"P0306":("Rough idling",0.50),
    "P0691":("Engine overheating",0.70),"P0692":("Engine overheating",0.70),
    "P0693":("Engine overheating",0.65),"P0694":("Engine overheating",0.65),
    "P0560":("Battery warning light ON",0.50),"P0616":("Vehicle not starting",0.60),
    "P0617":("Vehicle not starting",0.60),"P0634":("Multiple warning lights ON",0.50),
    "P068A":("Vehicle not starting",0.55),"P068B":("Vehicle not starting",0.50),
    "P0686":("Vehicle not starting",0.55),"P0116":("Engine overheating",0.50),
    "P0111":("Starting Problem",0.45),"P0101":("High fuel consumption",0.40),
    "P0096":("Starting Problem",0.45),"P0097":("Starting Problem",0.45),
    "P0098":("Starting Problem",0.40),"C0082":("Brake warning light ON",0.60),
    "P0571":("Brake warning light ON",0.65),"P0572":("Brake warning light ON",0.65),
    "P0573":("Brake warning light ON",0.60),"P0501":("Low pickup",0.50),
    "P0502":("Low pickup",0.50),"U0002":("Multiple warning lights ON",0.65),
    "U0028":("Multiple warning lights ON",0.65),"U0029":("Multiple warning lights ON",0.60),
    "U0037":("Multiple warning lights ON",0.60),"U0038":("Multiple warning lights ON",0.55),
    "U0401":("Multiple warning lights ON",0.55),"U0402":("Transmission jerking",0.50),
    "U0422":("Multiple warning lights ON",0.55),"U0128":("Brake warning light ON",0.55),
    "U0114":("Multiple warning lights ON",0.50),"P0341":("Rough idling",0.50),
    "P0342":("Rough idling",0.45),"P0343":("Rough idling",0.45),
    "P0327":("Rough idling",0.45),"P0328":("Rough idling",0.40),
    "P0457":("OBD Light ON",0.90),"P0104":("High fuel consumption",0.40),
    # DTC4 secondary pool entries
    "P0141":("Engine stalling",0.40),"P0161":("Engine stalling",0.40),
}

# ── Helper functions ───────────────────────────────────────────────────────────

def pick_complaint_with_dtc_bias(dtc_string, fallback_pool, fallback_weights):
    if dtc_string:
        primary_dtc = dtc_string.split(",")[0].strip().upper()
        if primary_dtc in DTC_COMPLAINT_BIAS:
            biased_complaint, bias_prob = DTC_COMPLAINT_BIAS[primary_dtc]
            if rng.random() < bias_prob and biased_complaint in ALL_COMPLAINTS:
                return biased_complaint
    return str(rng.choice(fallback_pool, p=fallback_weights))

def random_date(year, month_weights=None):
    if month_weights is None:
        month = int(rng.integers(1, 13))
    else:
        month = int(rng.choice(range(1, 13), p=month_weights))
    if month == 12:
        days_in_month = 31
    else:
        days_in_month = (datetime(year, month+1, 1) - datetime(year, month, 1)).days
    day = int(rng.integers(1, days_in_month+1))
    return f"{year}-{month:02d}-{day:02d}"

def truncated_normal(rng_instance, mean, std, lo, hi, size=None):
    a, b = (lo - mean) / std, (hi - mean) / std
    if size is None:
        u = float(rng_instance.uniform(0, 1))
        return float(truncnorm.ppf(u, a, b, loc=mean, scale=std))
    else:
        u = rng_instance.uniform(0, 1, size=size)
        return truncnorm.ppf(u, a, b, loc=mean, scale=std).astype(float)

def apply_eos_voltage_nudge(base_voltage, mileage_km, nudge_scale=0.10):
    """W5: mileage-proportional voltage nudge for EOS classes."""
    mileage_factor = mileage_km / 200_000
    nudge = nudge_scale * mileage_factor + float(rng.normal(0, 0.03))
    return round(base_voltage + nudge, 2)

def maybe_append_cross_fa_dtc(existing_dtc: str, fa_class: str) -> str:
    """DTC2: cross-FA injection pre-filtered to exclude native pool codes."""
    if rng.random() < CROSS_FA_INJECT_RATE:
        native = FA_NATIVE_DTC_POOLS.get(fa_class, set())
        candidates = [c for c in DTC_AMBIGUOUS_CROSS if c not in native]
        if not candidates:
            return existing_dtc
        extra = str(rng.choice(candidates))
        if existing_dtc:
            current_codes = [c.strip() for c in existing_dtc.split(",")]
            if extra not in current_codes:
                return existing_dtc + ", " + extra
        else:
            return extra
    return existing_dtc

def maybe_append_companion_dtc(existing_dtc: str) -> str:
    """DTC3: inject mechanistically linked companion code if one exists."""
    if not existing_dtc:
        return existing_dtc
    current_codes = [c.strip() for c in existing_dtc.split(",")]
    for code in current_codes:
        if code in DTC_COMPANIONS:
            companion, prob = DTC_COMPANIONS[code]
            if companion not in current_codes and rng.random() < prob:
                return existing_dtc + ", " + companion
    return existing_dtc

def gen_mileage_km(fa_class):
    if fa_class == "controller failure due to supplier production failure":
        mu, sigma = np.log(18000), 0.75
        return int(np.clip(rng.lognormal(mu, sigma), 500, 90000))
    elif fa_class == "ASIC CJ327 failure due to EOS":
        mu, sigma = np.log(45000), 0.65
        return int(np.clip(rng.lognormal(mu, sigma), 3000, 180000))
    elif fa_class == "NTF":
        mu, sigma = np.log(35000), 0.80
        return int(np.clip(rng.lognormal(mu, sigma), 500, 220000))
    elif fa_class == "Sensor short due to moisture":
        mu, sigma = np.log(50000), 0.70
        return int(np.clip(rng.lognormal(mu, sigma), 2000, 200000))
    elif fa_class == "Track burnt due to EOS":
        mu, sigma = np.log(60000), 0.65
        return int(np.clip(rng.lognormal(mu, sigma), 1000, 220000))
    elif fa_class == "Connector damage":
        # W6: wear-and-tear only; gen_connector_damage handles the full bimodal draw
        mu, sigma = np.log(75000), 0.60
        return int(np.clip(rng.lognormal(mu, sigma), 15000, 230000))
    else:
        mu, sigma = np.log(40000), 0.75
        return int(np.clip(rng.lognormal(mu, sigma), 500, 220000))

# C3 — Year-aware row allocation with FA drift
TARGET = 100_000
YEAR_WEIGHTS_DICT = dict(zip(YEARS, YEAR_WEIGHTS))

BASE_FA_PROPS = {
    "ntf": 0.300, "track": 0.200, "connector": 0.150,
    "asic": 0.120, "moisture": 0.120, "controller": 0.110,
}
FA_DRIFT_PER_YEAR = {
    "ntf": +0.002, "track": -0.003, "connector": +0.004,
    "asic": -0.003, "moisture": +0.003, "controller": -0.003,
}

def compute_year_counts(target, year_weights_dict, base_props, drift_per_year):
    base_year = min(year_weights_dict.keys())
    year_fa_counts = {}
    total_by_fa = {fa: 0 for fa in base_props}
    for year, yr_weight in year_weights_dict.items():
        yr_rows = int(round(target * yr_weight))
        age = year - base_year
        raw_props = {fa: max(0.01, base_p + drift_per_year[fa] * age)
                     for fa, base_p in base_props.items()}
        total_p = sum(raw_props.values())
        norm_props = {fa: p / total_p for fa, p in raw_props.items()}
        year_fa_counts[year] = {}
        allocated = 0
        fas = list(norm_props.keys())
        for i, fa in enumerate(fas):
            if i < len(fas) - 1:
                cnt = int(round(yr_rows * norm_props[fa]))
            else:
                cnt = yr_rows - allocated
            year_fa_counts[year][fa] = cnt
            allocated += cnt
            total_by_fa[fa] += cnt
    return total_by_fa, year_fa_counts

counts, year_fa_counts = compute_year_counts(TARGET, YEAR_WEIGHTS_DICT, BASE_FA_PROPS, FA_DRIFT_PER_YEAR)

def pick_year_for_fa(fa_key, yfc):
    years = list(yfc.keys())
    counts_for_fa = [yfc[yr][fa_key] for yr in years]
    total = sum(counts_for_fa)
    weights = [c / total for c in counts_for_fa]
    return int(rng.choice(years, p=weights))

# ── Generator functions ────────────────────────────────────────────────────────

def gen_asic_cj327(n):
    rows = []
    fallback = ["Starting Problem","Battery warning light ON","Low pickup","High fuel consumption",
                "Rough idling","Engine jerking during acceleration","Multiple warning lights ON"]
    fw = [0.20, 0.22, 0.18, 0.14, 0.12, 0.08, 0.06]
    for _ in range(n):
        year = pick_year_for_fa("asic", year_fa_counts)
        # W5: mileage drawn BEFORE voltage for nudge calculation
        mileage = gen_mileage_km("ASIC CJ327 failure due to EOS")
        voltage = round(float(truncated_normal(rng, mean=15.3, std=0.45, lo=13.8, hi=16.5)), 2)
        voltage = round(float(np.clip(apply_eos_voltage_nudge(voltage, mileage, nudge_scale=0.08), 13.8, 16.5)), 2)
        # DTC1: primary from expanded 12-code pool
        dtc_count = 1 if rng.random() < 0.88 else 2
        primary = str(rng.choice(DTC_ASIC_PRIMARY))
        if dtc_count == 2:
            secondary = str(rng.choice(DTC_ASIC))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary
        complaint = pick_complaint_with_dtc_bias(dtc, fallback, fw)
        # C6: customer weights aligned to global CUSTOMER_WEIGHTS
        customer = str(rng.choice(CUSTOMERS, p=CUSTOMER_WEIGHTS))
        supplier = str(rng.choice(SUPPLIERS, p=[0.08, 0.12, 0.08, 0.32, 0.40]))
        if voltage <= 14.7:
            warranty = str(rng.choice(["Production Failure","Customer Failure"], p=[0.78,0.22]))
        elif voltage >= 15.4:
            warranty = str(rng.choice(["Production Failure","Customer Failure"], p=[0.38,0.62]))
        else:
            warranty = str(rng.choice(["Production Failure","Customer Failure"], p=[0.60,0.40]))
        # DTC3 companion then DTC2 cross-FA
        dtc = maybe_append_companion_dtc(dtc)
        dtc = maybe_append_cross_fa_dtc(dtc, "ASIC CJ327 failure due to EOS")
        rows.append({"Customer": customer, "Year": year, "Date": random_date(year),
                     "Voltage": voltage, "DTC": dtc, "Customer Complaint": complaint,
                     "Failure Analysis": "ASIC CJ327 failure due to EOS",
                     "Warranty Decision": warranty, "Supplier": supplier, "Mileage_km": mileage})
    return rows

def gen_track_burnt(n):
    rows = []
    fallback = ["Vehicle not starting","Multiple warning lights ON","Engine overheating",
                "Starting Problem","Battery warning light ON","Engine jerking during acceleration","Engine stalling"]
    fw = [0.27,0.20,0.18,0.15,0.12,0.05,0.03]
    mw = [0.04,0.04,0.07,0.12,0.13,0.12,0.10,0.09,0.08,0.07,0.07,0.07]
    for _ in range(n):
        year = pick_year_for_fa("track", year_fa_counts)
        # W5: mileage drawn BEFORE voltage
        mileage = gen_mileage_km("Track burnt due to EOS")
        voltage = round(float(truncated_normal(rng, mean=17.8, std=1.1, lo=15.5, hi=21.0)), 2)
        voltage = round(float(np.clip(apply_eos_voltage_nudge(voltage, mileage, nudge_scale=0.10), 15.5, 21.0)), 2)
        dtc_count = int(rng.choice([2,3,4,5], p=[0.35,0.35,0.20,0.10]))
        dtc_codes = list(rng.choice(DTC_TRACK, size=min(dtc_count, len(DTC_TRACK)), replace=False))
        dtc = ", ".join(str(c) for c in dtc_codes)
        complaint = pick_complaint_with_dtc_bias(dtc, fallback, fw)
        warranty = str(rng.choice(
            ["Customer Failure", "Production Failure", "According to Specification"],
            p=[0.960, 0.030, 0.010]
        ))
        # DTC3 companion then DTC2 cross-FA
        dtc = maybe_append_companion_dtc(dtc)
        dtc = maybe_append_cross_fa_dtc(dtc, "Track burnt due to EOS")
        rows.append({"Customer": str(rng.choice(CUSTOMERS, p=CUSTOMER_WEIGHTS)),
                     "Year": year, "Date": random_date(year, mw), "Voltage": voltage,
                     "DTC": dtc, "Customer Complaint": complaint,
                     "Failure Analysis": "Track burnt due to EOS", "Warranty Decision": warranty,
                     "Supplier": str(rng.choice(SUPPLIERS, p=SUPPLIER_WEIGHTS_DEFAULT)),
                     "Mileage_km": mileage})
    return rows

def gen_sensor_moisture(n):
    rows = []
    fallback = ["Engine overheating","Rough idling","OBD Light ON","Engine stalling",
                "High fuel consumption","Starting Problem"]
    fw = [0.25,0.20,0.18,0.18,0.12,0.07]
    mw = [0.04,0.04,0.05,0.05,0.06,0.15,0.18,0.16,0.12,0.06,0.05,0.04]
    for _ in range(n):
        year = pick_year_for_fa("moisture", year_fa_counts)
        voltage = round(float(truncated_normal(rng, mean=12.7, std=0.55, lo=10.5, hi=14.2)), 2)
        primary = str(rng.choice(DTC_SENSOR_MOISTURE))
        # DTC4: directional secondary (60% cross-pool consequence, 40% same-pool)
        if rng.random() >= 0.75:
            if rng.random() < 0.60:
                secondary = str(rng.choice(DTC_SENSOR_MOISTURE_SECONDARY))
            else:
                secondary = str(rng.choice(DTC_SENSOR_MOISTURE))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary
        complaint = pick_complaint_with_dtc_bias(dtc, fallback, fw)
        # C4: probabilistic WD (was 100% Customer Failure)
        warranty = str(rng.choice(
            ["Customer Failure", "According to Specification", "Production Failure"],
            p=[0.965, 0.025, 0.010]
        ))
        dtc = maybe_append_cross_fa_dtc(dtc, "Sensor short due to moisture")
        rows.append({"Customer": str(rng.choice(CUSTOMERS, p=[0.12,0.15,0.18,0.14,0.14,0.15,0.12])),
                     "Year": year, "Date": random_date(year, mw), "Voltage": voltage, "DTC": dtc,
                     "Customer Complaint": complaint, "Failure Analysis": "Sensor short due to moisture",
                     "Warranty Decision": warranty,
                     "Supplier": str(rng.choice(SUPPLIERS, p=SUPPLIER_WEIGHTS_DEFAULT)),
                     "Mileage_km": gen_mileage_km("Sensor short due to moisture")})
    return rows

def gen_ntf(n):
    rows = []
    fallback = ["OBD Light ON","Engine stalling","Rough idling","Low pickup",
                "High fuel consumption","Engine overheating","Starting Problem"]
    fw = [0.38,0.13,0.14,0.11,0.09,0.07,0.08]
    for _ in range(n):
        year = pick_year_for_fa("ntf", year_fa_counts)
        voltage = round(float(truncated_normal(rng, mean=13.2, std=0.55, lo=11.8, hi=14.8)), 2)
        dtc = "" if rng.random() < 0.80 else str(rng.choice([c for c in DTC_NTF_MILD if c]))
        if dtc and dtc in DTC_COMPLAINT_BIAS:
            bc, bp = DTC_COMPLAINT_BIAS[dtc]
            complaint = bc if (rng.random() < bp*0.50 and bc in ALL_COMPLAINTS) else str(rng.choice(fallback, p=fw))
        else:
            complaint = str(rng.choice(fallback, p=fw))
        warranty = str(rng.choice(
            ["According to Specification", "Customer Failure", "Production Failure"],
            p=[0.965, 0.025, 0.010]
        ))
        dtc = maybe_append_cross_fa_dtc(dtc, "NTF")
        rows.append({"Customer": str(rng.choice(CUSTOMERS, p=CUSTOMER_WEIGHTS)),
                     "Year": year, "Date": random_date(year), "Voltage": voltage,
                     "DTC": dtc, "Customer Complaint": complaint,
                     "Failure Analysis": "NTF", "Warranty Decision": warranty,
                     # C5: supplier now uses SUPPLIER_WEIGHTS_DEFAULT
                     "Supplier": str(rng.choice(SUPPLIERS, p=SUPPLIER_WEIGHTS_DEFAULT)),
                     "Mileage_km": gen_mileage_km("NTF")})
    return rows

def gen_connector_damage(n):
    rows = []
    fallback = ["Brake warning light ON","ABS warning light ON","OBD Light ON",
                "Multiple warning lights ON","Starting Problem","Battery warning light ON"]
    fw = [0.30, 0.28, 0.14, 0.12, 0.09, 0.07]
    for _ in range(n):
        year = pick_year_for_fa("connector", year_fa_counts)
        voltage = round(float(truncated_normal(rng, mean=13.3, std=0.65, lo=11.5, hi=15.0)), 2)
        # W6: bimodal mileage + mileage-conditional WD
        if rng.random() < 0.15:
            mileage = int(np.clip(rng.lognormal(np.log(2500), 0.50), 200, 8000))
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.92, 0.08]))
        else:
            mileage = int(np.clip(rng.lognormal(np.log(75000), 0.60), 15000, 230000))
            warranty = str(rng.choice(["Production Failure", "Customer Failure"], p=[0.80, 0.20]))
        dtc_count = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        dtc_codes = list(rng.choice(DTC_CONNECTOR, size=min(dtc_count, len(DTC_CONNECTOR)),
                                    replace=False, p=DTC_CONNECTOR_WEIGHTS))
        if rng.random() < 0.06:
            dtc_codes.append(str(rng.choice(DTC_CONNECTOR_SECONDARY_P)))
        dtc = ", ".join(str(c) for c in dtc_codes)
        complaint = pick_complaint_with_dtc_bias(dtc, fallback, fw)
        dtc = maybe_append_cross_fa_dtc(dtc, "Connector damage")
        rows.append({"Customer": str(rng.choice(CUSTOMERS, p=CUSTOMER_WEIGHTS)),
                     "Year": year, "Date": random_date(year), "Voltage": voltage,
                     "DTC": dtc, "Customer Complaint": complaint,
                     "Failure Analysis": "Connector damage", "Warranty Decision": warranty,
                     "Supplier": str(rng.choice(SUPPLIERS, p=[0.15, 0.30, 0.28, 0.18, 0.09])),
                     "Mileage_km": mileage})
    return rows

def gen_controller_failure(n):
    rows = []
    fallback = ["Multiple warning lights ON","Vehicle not starting","Starting Problem",
                "Transmission jerking","Engine jerking during acceleration"]
    fw = [0.35,0.28,0.18,0.12,0.07]
    for _ in range(n):
        year = pick_year_for_fa("controller", year_fa_counts)
        voltage = round(float(truncated_normal(rng, mean=10.4, std=0.60, lo=8.5, hi=12.5)), 2)
        primary = str(rng.choice(DTC_CONTROLLER))
        if rng.random() >= 0.82:
            secondary = str(rng.choice(DTC_CONTROLLER))
            dtc = f"{primary}, {secondary}" if secondary != primary else primary
        else:
            dtc = primary
        complaint = pick_complaint_with_dtc_bias(dtc, fallback, fw)
        warranty = str(rng.choice(
            ["Production Failure", "Customer Failure", "According to Specification"],
            p=[0.960, 0.030, 0.010]
        ))
        # DTC3 companion then DTC2 cross-FA
        dtc = maybe_append_companion_dtc(dtc)
        dtc = maybe_append_cross_fa_dtc(dtc, "controller failure due to supplier production failure")
        rows.append({"Customer": str(rng.choice(CUSTOMERS, p=CUSTOMER_WEIGHTS)),
                     "Year": year, "Date": random_date(year), "Voltage": voltage,
                     "DTC": dtc, "Customer Complaint": complaint,
                     "Failure Analysis": "controller failure due to supplier production failure",
                     "Warranty Decision": warranty,
                     "Supplier": str(rng.choice(SUPPLIERS, p=[0.05,0.35,0.08,0.25,0.27])),
                     "Mileage_km": gen_mileage_km("controller failure due to supplier production failure")})
    return rows

# W3 + W7 — Label noise
def inject_warranty_label_noise(df, noise_rate=0.015):
    df = df.copy()
    total_flipped = 0

    # ASIC — boundary-zone voltage flip
    asic_mask = ((df["Failure Analysis"] == "ASIC CJ327 failure due to EOS") &
                 (df["Voltage"].between(14.8, 15.2)) &
                 (df["Warranty Decision"].isin(["Production Failure", "Customer Failure"])))
    asic_cand = df[asic_mask].index
    n_flip = int(len(asic_cand) * noise_rate * 1.2)
    if n_flip > 0:
        for idx in rng.choice(asic_cand, size=min(n_flip, len(asic_cand)), replace=False):
            c = df.at[idx, "Warranty Decision"]
            df.at[idx, "Warranty Decision"] = "Customer Failure" if c == "Production Failure" else "Production Failure"
        total_flipped += n_flip

    # Connector — random flip
    conn_mask = ((df["Failure Analysis"] == "Connector damage") &
                 (df["Warranty Decision"].isin(["Production Failure", "Customer Failure"])))
    conn_cand = df[conn_mask].index
    n_flip2 = int(len(conn_cand) * noise_rate)
    if n_flip2 > 0:
        for idx in rng.choice(conn_cand, size=min(n_flip2, len(conn_cand)), replace=False):
            c = df.at[idx, "Warranty Decision"]
            df.at[idx, "Warranty Decision"] = "Customer Failure" if c == "Production Failure" else "Production Failure"
        total_flipped += n_flip2

    # NTF — ATS -> CF
    ntf_cand = df[df["Failure Analysis"] == "NTF"].index
    n_flip3 = int(len(ntf_cand) * 0.008)
    if n_flip3 > 0:
        for idx in rng.choice(ntf_cand, size=min(n_flip3, len(ntf_cand)), replace=False):
            if df.at[idx, "Warranty Decision"] == "According to Specification":
                df.at[idx, "Warranty Decision"] = "Customer Failure"
        total_flipped += n_flip3

    # Track burnt — CF -> PF
    track_cand = df[df["Failure Analysis"] == "Track burnt due to EOS"].index
    n_flip4 = int(len(track_cand) * 0.007)
    if n_flip4 > 0:
        for idx in rng.choice(track_cand, size=min(n_flip4, len(track_cand)), replace=False):
            if df.at[idx, "Warranty Decision"] == "Customer Failure":
                df.at[idx, "Warranty Decision"] = "Production Failure"
        total_flipped += n_flip4

    # Controller — PF -> CF
    ctrl_cand = df[df["Failure Analysis"] == "controller failure due to supplier production failure"].index
    n_flip5 = int(len(ctrl_cand) * 0.007)
    if n_flip5 > 0:
        for idx in rng.choice(ctrl_cand, size=min(n_flip5, len(ctrl_cand)), replace=False):
            if df.at[idx, "Warranty Decision"] == "Production Failure":
                df.at[idx, "Warranty Decision"] = "Customer Failure"
        total_flipped += n_flip5

    # Sensor moisture — CF -> PF
    sensor_cand = df[df["Failure Analysis"] == "Sensor short due to moisture"].index
    n_flip6 = int(len(sensor_cand) * 0.010)
    if n_flip6 > 0:
        for idx in rng.choice(sensor_cand, size=min(n_flip6, len(sensor_cand)), replace=False):
            if df.at[idx, "Warranty Decision"] == "Customer Failure":
                df.at[idx, "Warranty Decision"] = "Production Failure"
        total_flipped += n_flip6

    # W7 — Mileage boundary zone: CF<->PF at 88k-112k km
    BOUNDARY_LO, BOUNDARY_HI = 88_000, 112_000
    BOUNDARY_NOISE_RATE = 0.035
    boundary_mask = (
        df["Mileage_km"].between(BOUNDARY_LO, BOUNDARY_HI) &
        df["Warranty Decision"].isin(["Customer Failure", "Production Failure"])
    )
    boundary_cand = df[boundary_mask].index
    n_boundary = int(len(boundary_cand) * BOUNDARY_NOISE_RATE)
    if n_boundary > 0:
        for idx in rng.choice(boundary_cand, size=min(n_boundary, len(boundary_cand)), replace=False):
            c = df.at[idx, "Warranty Decision"]
            df.at[idx, "Warranty Decision"] = "Customer Failure" if c == "Production Failure" else "Production Failure"
        total_flipped += n_boundary

    print(f"  Label noise: {total_flipped} rows flipped ({total_flipped / len(df) * 100:.2f}%)")
    print(f"  Boundary zone rows: {boundary_mask.sum():,}  |  boundary flips: {n_boundary}")
    return df

# ── Generation ─────────────────────────────────────────────────────────────────

print(f"Generating {TARGET:,} rows with temporal FA drift...")
print(f"  FA totals: { {k: v for k, v in counts.items()} }")

rows = (gen_ntf(counts["ntf"]) + gen_track_burnt(counts["track"]) +
        gen_asic_cj327(counts["asic"]) + gen_sensor_moisture(counts["moisture"]) +
        gen_connector_damage(counts["connector"]) + gen_controller_failure(counts["controller"]))

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nInjecting label noise...")
df = inject_warranty_label_noise(df, noise_rate=0.015)

df.insert(2, "QC_Number", [f"QC-{row['Year']}-{str(i+1).zfill(6)}" for i, row in df.iterrows()])
col_order = ["Customer","Year","Date","QC_Number","Customer Complaint","DTC","Voltage",
             "Failure Analysis","Warranty Decision","Supplier","Mileage_km"]
df = df[col_order]

print(f"\nGenerated {len(df):,} rows, {df.shape[1]} columns")

# ── Distribution reports ───────────────────────────────────────────────────────

print("\n=== FA DISTRIBUTION ===")
fa_c = df['Failure Analysis'].value_counts()
print(fa_c)
print((fa_c / len(df) * 100).round(1))

print("\n=== WD DISTRIBUTION ===")
wd_c = df['Warranty Decision'].value_counts()
print(wd_c)
print((wd_c / len(df) * 100).round(1))

print("\n=== YEAR DISTRIBUTION ===")
print(df['Year'].value_counts().sort_index())

df['has_P'] = df['DTC'].fillna('').str.contains(r'\bP\d', regex=True).astype(int)
df['has_U'] = df['DTC'].fillna('').str.contains(r'\bU\d', regex=True).astype(int)
df['has_C'] = df['DTC'].fillna('').str.contains(r'\bC\d', regex=True).astype(int)
df['has_B'] = df['DTC'].fillna('').str.contains(r'\bB\d', regex=True).astype(int)

print("\n=== VOLTAGE SEPARATION ===")
asic_v  = df[df['Failure Analysis'] == 'ASIC CJ327 failure due to EOS']['Voltage']
track_v = df[df['Failure Analysis'] == 'Track burnt due to EOS']['Voltage']
ctrl_v  = df[df['Failure Analysis'] == 'controller failure due to supplier production failure']['Voltage']
print(f"ASIC  {asic_v.min():.2f}-{asic_v.max():.2f}V  mean={asic_v.mean():.2f}  kurt={asic_v.kurtosis():.2f}")
print(f"Track {track_v.min():.2f}-{track_v.max():.2f}V  mean={track_v.mean():.2f}  kurt={track_v.kurtosis():.2f}")
print(f"Ctrl  {ctrl_v.min():.2f}-{ctrl_v.max():.2f}V  mean={ctrl_v.mean():.2f}  kurt={ctrl_v.kurtosis():.2f}")

print("\n=== MILEAGE SKEWNESS ===")
for fa in df['Failure Analysis'].unique():
    skew = df[df['Failure Analysis'] == fa]['Mileage_km'].skew()
    print(f"  {fa[:45]:<45}  skew={skew:.2f}")

print("\n=== WD RATES BY FA CLASS (post-noise) ===")
for fa in df['Failure Analysis'].unique():
    sub = df[df['Failure Analysis'] == fa]['Warranty Decision'].value_counts(normalize=True)
    print(f"  {fa[:45]:<45}  {dict(sub.round(3))}")

print("\n=== EOS VOLTAGE-MILEAGE CORRELATION ===")
for eos_fa in ['Track burnt due to EOS', 'ASIC CJ327 failure due to EOS']:
    sub = df[df['Failure Analysis'] == eos_fa]
    corr = sub['Voltage'].corr(sub['Mileage_km'])
    print(f"  {eos_fa:<45}  Pearson r(V,km)={corr:.4f}")

print("\n=== CONNECTOR MILEAGE DISTRIBUTION ===")
conn_df = df[df['Failure Analysis'] == 'Connector damage']
early = (conn_df['Mileage_km'] < 8000).sum()
late  = (conn_df['Mileage_km'] >= 8000).sum()
print(f"  Early (<8k km):         {early:,} ({early/len(conn_df)*100:.1f}%)")
print(f"  Wear-and-tear (>=8k km): {late:,} ({late/len(conn_df)*100:.1f}%)")

print("\n=== COMPANION DTC SPOT-CHECK ===")
track_df = df[df['Failure Analysis'] == 'Track burnt due to EOS']
p562_rows = track_df[track_df['DTC'].str.contains('P0562', na=False)]
p562_w_p563 = p562_rows[p562_rows['DTC'].str.contains('P0563', na=False)]
print(f"  Track P0562 rows: {len(p562_rows):,} | with P0563: {len(p562_w_p563):,} "
      f"({len(p562_w_p563)/max(1,len(p562_rows))*100:.1f}%) — expect ~55%")
asic_df = df[df['Failure Analysis'] == 'ASIC CJ327 failure due to EOS']
p601_rows = asic_df[asic_df['DTC'].str.contains('P0601', na=False)]
p601_w_p604 = p601_rows[p601_rows['DTC'].str.contains('P0604', na=False)]
print(f"  ASIC P0601 rows: {len(p601_rows):,} | with P0604: {len(p601_w_p604):,} "
      f"({len(p601_w_p604)/max(1,len(p601_rows))*100:.1f}%) — expect ~50%")

print("\n=== DTC COUNTS BY FA CLASS ===")
for fa in df['Failure Analysis'].unique():
    sub = df[df['Failure Analysis'] == fa]
    mean_dtcs = sub['DTC'].fillna('').apply(
        lambda x: len([c for c in x.split(',') if c.strip()]) if x.strip() else 0).mean()
    print(f"  {fa[:45]:<45}  mean DTC count={mean_dtcs:.2f}")

print("\n=== SENSOR MOISTURE SECONDARY DTC USAGE ===")
sensor_df = df[df['Failure Analysis'] == 'Sensor short due to moisture']
cross_pool_codes = set(DTC_SENSOR_MOISTURE_SECONDARY) - set(DTC_SENSOR_MOISTURE)
has_cross = sensor_df['DTC'].apply(
    lambda x: any(c.strip() in cross_pool_codes for c in x.split(','))).sum()
print(f"  Rows with cross-pool secondary DTC: {has_cross:,} ({has_cross/len(sensor_df)*100:.1f}%) — expect ~15%")

# ── Validation assertions ──────────────────────────────────────────────────────

print("\n=== VALIDATION ===")
assert df.shape[0] == sum(counts.values()), f"Row count mismatch"
assert set(df['Failure Analysis'].unique()) == {
    'NTF','Track burnt due to EOS','ASIC CJ327 failure due to EOS',
    'Sensor short due to moisture','Connector damage',
    'controller failure due to supplier production failure'
}
assert set(df['Warranty Decision'].unique()) == {
    'Production Failure','Customer Failure','According to Specification'
}
for col in ['Voltage','Customer Complaint','Failure Analysis','Warranty Decision']:
    assert df[col].isna().sum() == 0, f"Nulls in {col}"
assert len(set(df['Customer Complaint'].unique()) - set(ALL_COMPLAINTS)) == 0
for c in ALL_COMPLAINTS:
    assert (df['Customer Complaint'] == c).sum() > 0, f"Missing complaint: {c}"

# Voltage distribution checks
assert asic_v.mean() > 15.0,              "ASIC mean voltage too low"
assert track_v.mean() > 17.0,             "Track mean voltage too low"
assert ctrl_v.mean() < 11.0,              "Controller mean voltage too high"
assert asic_v.quantile(0.01) > 13.5,      "ASIC p1 voltage out of range"
assert ctrl_v.quantile(0.99) < 13.0,      "Controller p99 voltage out of range"

# DTC prefix checks
asic_df_va = df[df['Failure Analysis'] == 'ASIC CJ327 failure due to EOS']
assert asic_df_va['has_P'].all(), "ASIC rows must have at least one P-code"
ctrl_df_va = df[df['Failure Analysis'] == 'controller failure due to supplier production failure']
assert ctrl_df_va['has_U'].all(), "Controller rows must have at least one U-code"

# WD rate checks
ntf_ats_rate = (df[df['Failure Analysis'] == 'NTF']['Warranty Decision'] == 'According to Specification').mean()
assert ntf_ats_rate >= 0.94, f"NTF->ATS rate too low: {ntf_ats_rate:.3f}"
track_cf_rate = (df[df['Failure Analysis'] == 'Track burnt due to EOS']['Warranty Decision'] == 'Customer Failure').mean()
assert track_cf_rate >= 0.93, f"Track->CF rate too low: {track_cf_rate:.3f}"
ctrl_pf_rate = (ctrl_df_va['Warranty Decision'] == 'Production Failure').mean()
assert ctrl_pf_rate >= 0.93, f"Controller->PF rate too low: {ctrl_pf_rate:.3f}"

# C4: Sensor moisture must now have ATS > 0
sensor_ats_rate = (df[df['Failure Analysis'] == 'Sensor short due to moisture']['Warranty Decision']
                   == 'According to Specification').mean()
assert sensor_ats_rate > 0, "Sensor moisture ATS rate is zero — C4 not applied"
print(f"  [C4] Sensor moisture ATS rate: {sensor_ats_rate:.3f}  OK")

# W5: EOS correlation check
for eos_fa in ['Track burnt due to EOS', 'ASIC CJ327 failure due to EOS']:
    sub = df[df['Failure Analysis'] == eos_fa]
    corr = sub['Voltage'].corr(sub['Mileage_km'])
    assert corr > 0, f"EOS V-km correlation not positive for {eos_fa}: {corr:.4f}"
    print(f"  [W5] {eos_fa[:35]}: r={corr:.4f}  OK")

# W6: connector early-life population
conn_early_pct = (conn_df['Mileage_km'] < 8000).mean()
assert 0.08 <= conn_early_pct <= 0.22, f"Connector early-life pct out of range: {conn_early_pct:.3f}"
print(f"  [W6] Connector early-life pct: {conn_early_pct:.3f}  OK")

# W6: class-aware mileage skewness
MILEAGE_SKEW_FLOOR = {
    "NTF": 0.6,
    "Track burnt due to EOS": 0.6,
    "ASIC CJ327 failure due to EOS": 0.6,
    "Sensor short due to moisture": 0.6,
    "controller failure due to supplier production failure": 0.6,
    "Connector damage": 0.3,
}
for fa in df['Failure Analysis'].unique():
    skew = df[df['Failure Analysis'] == fa]['Mileage_km'].skew()
    floor = MILEAGE_SKEW_FLOOR.get(fa, 0.6)
    assert skew > floor, f"Mileage skewness too low for {fa}: {skew:.2f} (floor={floor})"
print("  [W6] Mileage skewness: all classes OK")

# ATS share sanity
ats_share = (df['Warranty Decision'] == 'According to Specification').mean()
assert 0.20 <= ats_share <= 0.35, f"ATS share out of range: {ats_share:.3f}"
print(f"  ATS share: {ats_share:.3f}  OK")

print("\nAll validation assertions passed.")

df.drop(columns=['has_P','has_U','has_C','has_B'], inplace=True)

out = "/mnt/user-data/outputs/synthetic_warranty_claims_v9.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}  shape={df.shape}")
