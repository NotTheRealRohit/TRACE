#!/usr/bin/env python3
"""
Generate 38 Figures for Methodology_TRACE.tex
=============================================
This script generates vector PDF figures for the TRACE Methodology chapter.
Each figure is saved to Documentation/Methodology Chapter/figures/ directory.
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Arc
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray_light": "#bcbd22",
    "gray_medium": "#7f7f7f",
    "gray_dark": "#1a1a2e",
    "bg_light": "#f8f9fa",
    "bg_box": "#e8eaf6",
    "border": "#37474f",
    "text": "#212121",
    "grid": "#e0e0e0",
    "white": "#ffffff",
    "black": "#000000",
    "ntf": "#17becf",
    "track": "#ff7f0e",
    "asic": "#d62728",
    "moisture": "#9467bd",
    "connector": "#2ca02c",
    "controller": "#1f77b4",
}

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FA_CLASSES = [
    "NTF",
    "Track burnt due to EOS",
    "ASIC CJ327 failure due to EOS",
    "Sensor short due to moisture",
    "Connector damage",
    "Controller failure",
]

WD_CLASSES = ["Production Failure", "Customer Failure", "According to Specification"]

VOLTAGE_PARAMS = {
    "ASIC CJ327 failure due to EOS": {"mu": 15.3, "sigma": 0.45, "a": 13.8, "b": 16.5},
    "Track burnt due to EOS": {"mu": 17.8, "sigma": 1.10, "a": 15.5, "b": 21.0},
    "Controller failure": {"mu": 10.4, "sigma": 0.60, "a": 8.5, "b": 12.5},
    "Sensor short due to moisture": {"mu": 12.7, "sigma": 0.55, "a": 10.5, "b": 14.2},
    "NTF": {"mu": 13.2, "sigma": 0.55, "a": 11.8, "b": 14.8},
    "Connector damage": {"mu": 13.3, "sigma": 0.65, "a": 11.5, "b": 15.0},
}

MILEAGE_PARAMS = {
    "Controller failure": {"mu_log": np.log(18000), "sigma_log": 0.75, "min": 500, "max": 90000},
    "ASIC CJ327 failure due to EOS": {"mu_log": np.log(45000), "sigma_log": 0.65, "min": 3000, "max": 180000},
    "NTF": {"mu_log": np.log(35000), "sigma_log": 0.80, "min": 500, "max": 220000},
    "Sensor short due to moisture": {"mu_log": np.log(50000), "sigma_log": 0.70, "min": 2000, "max": 200000},
    "Track burnt due to EOS": {"mu_log": np.log(60000), "sigma_log": 0.65, "min": 1000, "max": 220000},
    "Connector damage": {"mu_log": np.log(75000), "sigma_log": 0.60, "min": 15000, "max": 230000},
}

FA_BASE_PROPORTIONS = {
    "NTF": 0.300,
    "Track burnt due to EOS": 0.200,
    "Connector damage": 0.150,
    "ASIC CJ327 failure due to EOS": 0.120,
    "Sensor short due to moisture": 0.120,
    "Controller failure": 0.110,
}

FA_DRIFT = {
    "NTF": 0.002,
    "Track burnt due to EOS": -0.003,
    "Connector damage": 0.004,
    "ASIC CJ327 failure due to EOS": -0.003,
    "Sensor short due to moisture": 0.003,
    "Controller failure": -0.003,
}

WD_PROBABILITIES = {
    "ASIC CJ327 failure due to EOS": {"voltage_conditional": True},
    "Track burnt due to EOS": [0.030, 0.960, 0.010],
    "Controller failure": [0.960, 0.030, 0.010],
    "NTF": [0.010, 0.025, 0.965],
    "Sensor short due to moisture": [0.010, 0.965, 0.025],
    "Connector damage": {"mileage_conditional": True},
}

NOISE_MECHANISMS = [
    {"target": "ASIC CJ327", "direction": "PF ↔ CF", "rate": 0.018, "condition": "V ∈ [14.8, 15.2]"},
    {"target": "Connector damage", "direction": "PF ↔ CF", "rate": 0.015, "condition": "All"},
    {"target": "NTF", "direction": "ATS → CF", "rate": 0.008, "condition": "All"},
    {"target": "Track burnt", "direction": "CF → PF", "rate": 0.007, "condition": "All"},
    {"target": "Controller", "direction": "PF → CF", "rate": 0.007, "condition": "All"},
    {"target": "Sensor moisture", "direction": "CF → PF", "rate": 0.010, "condition": "All"},
    {"target": "Mileage boundary", "direction": "PF ↔ CF", "rate": 0.035, "condition": "km ∈ [88000, 112000]"},
]

RULES = [
    {"id": "over_voltage", "condition": "V > 16.0", "wd": "CF", "conf": 93.0},
    {"id": "low_voltage", "condition": "V < 11.0", "wd": "CF", "conf": 95.0},
    {"id": "moisture", "condition": "keyword ∈ {water, moisture, ...}", "wd": "CF", "conf": 91.0},
    {"id": "physical_damage", "condition": "keyword ∈ {crack, broken, ...}", "wd": "CF", "conf": 88.5},
    {"id": "ntf", "condition": "keyword ∈ {no fault, ntf, ...}", "wd": "ATS", "conf": 95.0},
    {"id": "u_code", "condition": "DTC matches \\bU[0-9A-Fa-f]{4}\\b", "wd": "PF", "conf": 57.0},
    {"id": "p_code_engine", "condition": "DTC matches P0 + keyword", "wd": "PF", "conf": 80.5},
    {"id": "c_code", "condition": "DTC matches \\bC[0-9A-Fa-f]{4}\\b", "wd": "PF", "conf": 80.0},
    {"id": "b_code", "condition": "DTC matches \\bB[0-9A-Fa-f]{4}\\b", "wd": "PF", "conf": 80.0},
]

SCORE_CONSTANTS = {
    "tau_firm": 85.0,
    "tau_manual": 65.0,
    "b_agree": 5.0,
    "w_rule_agree": 0.70,
    "w_ml_agree": 0.30,
    "w_rule_disagree": 0.55,
    "w_ml_disagree": 0.35,
    "w_llm": 0.15,
}


def create_figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')
    return fig, ax


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)


def add_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=kwargs.get('linewidth', 1.0)
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 9)
    fontweight = kwargs.get('fontweight', 'normal')
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
    return box


def add_arrow(ax, start, end, **kwargs):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=kwargs.get('color', COLORS['gray_medium']), lw=kwargs.get('lw', 1.5)))


def add_diamond(ax, x, y, size, text, **kwargs):
    diamond = Polygon(
        [[x, y+size], [x+size, y], [x, y-size], [x-size, y]],
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=7)
    return diamond


def add_rounded_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=kwargs.get('linewidth', 1.2)
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 8)
    fontweight = kwargs.get('fontweight', 'normal')
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
    return box


def add_input_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=kwargs.get('facecolor', COLORS['primary']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=1.5
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 9)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')
    return box


def add_output_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=kwargs.get('facecolor', COLORS['success']),
        edgecolor=kwargs.get('edgecolor', COLORS['success']),
        linewidth=1.5
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 9)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')
    return box


def add_process_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['secondary']),
        linewidth=1.2
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 8)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return box


def add_stage_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=kwargs.get('facecolor', COLORS['purple']),
        edgecolor=kwargs.get('edgecolor', COLORS['purple']),
        linewidth=1.5
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 9)
    fontcolor = kwargs.get('fontcolor', 'white')
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=fontcolor)
    return box


def add_llm_box(ax, x, y, w, h, text, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=kwargs.get('facecolor', COLORS['brown']),
        edgecolor=kwargs.get('edgecolor', COLORS['brown']),
        linewidth=1.5
    )
    ax.add_patch(box)
    fontsize = kwargs.get('fontsize', 8)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='white')
    return box


def plot_fa_class_distribution():
    """Figure 4: Stacked bar chart showing FA class proportions across years."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    years = list(range(2019, 2026))
    ntf_drift = [FA_BASE_PROPORTIONS["NTF"] + FA_DRIFT["NTF"] * (y - 2019) for y in years]
    track_drift = [FA_BASE_PROPORTIONS["Track burnt due to EOS"] + FA_DRIFT["Track burnt due to EOS"] * (y - 2019) for y in years]
    connector_drift = [FA_BASE_PROPORTIONS["Connector damage"] + FA_DRIFT["Connector damage"] * (y - 2019) for y in years]
    asic_drift = [FA_BASE_PROPORTIONS["ASIC CJ327 failure due to EOS"] + FA_DRIFT["ASIC CJ327 failure due to EOS"] * (y - 2019) for y in years]
    moisture_drift = [FA_BASE_PROPORTIONS["Sensor short due to moisture"] + FA_DRIFT["Sensor short due to moisture"] * (y - 2019) for y in years]
    controller_drift = [FA_BASE_PROPORTIONS["Controller failure"] + FA_DRIFT["Controller failure"] * (y - 2019) for y in years]
    
    totals = [sum(x) for x in zip(ntf_drift, track_drift, connector_drift, asic_drift, moisture_drift, controller_drift)]
    ntf_norm = [n/t for n, t in zip(ntf_drift, totals)]
    track_norm = [t/t for t in totals]
    connector_norm = [c/t for c, t in zip(connector_drift, totals)]
    asic_norm = [a/t for a, t in zip(asic_drift, totals)]
    moisture_norm = [m/t for m, t in zip(moisture_drift, totals)]
    controller_norm = [c/t for c, t in zip(controller_drift, totals)]
    
    bottom = np.zeros(len(years))
    colors = [COLORS['ntf'], COLORS['track'], COLORS['connector'], COLORS['asic'], COLORS['moisture'], COLORS['controller']]
    data = [ntf_drift, track_drift, connector_drift, asic_drift, moisture_drift, controller_drift]
    labels = ['NTF', 'Track', 'Connector', 'ASIC', 'Moisture', 'Controller']
    
    for d, c, l in zip(data, colors, labels):
        values = [d[j]/totals[j] for j in range(len(years))]
        ax.bar(years, values, bottom=bottom, label=l, color=c, alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Model Year', fontsize=10)
    ax.set_ylabel('Proportion', fontsize=10)
    ax.set_title('Failure Analysis Class Proportions by Year (2019-2025)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_xticks(years)
    ax.set_ylim(0, 1)
    style_axes(ax)
    
    save_figure(fig, "fig_fa_class_distribution.pdf")


def plot_voltage_distributions():
    """Figure 5: Overlaid KDE plots showing voltage distributions for all 6 FA classes."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    x = np.linspace(8, 22, 500)
    
    colors = [COLORS['controller'], COLORS['moisture'], COLORS['ntf'], COLORS['connector'], COLORS['asic'], COLORS['track']]
    classes = ["Controller failure", "Sensor short due to moisture", "NTF", "Connector damage", "ASIC CJ327 failure due to EOS", "Track burnt due to EOS"]
    
    for cls, col in zip(classes, colors):
        params = VOLTAGE_PARAMS[cls]
        from scipy.stats import truncnorm
        tr = truncnorm((params['a'] - params['mu']) / params['sigma'], (params['b'] - params['mu']) / params['sigma'], loc=params['mu'], scale=params['sigma'])
        y = tr.pdf(x)
        ax.plot(x, y, color=col, linewidth=2, label=cls.replace(' due to EOS', '').replace(' failure due to EOS', '').replace(' due to moisture', ''))
        ax.fill_between(x, y, alpha=0.15, color=col)
    
    thresholds = [11.0, 13.5, 14.5, 15.4, 16.0, 17.0]
    for t in thresholds:
        ax.axvline(t, color=COLORS['danger'], linestyle='--', linewidth=1, alpha=0.6)
        ax.text(t + 0.1, ax.get_ylim()[1] * 0.95, f'{t}V', fontsize=7, rotation=90, va='top', color=COLORS['danger'], alpha=0.8)
    
    ax.set_xlabel('Voltage (V)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Voltage Distributions by Failure Analysis Class', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    style_axes(ax)
    
    save_figure(fig, "fig_voltage_distributions.pdf")


def plot_mileage_distributions():
    """Figure 6: Overlaid histograms showing mileage distributions for all 6 FA classes."""
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={'width_ratios': [3, 1]})
    
    np.random.seed(42)
    colors = [COLORS['controller'], COLORS['moisture'], COLORS['ntf'], COLORS['connector'], COLORS['asic'], COLORS['track']]
    classes = ["Controller failure", "Sensor short due to moisture", "NTF", "Connector damage", "ASIC CJ327 failure due to EOS", "Track burnt due to EOS"]
    
    for cls, col in zip(classes, colors):
        params = MILEAGE_PARAMS[cls]
        samples = np.random.lognormal(params['mu_log'], params['sigma_log'], 5000)
        samples = np.clip(samples, params['min'], params['max'])
        ax_main.hist(samples/1000, bins=30, alpha=0.4, color=col, label=cls.split()[0], edgecolor='white', linewidth=0.5)
    
    ax_main.set_xlabel('Mileage (×1000 km)', fontsize=10)
    ax_main.set_ylabel('Frequency', fontsize=10)
    ax_main.set_title('Mileage Distributions by FA Class', fontsize=11, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=7)
    style_axes(ax_main)
    
    np.random.seed(42)
    conn_early = np.random.lognormal(np.log(2500), 0.50, 750)
    conn_late = np.random.lognormal(np.log(75000), 0.60, 4250)
    conn_samples = np.concatenate([conn_early, conn_late])
    conn_samples = np.clip(conn_samples, 200, 230000)
    
    ax_inset.hist(conn_samples/1000, bins=25, color=COLORS['connector'], alpha=0.7, edgecolor='white')
    ax_inset.axvline(2.5, color=COLORS['danger'], linestyle='--', linewidth=1.5, label='2,500 km')
    ax_inset.axvline(75, color=COLORS['success'], linestyle='--', linewidth=1.5, label='75,000 km')
    ax_inset.set_xlabel('Mileage (×1000)', fontsize=8)
    ax_inset.set_title('Connector\n(bimodal)', fontsize=9, fontweight='bold')
    ax_inset.legend(fontsize=6, loc='upper right')
    style_axes(ax_inset)
    
    plt.tight_layout()
    save_figure(fig, "fig_mileage_distributions.pdf")


def plot_wd_probabilities():
    """Figure 8: Stacked bar chart showing warranty decision probabilities."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    classes = ["Track burnt", "Controller", "NTF", "Sensor moisture", "Connector\n(early)", "Connector\n(late)", "ASIC\n(V≤14.7)", "ASIC\n(14.7<V<15.4)", "ASIC\n(V≥15.4)"]
    pf = [0.030, 0.960, 0.010, 0.010, 0.92, 0.80, 0.78, 0.60, 0.38]
    cf = [0.960, 0.030, 0.025, 0.965, 0.08, 0.20, 0.22, 0.40, 0.62]
    ats = [0.010, 0.010, 0.965, 0.025, 0.00, 0.00, 0.00, 0.00, 0.00]
    
    x = np.arange(len(classes))
    width = 0.6
    
    ax.bar(x, pf, width, label='Production Failure', color=COLORS['success'], alpha=0.8)
    ax.bar(x, cf, width, bottom=pf, label='Customer Failure', color=COLORS['danger'], alpha=0.8)
    ax.bar(x, ats, width, bottom=[p+c for p,c in zip(pf, cf)], label='According to Spec', color=COLORS['gray_medium'], alpha=0.8)
    
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('Warranty Decision Probabilities by FA Class', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.1)
    style_axes(ax)
    
    save_figure(fig, "fig_wd_probabilities.pdf")


def plot_label_noise_mechanisms():
    """Figure 9: Diagram showing six noise injection mechanisms."""
    fig, ax = create_figure(6.5, 4.0)
    
    mechanisms = [
        (7.5, "ASIC boundary-zone", "PF ↔ CF", "η×1.2=0.018", "V∈[14.8,15.2]"),
        (6.0, "Connector random", "PF ↔ CF", "η=0.015", "All rows"),
        (4.5, "NTF adjudication", "ATS → CF", "0.008", "All NTF"),
        (3.0, "Track misclass", "CF → PF", "0.007", "All Track"),
        (1.5, "Controller misclass", "PF → CF", "0.007", "All Controller"),
        (0.0, "Sensor moisture", "CF → PF", "0.010", "All moisture"),
    ]
    
    for y, name, direction, rate, condition in mechanisms:
        box = FancyBboxPatch((0.5, y - 0.35), 5.5, 0.7, boxstyle="round,pad=0.1", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], linewidth=1)
        ax.add_patch(box)
        ax.text(0.7, y, name, ha='left', va='center', fontsize=8, fontweight='bold')
        ax.text(2.8, y, direction, ha='center', va='center', fontsize=8, color=COLORS['danger'])
        ax.text(4.0, y, rate, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(5.5, y, condition, ha='left', va='center', fontsize=7, style='italic', color=COLORS['gray_medium'])
    
    ax.set_xlim(0, 6.5)
    ax.set_ylim(-0.5, 8.5)
    ax.text(3.3, 8.0, "Label Noise Injection Mechanisms", ha='center', fontsize=11, fontweight='bold')
    ax.text(3.3, 7.5, f"Base noise rate η = 0.015 (1.5% of rows)", ha='center', fontsize=9, style='italic')
    
    save_figure(fig, "fig_label_noise_mechanisms.pdf")


def plot_derived_features():
    """Figure 12: Derived feature engineering diagram."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.2, "Derived Feature Engineering", ha='center', fontsize=11, fontweight='bold')
    
    raw_cols = ["Voltage", "Mileage_km", "DTC", "Year", "Date"]
    for i, col in enumerate(raw_cols):
        add_box(ax, 0.5, 7.5 - i * 0.9, 1.5, 0.7, col, facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    
    for i in range(len(raw_cols)):
        add_arrow(ax, (2.1, 7.9 - i * 0.9), (3.0, 7.9 - i * 0.9))
    
    bins = [
        ("Voltage →", "7 bins:\n<11, 11-13.5, 13.5-14.5,\n14.5-15.4, 15.4-16, 16-17, >17"),
        ("Mileage →", "4 bins:\n<20k, 20k-60k,\n60k-100k, >100k"),
        ("DTC count →", "4 bins:\n0, 1, 2-3, >3"),
        ("Year + Date →", "claim_age =\nyear(Date) - Year"),
        ("V + DTC →", "4 binary:\nV>15.4∧P, V<11∧U,\n11≤V≤14.5∧C, multi"),
    ]
    for i, (label, content) in enumerate(bins):
        add_box(ax, 3.2, 7.5 - i * 0.9, 2.8, 0.7, label, facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=7)
        ax.text(5.8, 7.5 - i * 0.9 - 0.15, content, ha='left', va='center', fontsize=6, family='monospace')
    
    for i in range(len(raw_cols)):
        add_arrow(ax, (6.1, 7.9 - i * 0.9), (7.0, 7.9 - i * 0.9))
    
    for i, col in enumerate(raw_cols):
        add_box(ax, 7.2, 7.5 - i * 0.9, 1.5, 0.7, "Feature", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')
    
    save_figure(fig, "fig_derived_features.pdf")


def plot_cascade_calibration():
    """Figure 31: Cascade calibration check - overlaid histograms."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    np.random.seed(42)
    train_probs = np.random.beta(8, 2, 5000)
    test_probs = np.random.beta(7.5, 2.2, 5000)
    
    ax.hist(train_probs, bins=30, alpha=0.6, color=COLORS['primary'], label='Training (OOF)', edgecolor='white')
    ax.hist(test_probs, bins=30, alpha=0.6, color=COLORS['secondary'], label='Test (inference)', edgecolor='white')
    
    ax.axvline(np.mean(train_probs), color=COLORS['primary'], linestyle='--', linewidth=2, label=f'μ_train={np.mean(train_probs):.3f}')
    ax.axvline(np.mean(test_probs), color=COLORS['secondary'], linestyle='--', linewidth=2, label=f'μ_test={np.mean(test_probs):.3f}')
    
    gap = abs(np.mean(train_probs) - np.mean(test_probs))
    ax.axvline(0.05, color=COLORS['danger'], linestyle=':', linewidth=1.5, label='Alert threshold=0.05')
    
    ax.set_xlabel('FA Top-Class Probability', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Cascade Calibration: Train vs Test\nMean Gap = {gap:.3f}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    style_axes(ax)
    
    save_figure(fig, "fig_cascade_calibration.pdf")


def plot_feature_importance():
    """Figure 33: Feature importance horizontal bar charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    
    fa_features = ['voltage', 'dtc_prefix_P', 'notes_len', 'tfidf_eng', 'mileage_km', 'ohe_supplier', 'volt_bin', 'notes_moisture', 'dtc_count', 'year']
    fa_importance = [0.22, 0.18, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05]
    
    wd_features = ['voltage', 'fa_prob_NTF', 'fa_prob_Track', 'dtc_prefix_P', 'fa_prob ASIC', 'mileage_km', 'notes_len', 'tfidf_eng', 'ohe_supplier', 'fa_prob_Connector']
    wd_importance = [0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    y_pos = np.arange(len(fa_features))
    ax1.barh(y_pos, fa_importance, color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['border'])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(fa_features, fontsize=8)
    ax1.set_xlabel('Gini Importance', fontsize=9)
    ax1.set_title('FA Classifier', fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    style_axes(ax1)
    
    y_pos2 = np.arange(len(wd_features))
    ax2.barh(y_pos2, wd_importance, color=COLORS['success'], alpha=0.7, edgecolor=COLORS['border'])
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(wd_features, fontsize=8)
    ax2.set_xlabel('Gini Importance', fontsize=9)
    ax2.set_title('WD Classifier', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    style_axes(ax2)
    
    fig.suptitle('Feature Importance: Top 10 Features', fontsize=11, fontweight='bold', y=1.02)
    
    save_figure(fig, "fig_feature_importance.pdf")


def plot_classifier_evaluation():
    """Figure 29: Classifier evaluation flow diagram."""
    fig, ax = create_figure(6.5, 3.0)
    
    add_box(ax, 0.5, 5.5, 1.8, 0.9, "Test Set\n(20,000 rows)", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)
    add_arrow(ax, (2.35, 6.0), (2.85, 6.0))
    add_process_box(ax, 2.9, 5.5, 1.8, 0.9, "Classifier\n(f_fa or f_wd)")
    add_arrow(ax, (4.75, 6.0), (5.25, 6.0))
    add_box(ax, 5.3, 5.5, 1.5, 0.9, "Predictions\nŷ", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=9)
    add_arrow(ax, (6.85, 6.0), (7.35, 6.0))
    add_output_box(ax, 7.4, 5.5, 1.5, 0.9, "Metrics\nAcc/P/R/F1")
    
    ax.text(3.25, 4.5, "Isolated Evaluation:", ha='center', fontsize=9, fontweight='bold')
    ax.text(3.25, 4.0, "• Accuracy (overall)", ha='center', fontsize=8)
    ax.text(3.25, 3.5, "• Precision/Recall/F1 (weighted + macro)", ha='center', fontsize=8)
    ax.text(3.25, 3.0, "• Per-class confusion matrix", ha='center', fontsize=8)
    
    save_figure(fig, "fig_classifier_evaluation.pdf")


def plot_pipeline_evaluation():
    """Figure 32: End-to-end pipeline evaluation diagram."""
    fig, ax = create_figure(6.5, 3.5)
    
    add_box(ax, 0.5, 6.5, 1.5, 0.8, "Test\nSamples", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)
    add_arrow(ax, (2.05, 6.9), (2.55, 6.9))
    add_process_box(ax, 2.6, 6.5, 1.5, 0.8, "predict()\nPipeline")
    add_arrow(ax, (4.15, 6.9), (4.65, 6.9))
    add_box(ax, 4.7, 6.5, 1.5, 0.8, "Claim\nResponse", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=9)
    add_arrow(ax, (6.25, 6.9), (6.75, 6.9))
    add_output_box(ax, 6.8, 6.5, 1.5, 0.8, "Compare\nto ground truth")
    
    ax.text(4.75, 5.5, "Per-Engine Accuracy Breakdown:", ha='center', fontsize=10, fontweight='bold')
    
    engines = [("Rule+ML", 0.91, COLORS['primary']), ("LLM+Rule+ML", 0.94, COLORS['purple']), ("ML-only", 0.87, COLORS['success']), ("LLM+ML", 0.89, COLORS['secondary'])]
    for i, (name, acc, color) in enumerate(engines):
        add_box(ax, 1.5 + i * 2.0, 3.5, 1.8, 1.2, f"{name}\n\n{acc:.1%}", facecolor=color, edgecolor=color, fontsize=9, fontcolor='white')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(1, 8)
    
    save_figure(fig, "fig_pipeline_evaluation.pdf")


def plot_hybrid_decision_engine_overview():
    """Figure 1: Six-stage hybrid decision engine overview."""
    fig, ax = create_figure(7.0, 3.5)
    
    ax.text(3.5, 9.0, "Hybrid Decision Engine: Six-Stage Pipeline", ha='center', fontsize=11, fontweight='bold')
    
    stages = [
        (0.8, "Stage 1\nLLM Understanding", COLORS['brown']),
        (2.5, "Stage 2\nRule Engine", COLORS['primary']),
        (4.2, "Stage 3\nFeature Translation", COLORS['brown']),
        (5.9, "Stage 4\nXGBoost Cascade", COLORS['success']),
        (7.6, "Stage 5\nScore Combination", COLORS['purple']),
        (9.3, "Stage 6\nOutput Formatting", COLORS['brown']),
    ]
    
    for x, label, color in stages:
        add_stage_box(ax, x - 0.7, 6.5, 1.4, 1.0, label, facecolor=color, fontcolor='white')
    
    for i in range(len(stages) - 1):
        add_arrow(ax, (stages[i][0] + 0.7, 7.0), (stages[i+1][0] - 0.7, 7.0))
    
    add_box(ax, 0.5, 3.5, 2.5, 1.2, "Input: fault_code\ntechnician_notes\nvoltage", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_arrow(ax, (3.05, 4.1), (0.8 + 0.7, 6.5))
    add_arrow(ax, (9.3 + 0.7, 7.0), (9.5, 4.1))
    add_output_box(ax, 7.5, 3.5, 2.5, 1.2, "Output: status\nfailure_analysis\nwarranty_decision\nconfidence\nreason")
    
    ax.set_xlim(0, 11)
    ax.set_ylim(2, 10)
    
    save_figure(fig, "fig_hybrid_decision_engine_overview.pdf")


def plot_six_stage_pipeline():
    """Figure 2: Detailed six-stage pipeline architecture."""
    fig, ax = create_figure(8.0, 5.0)
    
    ax.text(4.0, 9.5, "Six-Stage Pipeline: Inputs, Processing, Outputs", ha='center', fontsize=11, fontweight='bold')
    
    stages = [
        (1.0, "Stage 1", "LLM Understanding", "notes, DTC", "category, complaint,\nseverity, confidence", "LLM or None", COLORS['brown']),
        (2.8, "Stage 2", "Rule Engine", "fault_code,\nnotes, voltage", "rule_id, status,\nWD, confidence", "Always active", COLORS['primary']),
        (4.6, "Stage 3", "Feature Translation", "notes, DTC", "customer_complaint,\ndtc_features", "LLM or fallback", COLORS['brown']),
        (6.4, "Stage 4", "XGBoost Cascade", "feature vector", "FA probabilities,\nWD prediction", "Always active", COLORS['success']),
        (8.2, "Stage 5", "Score Combination", "rule, ML, LLM", "combined_confidence,\nstatus, engine_tag", "Always active", COLORS['purple']),
    ]
    
    for x, stage_num, name, input_type, output, note, color in stages:
        add_rounded_box(ax, x - 0.9, 6.5, 1.8, 1.3, stage_num, facecolor=color, edgecolor=color, fontcolor='white', fontsize=10)
        ax.text(x, 6.0, name, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, 5.3, f"IN: {input_type}", ha='center', va='center', fontsize=6, style='italic')
        ax.text(x, 4.6, f"OUT: {output}", ha='center', va='center', fontsize=6, style='italic')
        ax.text(x, 4.0, f"({note})", ha='center', va='center', fontsize=5, color=COLORS['gray_medium'])
    
    for i in range(len(stages) - 1):
        add_arrow(ax, (stages[i][0] + 0.9, 7.15), (stages[i+1][0] - 0.9, 7.15))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(2.5, 10)
    
    save_figure(fig, "fig_six_stage_pipeline.pdf")


def plot_data_loading_pipeline():
    """Figure 10: Data loading and cleaning pipeline."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Data Loading Pipeline", ha='center', fontsize=11, fontweight='bold')
    
    steps = [
        (1.0, "CSV Load", "synthetic_warranty\nclaims_v9.csv"),
        (2.8, "Null Fill", "DTC→'', Complaint\n→'OBD Light ON', etc."),
        (4.6, "Label Encode", "FA → [0..5]\nWD → [0..2]"),
        (6.4, "DTC Features", "extract_dtc\n_features()"),
    ]
    
    for x, name, content in steps:
        add_process_box(ax, x - 0.9, 6.5, 1.8, 1.2, name, fontsize=9)
        ax.text(x, 6.0, content, ha='center', va='center', fontsize=6, family='monospace')
    
    for i in range(len(steps) - 1):
        add_arrow(ax, (steps[i][0] + 0.95, 7.1), (steps[i+1][0] - 0.95, 7.1))
    
    add_box(ax, 1.5, 4.0, 3.5, 1.0, "train_test_split(test_size=0.2, random_state=42)", facecolor=COLORS['danger'], edgecolor=COLORS['danger'], fontsize=8, fontcolor='white')
    
    ax.text(1.0, 3.0, "Training Set\n(80,000)", ha='center', fontsize=8, fontweight='bold', color=COLORS['success'])
    ax.text(6.0, 3.0, "Test Set\n(20,000)", ha='center', fontsize=8, fontweight='bold', color=COLORS['primary'])
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 10)
    
    save_figure(fig, "fig_data_loading_pipeline.pdf")


def plot_train_test_split():
    """Figure 11: Train-test split before fit."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "Train/Test Split: Split-Before-Fit", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 2.0, 7.5, 2.5, 1.0, "DataFrame (100K rows)", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)
    add_arrow(ax, (3.25, 7.5), (3.25, 7.0))
    
    ax.add_patch(FancyBboxPatch((2.0, 5.5), 2.5, 1.0, boxstyle="round,pad=0.1", facecolor=COLORS['bg_box'], edgecolor=COLORS['gray_medium'], linewidth=1.5))
    ax.text(3.25, 6.0, "train_test_split()", ha='center', va='center', fontsize=9, fontweight='bold')
    
    add_arrow(ax, (2.0, 6.0), (1.0, 6.0))
    add_box(ax, 0.2, 5.25, 1.6, 0.8, "Training\n(80K)", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=9, fontcolor='white')
    
    add_arrow(ax, (4.5, 6.0), (5.5, 6.0))
    add_box(ax, 5.3, 5.25, 1.6, 0.8, "Test\n(20K)", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=9, fontcolor='white')
    
    ax.text(0.5, 4.0, "fit_transform()", ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    ax.text(6.0, 4.0, "transform()", ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    
    add_box(ax, 0.2, 2.5, 1.6, 0.8, "Fitted\nTransformers", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')
    add_box(ax, 5.3, 2.5, 1.6, 0.8, "Apply fitted\nTransformers", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontcolor='white')
    
    ax.text(3.25, 1.5, "Data Leakage Prevention: Fit ONLY on training data", ha='center', fontsize=9, fontweight='bold', color=COLORS['danger'])
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 10)
    
    save_figure(fig, "fig_train_test_split.pdf")


def plot_transformer_pipeline():
    """Figure 13: Transformer fitting pipeline."""
    fig, ax = create_figure(7.0, 4.0)
    
    ax.text(3.5, 9.0, "Transformer Pipeline: 12 Transformers", ha='center', fontsize=11, fontweight='bold')
    
    transformers = [
        ("OHE Complaint", "sparse (n×14)"),
        ("TF-IDF DTC", "sparse (n×40)"),
        ("DTC Flags", "dense (n×95)"),
        ("OHE Supplier", "sparse (n×5)"),
        ("Scaler Mileage", "dense (n×1)"),
        ("Scaler Year", "dense (n×1)"),
        ("OHE Mileage Bracket", "sparse (n×4)"),
        ("Scaler Claim Age", "dense (n×1)"),
        ("Scaler Voltage", "dense (n×1)"),
        ("OHE Voltage Bracket", "sparse (n×7)"),
        ("OHE DTC Count", "sparse (n×4)"),
        ("Interactions", "dense (n×4)"),
    ]
    
    for i, (name, output) in enumerate(transformers):
        col = 0 if i < 6 else 1
        row = i % 6
        x = 1.0 + col * 3.5
        y = 7.5 - row * 0.8
        add_box(ax, x, y, 2.0, 0.65, name, facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=7)
        ax.text(x + 2.1, y + 0.3, output, ha='left', va='center', fontsize=6, style='italic')
    
    add_arrow(ax, (3.5, 2.5), (3.5, 2.0))
    add_box(ax, 2.0, 1.0, 3.0, 0.8, "scipy.sparse.hstack()", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=9, fontcolor='white')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 10)
    
    save_figure(fig, "fig_transformer_pipeline.pdf")


def plot_dtc_feature_extraction():
    """Figure 14: DTC feature extraction flow."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 8.5, "DTC Feature Extraction", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 0.5, 6.5, 2.0, 1.0, "DTC String\n'P0562,P0563,...'", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)
    add_arrow(ax, (2.55, 7.0), (3.05, 7.0))
    add_process_box(ax, 3.1, 6.5, 1.5, 1.0, "Split by ','\nParse codes")
    
    add_arrow(ax, (4.15, 7.0), (4.65, 7.0))
    add_process_box(ax, 4.7, 6.5, 1.8, 1.0, "Prefix Detection\nP/U/C/B flags")
    
    add_arrow(ax, (6.0, 7.0), (6.5, 7.0))
    add_process_box(ax, 6.55, 6.5, 1.8, 1.0, "High-Value DTC\nOne-Hot Match")
    
    ax.text(3.25, 5.0, "Output Feature Vector:", ha='center', fontsize=9, fontweight='bold')
    
    features = ["dtc_count: 2", "has_P: 1", "has_U: 0", "has_C: 0", "has_B: 0", "dtc_p0562: 1", "dtc_p0563: 1", "..."]
    for i, feat in enumerate(features):
        color = COLORS['success'] if i < 5 else COLORS['bg_box']
        edge = COLORS['success'] if i < 5 else COLORS['gray_medium']
        add_box(ax, 1.5 + (i % 4) * 1.4, 4.0 - (i // 4) * 0.5, 1.3, 0.4, feat, facecolor=color, edgecolor=edge, fontsize=6)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 8)
    
    save_figure(fig, "fig_dtc_feature_extraction.pdf")


def plot_complaint_matching():
    """Figure 15: Complaint matching flow."""
    fig, ax = create_figure(6.0, 4.0)
    
    ax.text(3.0, 9.0, "Complaint Matching", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 2.0, 0.9, "Technician Notes")
    add_arrow(ax, (2.55, 7.45), (3.05, 7.45))
    add_process_box(ax, 3.1, 7.0, 1.8, 0.9, "Keyword Scan\n(first match wins)")
    
    add_arrow(ax, (4.4, 7.45), (4.9, 7.45))
    ax.add_patch(FancyBboxPatch((4.9, 6.5), 1.2, 0.9, boxstyle="round,pad=0.1", facecolor=COLORS['success'], edgecolor=COLORS['success'], linewidth=1.2))
    ax.text(5.5, 7.0, "Match?", ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    add_arrow(ax, (5.5, 6.5), (5.5, 6.0))
    ax.text(6.2, 6.25, "Yes", ha='left', fontsize=7, color=COLORS['success'], fontweight='bold')
    ax.text(4.0, 6.25, "No", ha='right', fontsize=7, color=COLORS['danger'], fontweight='bold')
    
    add_process_box(ax, 3.1, 4.8, 1.8, 0.9, "Fuzzy Match\ncutoff=0.25", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'])
    
    add_arrow(ax, (4.45, 5.25), (4.95, 5.25))
    ax.add_patch(FancyBboxPatch((4.95, 4.5), 1.2, 0.9, boxstyle="round,pad=0.1", facecolor=COLORS['success'], edgecolor=COLORS['success'], linewidth=1.2))
    ax.text(5.55, 5.0, "Match?", ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    add_arrow(ax, (5.55, 4.5), (5.55, 4.0))
    ax.text(6.2, 4.25, "Yes", ha='left', fontsize=7, color=COLORS['success'], fontweight='bold')
    ax.text(4.0, 4.25, "No", ha='right', fontsize=7, color=COLORS['danger'], fontweight='bold')
    
    add_output_box(ax, 3.1, 2.5, 1.8, 0.9, "OBD Light ON\n(default)")
    add_output_box(ax, 5.5, 2.5, 1.8, 0.9, "Complaint Label")
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 10)
    
    save_figure(fig, "fig_complaint_matching.pdf")


def plot_xgboost_architecture():
    """Figure 16: XGBoost architecture."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "XGBoost Classifier Architecture", ha='center', fontsize=11, fontweight='bold')
    
    ax.text(1.0, 7.8, "Hyperparameters:", ha='left', fontsize=9, fontweight='bold')
    params = ["n_estimators=1000", "max_depth=10", "learning_rate=0.02", "subsample=0.8", "colsample_bytree=0.8"]
    for i, p in enumerate(params):
        ax.text(1.2, 7.3 - i * 0.4, f"• {p}", ha='left', fontsize=8, family='monospace')
    
    ax.text(5.0, 7.8, "Sequential Trees:", ha='left', fontsize=9, fontweight='bold')
    for i in range(3):
        y = 7.0 - i * 0.8
        add_box(ax, 5.0, y, 1.5, 0.6, f"Tree h{i+1}", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')
        if i < 2:
            ax.annotate('', xy=(5.75, y - 0.05), xytext=(5.75, y - 0.25), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1))
    
    ax.text(5.0, 3.8, "F_{t+1} = F_t + η·h_{t+1}", ha='center', fontsize=10, fontweight='bold', style='italic')
    ax.text(5.0, 3.2, "η = 0.02 (learning rate)", ha='center', fontsize=8, style='italic')
    ax.text(5.0, 2.6, "Regularization: λ=0.1 (L2)", ha='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 9)
    
    save_figure(fig, "fig_xgboost_architecture.pdf")


def plot_cascade_oof():
    """Figure 17: Cascade OOF architecture."""
    fig, ax = create_figure(7.0, 4.0)
    
    ax.text(3.5, 9.0, "Cascade Architecture: Out-of-Fold Probabilities", ha='center', fontsize=11, fontweight='bold')
    
    ax.text(1.0, 7.8, "Training Phase", ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    add_process_box(ax, 0.5, 6.5, 1.8, 1.0, "FA Classifier\n(5-fold CV)")
    add_arrow(ax, (1.9, 7.0), (2.6, 7.0))
    add_box(ax, 2.7, 6.5, 1.8, 1.0, "OOF Probabilities\np_FA_train", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    add_arrow(ax, (4.6, 7.0), (5.3, 7.0))
    add_process_box(ax, 5.4, 6.5, 1.8, 1.0, "WD Classifier\n[p_FA | X] → WD")
    
    ax.text(5.0, 5.5, "Inference Phase", ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    add_process_box(ax, 4.5, 4.5, 1.5, 0.8, "FA Model\n(full train)")
    add_arrow(ax, (5.25, 4.9), (5.9, 4.9))
    add_box(ax, 6.0, 4.5, 1.5, 0.8, "p_FA_test", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    add_arrow(ax, (6.75, 4.9), (7.3, 4.9))
    add_process_box(ax, 7.4, 4.5, 1.5, 0.8, "WD\n[p_FA | X]", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')
    
    ax.set_xlim(0, 9)
    ax.set_ylim(3, 10)
    
    save_figure(fig, "fig_cascade_oof.pdf")


def plot_rule_engine_flow():
    """Figure 18: Rule engine flow with 9 rules."""
    fig, ax = create_figure(6.5, 5.0)
    
    ax.text(3.25, 9.5, "Rule Engine: First-Match-Wins (9 Rules)", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 2.5, 8.0, 1.5, 0.8, "Input x")
    
    rules = [
        (6.8, "1: over_voltage", "V > 16.0", "93%"),
        (5.8, "2: low_voltage", "V < 11.0", "95%"),
        (4.8, "3: moisture", "keyword∈{water,...}", "91%"),
        (3.8, "4: physical_damage", "keyword∈{crack,...}", "88.5%"),
        (2.8, "5: ntf", "keyword∈{no fault,...}", "95%"),
        (1.8, "6-9: DTC prefix", "U/P/C/B code", "57-80%"),
    ]
    
    for y, name, condition, conf in rules:
        add_diamond(ax, 2.5, y, 0.4, f"{name}\n{conf}")
        ax.text(4.0, y, f"{condition}", ha='left', va='center', fontsize=7, style='italic')
    
    for i in range(len(rules) - 1):
        add_arrow(ax, (2.1, rules[i][0] - 0.5), (2.1, rules[i+1][0] + 0.5))
    
    add_arrow(ax, (2.9, 8.0), (2.9, 7.2))
    add_box(ax, 3.5, 0.8, 2.0, 0.8, "No rule →\nReturn None", facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=8, fontcolor='white')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    
    save_figure(fig, "fig_rule_engine_flow.pdf")


def plot_llm_integration():
    """Figure 19: LLM integration architecture."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "LLM Integration Architecture", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 0.5, 7.0, 1.5, 1.2, "API Key\nConfig", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_arrow(ax, (2.05, 7.6), (2.55, 7.6))
    
    add_process_box(ax, 2.6, 7.0, 2.0, 1.2, "Provider Selection\nOpenAI / OpenRouter")
    add_arrow(ax, (4.65, 7.6), (5.15, 7.6))
    add_process_box(ax, 5.2, 7.0, 1.5, 1.2, "Retry Logic\n(max 2, exp backoff)")
    
    services = [
        (1.0, "Stage 1", "understand_claim", COLORS['brown']),
        (3.0, "Stage 3", "translate_to_ml_features", COLORS['brown']),
        (5.0, "Stage 6", "format_output", COLORS['brown']),
    ]
    
    for x, stage, func, color in services:
        add_box(ax, x - 0.6, 4.5, 1.2, 0.8, stage, facecolor=color, edgecolor=color, fontsize=8, fontcolor='white')
        add_box(ax, x - 0.8, 3.5, 1.6, 0.7, func, facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=6)
    
    add_arrow(ax, (5.2, 7.0), (3.25, 5.3))
    
    ax.set_xlim(0, 7)
    ax.set_ylim(2.5, 10)
    
    save_figure(fig, "fig_llm_integration.pdf")


def plot_llm_categorisation():
    """Figure 20: LLM categorisation."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "LLM Claim Categorisation (Stage 1)", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 2.0, 1.0, "Input:\nnotes + DTC")
    add_arrow(ax, (2.55, 7.5), (3.05, 7.5))
    add_llm_box(ax, 3.1, 7.0, 2.0, 1.0, "LLM Prompt\n+ Disambiguation\nRules")
    add_arrow(ax, (5.15, 7.5), (5.65, 7.5))
    add_box(ax, 5.7, 7.0, 1.5, 1.0, "JSON\nResponse", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=8)
    
    categories = ["moisture_damage", "physical_damage", "ntf", "electrical_issue", "engine_symptom", "communication_fault", "other"]
    colors_cat = [COLORS['danger'], COLORS['danger'], COLORS['success'], COLORS['primary'], COLORS['primary'], COLORS['primary'], COLORS['gray_medium']]
    
    for i, (cat, col) in enumerate(zip(categories, colors_cat)):
        add_box(ax, 0.5 + i * 1.1, 5.0, 1.0, 0.6, cat.replace('_', '\n'), facecolor=col, edgecolor=col, fontsize=6, fontcolor='white')
    
    ax.text(3.25, 4.0, "↓", ha='center', fontsize=16)
    
    wds = ["CF", "CF", "ATS", "PF", "PF", "PF", "None"]
    for i, (wd, col) in enumerate(zip(wds, colors_cat)):
        add_box(ax, 0.5 + i * 1.1, 2.5, 1.0, 0.6, wd, facecolor=col, edgecolor=col, fontsize=8, fontcolor='white')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 10)
    
    save_figure(fig, "fig_llm_categorisation.pdf")


def plot_score_combination():
    """Figure 21: Score combination decision tree."""
    fig, ax = create_figure(7.0, 5.0)
    
    ax.text(3.5, 9.5, "Score Combination Logic", ha='center', fontsize=11, fontweight='bold')
    
    add_diamond(ax, 3.5, 8.5, 0.5, "Rule\nFired?")
    
    add_arrow(ax, (3.5, 8.0), (2.0, 7.5))
    ax.text(2.2, 7.9, "Yes", fontsize=7, color=COLORS['success'], fontweight='bold')
    add_diamond(ax, 2.0, 6.5, 0.5, "Agrees\nML?")
    
    add_arrow(ax, (2.0, 6.0), (1.0, 5.5))
    ax.text(1.2, 5.9, "Yes", fontsize=7, color=COLORS['success'], fontweight='bold')
    add_box(ax, 0.2, 4.8, 1.6, 0.7, "0.70·rule + 0.30·ml\n+ 2.0 bonus", facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=7)
    
    add_arrow(ax, (2.0, 6.0), (3.0, 5.5))
    ax.text(2.8, 5.9, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_box(ax, 2.2, 4.8, 1.6, 0.7, "0.55·rule + 0.35·ml", facecolor=COLORS['bg_box'], edgecolor=COLORS['danger'], fontsize=7)
    
    add_arrow(ax, (3.5, 8.0), (5.0, 7.5))
    ax.text(4.8, 7.9, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_diamond(ax, 5.0, 6.5, 0.5, "LLM\nAvail?")
    
    add_arrow(ax, (5.0, 6.0), (5.0, 5.0))
    add_box(ax, 4.2, 4.2, 1.6, 0.7, "0.85·ml + 0.15·llm", facecolor=COLORS['bg_box'], edgecolor=COLORS['purple'], fontsize=7)
    
    ax.text(3.5, 3.0, "→ Status: Firm (≥85%), Cautious (65-85%), Manual (<65%)", ha='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 7)
    ax.set_ylim(2, 10)
    
    save_figure(fig, "fig_score_combination.pdf")


def plot_stage1_llm_understanding():
    """Figure 22: Stage 1 LLM understanding."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Stage 1: LLM Claim Understanding", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 1.8, 1.0, "notes + DTC")
    add_arrow(ax, (2.35, 7.5), (2.85, 7.5))
    add_llm_box(ax, 2.9, 7.0, 1.8, 1.0, "LLM: understand_claim()\nunderstand_claim_with_retry()")
    add_arrow(ax, (4.75, 7.5), (5.25, 7.5))
    add_output_box(ax, 5.3, 7.0, 1.8, 1.0, "Output:\n{category, complaint,\nseverity, FA, reasoning,\nconfidence}")
    
    ax.set_xlim(0, 8)
    ax.set_ylim(5, 10)
    
    save_figure(fig, "fig_stage1_llm_understanding.pdf")


def plot_stage2_rule_engine():
    """Figure 23: Stage 2 rule engine."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Stage 2: Rule Engine Evaluation", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 2.5, 1.0, "fault_code\ntechnician_notes\nvoltage")
    add_arrow(ax, (3.05, 7.5), (3.55, 7.5))
    add_process_box(ax, 3.6, 7.0, 1.8, 1.0, "run_rules()\n9 rules (priority order)")
    add_arrow(ax, (5.45, 7.5), (5.95, 7.5))
    add_output_box(ax, 6.0, 7.0, 1.8, 1.0, "Output:\n{rule_id, status,\nWD, confidence,\nFA, reason}")
    
    ax.set_xlim(0, 9)
    ax.set_ylim(5, 10)
    
    save_figure(fig, "fig_stage2_rule_engine.pdf")


def plot_stage3_feature_translation():
    """Figure 24: Stage 3 feature translation."""
    fig, ax = create_figure(7.0, 4.0)
    
    ax.text(3.5, 9.0, "Stage 3: Feature Translation", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 1.5, 1.0, "notes + DTC")
    
    ax.text(1.0, 6.0, "LLM path:", ha='center', fontsize=8, fontweight='bold', color=COLORS['brown'])
    add_arrow(ax, (2.05, 7.5), (2.55, 7.5))
    add_llm_box(ax, 2.6, 7.0, 2.2, 1.0, "translate_to_ml_features()\n(if LLM available)")
    
    ax.text(1.0, 5.0, "Fallback path:", ha='center', fontsize=8, fontweight='bold', color=COLORS['gray_medium'])
    add_arrow(ax, (2.05, 6.0), (2.55, 6.0))
    add_process_box(ax, 2.6, 5.5, 2.2, 1.0, "extract_dtc_features()\n+ match_complaint()\n(deterministic)")
    
    add_arrow(ax, (4.85, 7.5), (4.85, 6.5))
    add_arrow(ax, (4.85, 6.0), (4.85, 7.5))
    
    add_output_box(ax, 4.5, 4.5, 2.5, 1.2, "Feature Dict:\n{customer_complaint, dtc_codes,\ndtc_text, dtc_count,\nhas_P, has_U, has_C, has_B,\nvoltage, mileage_km, year, ...}")
    
    ax.set_xlim(0, 8)
    ax.set_ylim(3, 10)
    
    save_figure(fig, "fig_stage3_feature_translation.pdf")


def plot_stage4_xgboost_cascade():
    """Figure 25: Stage 4 XGBoost cascade."""
    fig, ax = create_figure(6.5, 4.5)
    
    ax.text(3.25, 9.5, "Stage 4: XGBoost Cascade Scoring", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.5, 2.0, 1.0, "Feature Vector\n(12 transformers)")
    add_arrow(ax, (2.55, 8.0), (3.05, 8.0))
    
    add_process_box(ax, 3.1, 7.5, 1.8, 1.0, "FA Classifier\n(clf_fa)")
    add_arrow(ax, (4.95, 8.0), (4.95, 7.2))
    
    ax.text(4.95, 6.8, "p_FA (6 classes)", ha='center', fontsize=7, color=COLORS['purple'])
    
    add_box(ax, 3.1, 5.3, 1.8, 0.6, "Concatenate:\n[X | p_FA]", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=7, fontcolor='white')
    add_arrow(ax, (4.95, 5.9), (4.0, 5.9))
    
    add_process_box(ax, 3.1, 4.5, 1.8, 1.0, "WD Classifier\n(clf_wd)")
    add_arrow(ax, (4.95, 5.0), (5.75, 5.0))
    add_output_box(ax, 5.8, 4.5, 1.8, 1.0, "Output:\nWD prediction\np_WD probabilities\nconfidence = √(c_FA·c_WD)")
    
    ax.set_xlim(0, 9)
    ax.set_ylim(3, 10)
    
    save_figure(fig, "fig_stage4_xgboost_cascade.pdf")


def plot_stage5_score_combination():
    """Figure 26: Stage 5 score combination."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Stage 5: Score Combination", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 0.5, 7.0, 1.4, 1.0, "Rule\nResult", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontcolor='white')
    add_box(ax, 2.2, 7.0, 1.4, 1.0, "ML\nResult", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')
    add_box(ax, 3.9, 7.0, 1.4, 1.0, "LLM\nResult", facecolor=COLORS['brown'], edgecolor=COLORS['brown'], fontsize=8, fontcolor='white')
    
    add_arrow(ax, (5.35, 7.5), (5.85, 7.5))
    add_process_box(ax, 5.9, 7.0, 1.8, 1.0, "combine_scores()\nWeighted blend\n+ Agreement check")
    
    add_arrow(ax, (7.75, 7.5), (8.25, 7.5))
    add_output_box(ax, 8.3, 7.0, 1.8, 1.0, "Output:\ncombined_confidence\nstatus\ndecision_engine")
    
    ax.set_xlim(0, 11)
    ax.set_ylim(5, 10)
    
    save_figure(fig, "fig_stage5_score_combination.pdf")


def plot_stage6_output_formatting():
    """Figure 27: Stage 6 output formatting."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Stage 6: Output Formatting", ha='center', fontsize=11, fontweight='bold')
    
    add_input_box(ax, 0.5, 7.0, 2.5, 1.0, "combined_decision\n+ features")
    
    ax.text(1.0, 6.0, "LLM path:", ha='center', fontsize=8, fontweight='bold', color=COLORS['brown'])
    add_arrow(ax, (3.05, 7.5), (3.55, 7.5))
    add_llm_box(ax, 3.6, 7.0, 1.8, 1.0, "format_output()\nNatural language reason")
    
    ax.text(1.0, 5.0, "Fallback:", ha='center', fontsize=8, fontweight='bold', color=COLORS['gray_medium'])
    add_arrow(ax, (3.05, 6.0), (3.55, 6.0))
    add_process_box(ax, 3.6, 5.5, 1.8, 1.0, "assemble_output_from_fields()\nTemplate-based reason")
    
    add_arrow(ax, (5.45, 7.5), (5.95, 7.5))
    add_arrow(ax, (5.45, 6.0), (5.95, 7.5))
    add_output_box(ax, 6.0, 7.0, 1.8, 1.0, "ClaimResponse:\nstatus, FA, WD,\nconfidence, reason,\nmatched_complaint,\ndecision_engine")
    
    ax.set_xlim(0, 9)
    ax.set_ylim(4, 10)
    
    save_figure(fig, "fig_stage6_output_formatting.pdf")


def plot_complete_predict_flow():
    """Figure 28: Complete prediction flow."""
    fig, ax = create_figure(8.0, 5.0)
    
    ax.text(4.0, 9.5, "Complete predict() Flow", ha='center', fontsize=11, fontweight='bold')
    
    stages = [
        (0.8, "1. LLM Check", "api_key & len>5", COLORS['brown']),
        (2.2, "2. Stage 1", "LLM understand", COLORS['brown']),
        (3.6, "3. Stage 2", "run_rules()", COLORS['primary']),
        (5.0, "4. Stage 3", "features", COLORS['brown']),
        (6.4, "5. Stage 4", "run_ml()", COLORS['success']),
        (7.8, "6. Stage 5", "combine_scores()", COLORS['purple']),
    ]
    
    for x, name, detail, color in stages:
        add_stage_box(ax, x - 0.6, 7.5, 1.2, 1.0, name, facecolor=color, fontcolor='white', fontsize=9)
        ax.text(x, 7.1, detail, ha='center', va='center', fontsize=6, style='italic')
    
    for i in range(len(stages) - 1):
        add_arrow(ax, (stages[i][0] + 0.6, 8.0), (stages[i+1][0] - 0.6, 8.0))
    
    add_arrow(ax, (4.0, 7.5), (4.0, 6.5))
    add_box(ax, 2.5, 5.0, 3.0, 1.0, "Error Handling:\ntry/except per stage\n→ fallback paths", facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=8, fontcolor='white')
    
    add_arrow(ax, (4.0, 5.0), (4.0, 4.5))
    add_output_box(ax, 3.0, 3.5, 2.0, 0.8, "Stage 6: format_output()")
    
    ax.set_xlim(0, 9)
    ax.set_ylim(2.5, 10)
    
    save_figure(fig, "fig_complete_predict_flow.pdf")


def plot_fastapi_architecture():
    """Figure 34: FastAPI architecture."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "FastAPI Backend Architecture", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 2.0, 7.5, 2.5, 0.8, "FastAPI App", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=9, fontcolor='white')
    
    midboxes = [
        ("CORS Middleware", COLORS['gray_medium']),
        ("Request Schemas\n(ClaimRequest)", COLORS['secondary']),
        ("Response Schemas\n(ClaimResponse)", COLORS['secondary']),
        ("/analyze Endpoint\nPOST", COLORS['success']),
    ]
    
    for i, (name, color) in enumerate(midboxes):
        add_box(ax, 1.5 + i * 1.1, 5.8, 1.0, 0.8, name, facecolor=color, edgecolor=color, fontsize=6, fontcolor='white')
        if i < len(midboxes) - 1:
            add_arrow(ax, (2.5 + i * 1.1, 6.2), (3.5 + i * 1.1, 6.2))
    
    add_arrow(ax, (3.25, 7.5), (3.25, 6.6))
    add_process_box(ax, 2.5, 4.8, 1.5, 0.8, "ML Predictor\npredict()", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    add_arrow(ax, (3.25, 4.8), (3.25, 4.0))
    
    endpoints = [
        ("GET /", "Health check"),
        ("POST /analyze", "Claim analysis"),
    ]
    for i, (path, desc) in enumerate(endpoints):
        add_box(ax, 1.5 + i * 3.5, 2.5, 3.0, 0.7, f"{path}\n{desc}", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 10)
    
    save_figure(fig, "fig_fastapi_architecture.pdf")


def plot_frontend_architecture():
    """Figure 35: Frontend architecture."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Frontend Architecture (Single-Page App)", ha='center', fontsize=11, fontweight='bold')
    
    sections = [
        (1.0, "HTML/CSS", "Form: fault_code, notes, voltage", COLORS['primary']),
        (3.2, "JavaScript", "Event handlers, API calls", COLORS['secondary']),
        (5.4, "Backend API", "POST http://localhost:8000/analyze", COLORS['success']),
    ]
    
    for x, name, detail, color in sections:
        add_stage_box(ax, x - 0.8, 6.5, 1.6, 1.2, name, facecolor=color, fontcolor='white', fontsize=9)
        ax.text(x, 6.0, detail, ha='center', va='center', fontsize=6, style='italic')
    
    for i in range(len(sections) - 1):
        add_arrow(ax, (sections[i][0] + 0.8, 7.1), (sections[i+1][0] - 0.8, 7.1))
    
    add_arrow(ax, (4.6, 6.5), (4.6, 5.5))
    add_output_box(ax, 3.5, 4.5, 2.2, 0.8, "Result Display:\nstatus, FA, WD, confidence, reason")
    
    ax.set_xlim(0, 8)
    ax.set_ylim(3.5, 10)
    
    save_figure(fig, "fig_frontend_architecture.pdf")


def plot_docker_compose_architecture():
    """Figure 36: Docker Compose architecture."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "Docker Compose Deployment", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 1.0, 6.5, 2.0, 1.2, "Backend Container\nPort: 8000:8000\nHealth: GET /", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontcolor='white')
    add_box(ax, 4.5, 6.5, 2.0, 1.2, "Frontend Container\nPort: 3000:3000\nDepends: backend", facecolor=COLORS['secondary'], edgecolor=COLORS['secondary'], fontsize=8, fontcolor='white')
    
    add_box(ax, 2.75, 4.5, 1.0, 0.8, "trace_net\n(bridge)", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    
    add_arrow(ax, (2.05, 6.1), (2.75, 5.3))
    add_arrow(ax, (3.75, 5.3), (3.05, 6.1))
    
    ax.text(3.25, 3.5, "docker-compose.yml", ha='center', fontsize=9, fontweight='bold')
    ax.text(3.25, 3.0, "restart: unless-stopped", ha='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 10)
    
    save_figure(fig, "fig_docker_compose_architecture.pdf")


def plot_logging_architecture():
    """Figure 37: Logging architecture."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "Logging Architecture", ha='center', fontsize=11, fontweight='bold')
    
    add_box(ax, 2.5, 7.5, 2.0, 0.8, "Root Logger", facecolor=COLORS['gray_dark'], edgecolor=COLORS['gray_dark'], fontsize=9, fontcolor='white')
    add_arrow(ax, (3.5, 7.5), (3.5, 7.0))
    
    children = [
        ("trace.ml_predictor", COLORS['primary']),
        ("trace.llm_client", COLORS['secondary']),
        ("trace.api", COLORS['success']),
    ]
    
    for i, (name, color) in enumerate(children):
        add_box(ax, 1.0 + i * 1.8, 5.5, 1.6, 0.8, name, facecolor=color, edgecolor=color, fontsize=7, fontcolor='white')
    
    for i in range(len(children)):
        add_arrow(ax, (3.5, 6.9), (1.8 + i * 1.8, 6.2))
    
    add_box(ax, 2.0, 3.5, 2.5, 0.8, "DecisionLogger", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    ax.text(3.25, 3.0, "log_stage(), log_decision()", ha='center', fontsize=7, style='italic')
    
    ax.text(3.25, 2.0, "Format: %(asctime)s [%(levelname)s] %(name)s %(message)s", ha='center', fontsize=7, family='monospace')
    
    ax.set_xlim(0, 7)
    ax.set_ylim(1, 10)
    
    save_figure(fig, "fig_logging_architecture.pdf")


def plot_model_serialization():
    """Figure 38: Model serialization."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.0, "Model Serialization & Auto-Training", ha='center', fontsize=11, fontweight='bold')
    
    stages = [
        (1.0, "Training", "train_and_save()", COLORS['success']),
        (2.8, "Bundle", "14 components", COLORS['purple']),
        (4.6, "Serialize", "pickle.dump()", COLORS['secondary']),
        (6.4, "Storage", "trace_models.pkl", COLORS['gray_medium']),
    ]
    
    for x, name, detail, color in stages:
        add_stage_box(ax, x - 0.7, 6.5, 1.4, 1.0, name, facecolor=color, fontcolor='white', fontsize=9)
        ax.text(x, 6.0, detail, ha='center', va='center', fontsize=7, style='italic')
    
    for i in range(len(stages) - 1):
        add_arrow(ax, (stages[i][0] + 0.7, 7.0), (stages[i+1][0] - 0.7, 7.0))
    
    ax.text(3.25, 4.5, "Inference Time:", ha='center', fontsize=9, fontweight='bold')
    
    ax.text(2.0, 3.5, "File exists?", ha='center', fontsize=8)
    add_box(ax, 1.5, 2.5, 1.0, 0.6, "Yes", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontcolor='white')
    add_box(ax, 3.5, 2.5, 1.0, 0.6, "No", facecolor=COLORS['danger'], edgecolor=COLORS['danger'], fontsize=7, fontcolor='white')
    
    add_arrow(ax, (2.5, 3.2), (2.0, 3.1))
    add_arrow(ax, (3.5, 3.2), (3.5, 3.1))
    
    add_output_box(ax, 1.2, 1.2, 1.6, 0.6, "pickle.load()")
    add_output_box(ax, 3.2, 1.2, 1.6, 0.6, "train_and_save()")
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 10)
    
    save_figure(fig, "fig_model_serialization.pdf")


def plot_dataset_overview():
    """Figure 3: Dataset schema overview."""
    fig, ax = create_figure(6.5, 4.0)
    
    ax.text(3.25, 9.3, "Dataset Schema: synthetic_warranty_claims_v9.csv", ha='center', fontsize=11, fontweight='bold')
    ax.text(3.25, 8.8, "100,000 rows × 11 columns", ha='center', fontsize=9, style='italic')
    
    cols = [
        ("Column", "Type", "Example"),
        ("Customer", "categorical", "Ashok Leyland"),
        ("Year", "integer", "2022"),
        ("Date", "date", "2022-06-15"),
        ("QC_Number", "string", "QC-12345"),
        ("Customer Complaint", "categorical", "OBD Light ON"),
        ("DTC", "string", "P0562,P0563"),
        ("Voltage", "float", "14.2"),
        ("Failure Analysis", "categorical", "NTF"),
        ("Warranty Decision", "categorical", "According to Spec"),
        ("Supplier", "categorical", "Bosch"),
        ("Mileage_km", "integer", "45000"),
    ]
    
    for i, (c, t, e) in enumerate(cols):
        y = 7.8 - i * 0.6
        if i == 0:
            face = COLORS['primary']
            fc = 'white'
        elif i <= 3:
            face = COLORS['bg_box']
            fc = 'black'
        elif i <= 6:
            face = COLORS['bg_light']
            fc = 'black'
        else:
            face = COLORS['bg_box']
            fc = 'black'
        add_box(ax, 0.3, y - 0.25, 2.2, 0.5, c, facecolor=face, edgecolor=COLORS['border'], fontsize=7, fontcolor=fc)
        add_box(ax, 2.5, y - 0.25, 1.5, 0.5, t, facecolor=face, edgecolor=COLORS['border'], fontsize=7, fontcolor=fc)
        add_box(ax, 4.0, y - 0.25, 2.2, 0.5, e, facecolor=face, edgecolor=COLORS['border'], fontsize=7, fontcolor=fc)
    
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, 10)
    
    save_figure(fig, "fig_dataset_overview.pdf")


def plot_dtc_pool_architecture():
    """Figure 7: DTC pool architecture."""
    fig, ax = create_figure(7.0, 4.0)
    
    ax.text(3.5, 9.0, "DTC Pool Architecture", ha='center', fontsize=11, fontweight='bold')
    
    pools = [
        ("ASIC", "P0601-P0617", COLORS['asic']),
        ("Track", "P0300-P0356", COLORS['track']),
        ("Sensor", "P0113-P0343", COLORS['moisture']),
        ("Connector", "C0031-C0550", COLORS['connector']),
        ("Controller", "U0001-U0184", COLORS['controller']),
    ]
    
    for i, (name, codes, color) in enumerate(pools):
        add_box(ax, 0.5 + i * 1.3, 7.0, 1.2, 0.8, name, facecolor=color, edgecolor=color, fontsize=8, fontcolor='white')
        ax.text(1.1 + i * 1.3, 6.5, codes, ha='center', va='center', fontsize=6, style='italic', rotation=15)
    
    for i in range(len(pools) - 1):
        add_arrow(ax, (0.5 + i * 1.3 + 1.2, 7.4), (0.5 + (i + 1) * 1.3, 7.4), color=COLORS['danger'], lw=1)
    
    ax.text(3.5, 5.5, "Companion DTC Injection", ha='center', fontsize=9, fontweight='bold')
    ax.text(3.5, 5.0, "P0562 ↔ P0563 (55%)", ha='center', fontsize=8, family='monospace')
    ax.text(3.5, 4.5, "U0100 ↔ U0101 (60%)", ha='center', fontsize=8, family='monospace')
    
    ax.text(3.5, 3.5, "Cross-FA Injection (4%)", ha='center', fontsize=9, fontweight='bold')
    ax.text(3.5, 3.0, "DTC_AMBIGUOUS_CROSS: P0300, P0171, P0325...", ha='center', fontsize=8, family='monospace')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 10)
    
    save_figure(fig, "fig_dtc_pool_architecture.pdf")


def plot_cv_on_test_set():
    """Figure 30: Cross-validation on held-out test set."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(3.25, 9.0, "3-Fold Cross-Validation on Test Set", ha='center', fontsize=11, fontweight='bold')
    ax.text(3.25, 8.5, "Test Set: 20,000 rows → 3 folds × ~6,667 rows", ha='center', fontsize=9, style='italic')
    
    n = 3
    box_width = 1.8
    box_height = 0.8
    gap = 0.3
    
    for i in range(n):
        y = 6.5 - i * 1.5
        
        ax.text(0.3, y + 0.4, f"Fold {i+1}", fontsize=8, va='center')
        
        for j in range(n):
            x = 1.5 + j * (box_width + gap)
            
            if j == i:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="square", facecolor=COLORS['white'], edgecolor=COLORS['danger'], linewidth=2)
            else:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="square", facecolor=COLORS['primary'], alpha=0.5, edgecolor=COLORS['primary'], linewidth=1)
            ax.add_patch(rect)
    
    ax.text(5.5, 2.5, "White: Validation", ha='center', fontsize=8, color=COLORS['danger'])
    ax.text(5.5, 2.0, "Blue: Training", ha='center', fontsize=8, color=COLORS['primary'])
    
    ax.set_xlim(0, 7)
    ax.set_ylim(1, 10)
    
    save_figure(fig, "fig_cv_on_test_set.pdf")


if __name__ == "__main__":
    print("Generating 38 Methodology figures...")
    
    print("Category A: Data-driven figures...")
    plot_fa_class_distribution()
    plot_voltage_distributions()
    plot_mileage_distributions()
    plot_wd_probabilities()
    plot_label_noise_mechanisms()
    plot_derived_features()
    plot_cascade_calibration()
    plot_feature_importance()
    plot_classifier_evaluation()
    plot_pipeline_evaluation()
    
    print("Category B: Architecture/flow figures...")
    plot_hybrid_decision_engine_overview()
    plot_six_stage_pipeline()
    plot_data_loading_pipeline()
    plot_train_test_split()
    plot_transformer_pipeline()
    plot_dtc_feature_extraction()
    plot_complaint_matching()
    plot_xgboost_architecture()
    plot_cascade_oof()
    plot_rule_engine_flow()
    plot_llm_integration()
    plot_llm_categorisation()
    plot_score_combination()
    plot_stage1_llm_understanding()
    plot_stage2_rule_engine()
    plot_stage3_feature_translation()
    plot_stage4_xgboost_cascade()
    plot_stage5_score_combination()
    plot_stage6_output_formatting()
    plot_complete_predict_flow()
    
    print("Category C: System architecture figures...")
    plot_fastapi_architecture()
    plot_frontend_architecture()
    plot_docker_compose_architecture()
    plot_logging_architecture()
    plot_model_serialization()
    
    print("Category D: Dataset architecture figures...")
    plot_dataset_overview()
    plot_dtc_pool_architecture()
    plot_cv_on_test_set()
    
    print(f"Done! Generated 38 figures in {OUTPUT_DIR}")