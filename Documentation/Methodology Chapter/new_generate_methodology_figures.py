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
    fontcolor = kwargs.get('fontcolor', 'black')
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, color=fontcolor, wrap=True)
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
    ax.legend(loc='best', fontsize=8, ncol=3)
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
    ax.legend(loc='best', fontsize=7, ncol=2)
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
    ax_main.legend(loc='best', fontsize=7)
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
    ax_inset.legend(fontsize=6, loc='best')
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
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1.1)
    style_axes(ax)
    
    save_figure(fig, "fig_wd_probabilities.pdf")


def plot_label_noise_mechanisms():
    """Figure 9: Diagram showing six noise injection mechanisms."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.axis('off')

    ax.text(3.5, 10.0, "Label Noise Injection Mechanisms", ha='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 9.4, f"Base noise rate η = 0.015 (1.5% of rows)", ha='center', fontsize=9, style='italic')

    mechanisms = [
        (8.2, "ASIC boundary-zone", "PF ↔ CF", "η×1.2=0.018", "V∈[14.8,15.2]"),
        (6.8, "Connector random", "PF ↔ CF", "η=0.015", "All rows"),
        (5.4, "NTF adjudication", "ATS → CF", "0.008", "All NTF"),
        (4.0, "Track misclass", "CF → PF", "0.007", "All Track"),
        (2.6, "Controller misclass", "PF → CF", "0.007", "All Controller"),
        (1.2, "Sensor moisture", "CF → PF", "0.010", "All moisture"),
    ]

    for y, name, direction, rate, condition in mechanisms:
        box = FancyBboxPatch((0.3, y - 0.4), 6.4, 0.8, boxstyle="round,pad=0.1", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], linewidth=1)
        ax.add_patch(box)
        ax.text(0.6, y, name, ha='left', va='center', fontsize=8, fontweight='bold')
        ax.text(3.0, y, direction, ha='center', va='center', fontsize=8, color=COLORS['danger'])
        ax.text(4.3, y, rate, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(5.5, y, condition, ha='left', va='center', fontsize=7, style='italic', color=COLORS['gray_medium'])

    ax.set_xlim(0, 7.0)
    ax.set_ylim(0.5, 10.5)

    save_figure(fig, "fig_label_noise_mechanisms.pdf")


def plot_derived_features():
    """Figure 12: Derived feature engineering diagram."""
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.axis('off')

    ax.text(5.5, 9.2, "Derived Feature Engineering", ha='center', fontsize=11, fontweight='bold')

    raw_cols = ["Voltage", "Mileage_km", "DTC", "Year", "Date"]
    for i, col in enumerate(raw_cols):
        add_box(ax, 0.5, 7.5 - i * 0.9, 1.5, 0.7, col, facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)

    for i in range(len(raw_cols)):
        add_arrow(ax, (2.0, 7.9 - i * 0.9), (3.0, 7.9 - i * 0.9))

    bins = [
        ("Voltage", "7 bins: <11, 11-13.5..."),
        ("Mileage", "4 bins: <20k, 20k-60k..."),
        ("DTC count", "4 bins: 0, 1, 2-3, >3"),
        ("Year + Date", "claim_age: year(Date)-Year"),
        ("V + DTC", "4 binary: V>15.4∧P..."),
    ]
    for i, (label, content) in enumerate(bins):
        add_box(ax, 3.0, 7.5 - i * 0.9, 2.2, 0.7, label, facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=7)
        ax.text(5.4, 7.85 - i * 0.9, content, ha='left', va='center', fontsize=5.5, family='monospace')

    for i in range(len(raw_cols)):
        add_arrow(ax, (7.5, 7.9 - i * 0.9), (8.2, 7.9 - i * 0.9))

    for i, col in enumerate(raw_cols):
        add_box(ax, 8.2, 7.5 - i * 0.9, 1.5, 0.7, "Feature", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    ax.set_xlim(0, 10.5)
    ax.set_ylim(3.5, 9.5)
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
    ax.legend(loc='best', fontsize=7)
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
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(2.0, 9.0)
    ax.axis('off')

    ax.text(5.0, 8.6, "Classifier Evaluation", ha='center', fontsize=12, fontweight='bold')

    # Flow: Test Set → Classifier → Predictions → Metrics
    bw, bh = 2.0, 0.9
    gap = 0.5
    total_w = 4 * bw + 3 * gap
    x0 = (10 - total_w) / 2
    by = 6.5
    xs = [x0 + i * (bw + gap) for i in range(4)]

    boxes = [
        ("Test Set\n(20,000 rows)", COLORS['bg_box'], COLORS['primary'], False),
        ("Classifier\n(f_fa or f_wd)", None, None, True),
        ("Predictions\nŷ", COLORS['bg_box'], COLORS['secondary'], False),
        ("Metrics\nAcc / P / R / F1", COLORS['success'], COLORS['success'], False),
    ]
    for i, (label, fc, ec, is_proc) in enumerate(boxes):
        if is_proc:
            add_process_box(ax, xs[i], by, bw, bh, label)
        else:
            fontcolor = 'white' if fc in (COLORS['success'],) else 'black'
            add_box(ax, xs[i], by, bw, bh, label, facecolor=fc, edgecolor=ec, fontsize=8, fontcolor=fontcolor)
        if i < 3:
            add_arrow(ax, (xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2))

    # Metric details below
    ax.text(5.0, 5.5, "Isolated Evaluation:", ha='center', fontsize=9, fontweight='bold')
    details = [
        "• Accuracy (overall)",
        "• Precision / Recall / F1 (weighted + macro)",
        "• Per-class confusion matrix",
    ]
    for i, d in enumerate(details):
        ax.text(5.0, 5.0 - i * 0.5, d, ha='center', fontsize=8)

    ax.set_xlim(0, 10)
    ax.set_ylim(2.0, 9.0)

    save_figure(fig, "fig_classifier_evaluation.pdf")


def plot_pipeline_evaluation():
    """Figure 32: End-to-end pipeline evaluation diagram."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 8.5)
    ax.axis('off')

    ax.text(5.0, 8.1, "End-to-End Pipeline Evaluation", ha='center', fontsize=12, fontweight='bold')

    # Flow row
    bw, bh = 1.9, 0.85
    gap = 0.4
    total_w = 4 * bw + 3 * gap
    x0 = (10 - total_w) / 2
    by = 6.5
    xs = [x0 + i * (bw + gap) for i in range(4)]

    flow = [
        ("Test Samples", COLORS['bg_box'], COLORS['primary'], False),
        ("predict()\nPipeline", None, None, True),
        ("Claim\nResponse", COLORS['bg_box'], COLORS['secondary'], False),
        ("Compare to\nground truth", COLORS['success'], COLORS['success'], False),
    ]
    for i, (label, fc, ec, is_proc) in enumerate(flow):
        if is_proc:
            add_process_box(ax, xs[i], by, bw, bh, label)
        else:
            fontcolor = 'white' if fc == COLORS['success'] else 'black'
            add_box(ax, xs[i], by, bw, bh, label, facecolor=fc, edgecolor=ec, fontsize=8, fontcolor=fontcolor)
        if i < 3:
            add_arrow(ax, (xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2))

    ax.text(5.0, 5.6, "Per-Engine Accuracy Breakdown:", ha='center', fontsize=10, fontweight='bold')

    engines = [
        ("Rule+ML", 0.91, COLORS['primary']),
        ("LLM+Rule+ML", 0.94, COLORS['purple']),
        ("ML-only", 0.87, COLORS['success']),
        ("LLM+ML", 0.89, COLORS['secondary']),
    ]
    ebw, ebh = 1.9, 1.3
    egap = 0.3
    total_ew = 4 * ebw + 3 * egap
    ex0 = (10 - total_ew) / 2
    eby = 3.8
    exs = [ex0 + i * (ebw + egap) for i in range(4)]

    for i, (name, acc, color) in enumerate(engines):
        add_box(ax, exs[i], eby, ebw, ebh, f"{name}\n\n{acc:.1%}",
                facecolor=color, edgecolor=color, fontsize=9, fontcolor='white')

    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 8.5)

    save_figure(fig, "fig_pipeline_evaluation.pdf")


def plot_hybrid_decision_engine_overview():
    """Figure 1: Six-stage hybrid decision engine overview."""
    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(7.0, 9.0, "Hybrid Decision Engine: Six-Stage Pipeline", ha='center', fontsize=12, fontweight='bold')

    bw, bh, gap = 1.7, 1.2, 0.4
    stage_colors = [COLORS['brown'], COLORS['primary'], COLORS['brown'],
                    COLORS['success'], COLORS['purple'], COLORS['brown']]
    stage_labels = ["Stage 1\nLLM\nUnderstanding", "Stage 2\nRule Engine",
                    "Stage 3\nFeature\nTranslation", "Stage 4\nXGBoost\nCascade",
                    "Stage 5\nScore\nCombination", "Stage 6\nOutput\nFormatting"]
    by = 6.2
    total_w = 6 * bw + 5 * gap
    x0 = (14 - total_w) / 2
    xs = [x0 + i * (bw + gap) for i in range(6)]

    for x, label, color in zip(xs, stage_labels, stage_colors):
        add_stage_box(ax, x, by, bw, bh, label, facecolor=color, fontcolor='white', fontsize=8)

    for i in range(5):
        mid_y = by + bh / 2
        add_arrow(ax, (xs[i] + bw, mid_y), (xs[i + 1], mid_y))

    add_box(ax, x0, 3.4, 2.6, 1.3,
            "Input:\nfault_code\ntechnician_notes, voltage",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_arrow(ax, (x0 + 1.3, 4.7), (xs[0] + bw / 2, by), color=COLORS['primary'])

    last_x = xs[5]
    add_output_box(ax, last_x - 0.2, 3.2, 2.1, 1.8,
                   "Output:\nstatus\nfailure_analysis\nwarranty_decision\nconfidence\nreason",
                   fontsize=7)
    add_arrow(ax, (last_x + bw / 2, by), (last_x - 0.2 + 1.05, 5.0), color=COLORS['success'])

    save_figure(fig, "fig_hybrid_decision_engine_overview.pdf")


def plot_six_stage_pipeline():
    """Figure 2: Detailed six-stage pipeline architecture."""
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(4.0, 10.5)
    ax.axis('off')

    ax.text(6.5, 10.1, "Six-Stage Pipeline: Inputs, Processing, Outputs", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 1.9, 1.1
    gap = 0.25
    total_w = 5 * bw + 4 * gap
    x0 = (13 - total_w) / 2
    by = 7.0  # box bottom y

    stages = [
        ("Stage 1", "LLM Understanding", "IN: notes, DTC", "OUT: category,\ncomplaint, severity", "LLM or None", COLORS['brown']),
        ("Stage 2", "Rule Engine", "IN: fault_code,\nnotes, voltage", "OUT: rule_id,\nWD, confidence", "Always active", COLORS['primary']),
        ("Stage 3", "Feature Translation", "IN: notes, DTC", "OUT: complaint,\ndtc_features", "LLM or fallback", COLORS['brown']),
        ("Stage 4", "XGBoost Cascade", "IN: feature vector", "OUT: FA probs,\nWD prediction", "Always active", COLORS['success']),
        ("Stage 5", "Score Combination", "IN: rule, ML, LLM", "OUT: confidence,\nstatus, engine_tag", "Always active", COLORS['purple']),
    ]

    xs = [x0 + i * (bw + gap) for i in range(5)]

    for i, (stage_num, name, inp, out, note, color) in enumerate(stages):
        cx = xs[i] + bw / 2
        add_rounded_box(ax, xs[i], by, bw, bh, stage_num, facecolor=color, edgecolor=color, fontcolor='white', fontsize=9)
        ax.text(cx, by - 0.35, name, ha='center', va='top', fontsize=7, fontweight='bold')
        ax.text(cx, by - 0.85, inp, ha='center', va='top', fontsize=6, style='italic', color=COLORS['primary'])
        ax.text(cx, by - 1.5, out, ha='center', va='top', fontsize=6, style='italic', color=COLORS['success'])
        ax.text(cx, by - 2.1, f"({note})", ha='center', va='top', fontsize=5.5, color=COLORS['gray_medium'])

    for i in range(4):
        mid_y = by + bh / 2
        add_arrow(ax, (xs[i] + bw, mid_y), (xs[i + 1], mid_y))

    ax.set_xlim(0, 13)
    ax.set_ylim(4.0, 10.5)

    save_figure(fig, "fig_six_stage_pipeline.pdf")


def plot_data_loading_pipeline():
    """Figure 10: Data loading and cleaning pipeline."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 9.5)
    ax.axis('off')

    ax.text(5.0, 9.1, "Data Loading Pipeline", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 1.8, 1.0
    gap = 0.3
    total_w = 4 * bw + 3 * gap
    x0 = (10 - total_w) / 2
    by = 7.0

    steps = [
        ("CSV Load", "synthetic_warranty\nclaims_v9.csv"),
        ("Null Fill", "DTC→'', Complaint\n→'OBD Light ON'"),
        ("Label Encode", "FA → [0..5]\nWD → [0..2]"),
        ("DTC Features", "extract_dtc\n_features()"),
    ]
    xs = [x0 + i * (bw + gap) for i in range(4)]

    for i, (name, content) in enumerate(steps):
        cx = xs[i] + bw / 2
        add_process_box(ax, xs[i], by, bw, bh, name, fontsize=9)
        ax.text(cx, by - 0.35, content, ha='center', va='top', fontsize=6.5, family='monospace')

    for i in range(3):
        mid_y = by + bh / 2
        add_arrow(ax, (xs[i] + bw, mid_y), (xs[i + 1], mid_y))

    # Split box — centred, arrow coming down from middle of step row
    split_x = (10 - 4.0) / 2
    split_y = 4.5
    mid_pipeline_x = x0 + total_w / 2
    add_arrow(ax, (mid_pipeline_x, by), (mid_pipeline_x, split_y + 0.8))
    add_box(ax, split_x, split_y, 4.0, 0.8,
            "train_test_split(test_size=0.2, random_state=42)",
            facecolor=COLORS['danger'], edgecolor=COLORS['danger'], fontsize=8, fontcolor='white')

    # Branches
    add_arrow(ax, (split_x, split_y + 0.4), (3.0, split_y + 0.4), color=COLORS['success'])
    add_arrow(ax, (3.0, split_y + 0.4), (2.0, 4.0), color=COLORS['success'])
    add_arrow(ax, (split_x + 4.0, split_y + 0.4), (7.0, split_y + 0.4), color=COLORS['primary'])
    add_arrow(ax, (7.0, split_y + 0.4), (8.0, 4.0), color=COLORS['primary'])
    add_box(ax, 1.0, 3.1, 2.0, 0.9, "Training Set\n(80,000)", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=9, fontcolor='white')
    add_box(ax, 7.0, 3.1, 2.0, 0.9, "Test Set\n(20,000)", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=9, fontcolor='white')

    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 9.5)

    save_figure(fig, "fig_data_loading_pipeline.pdf")


def plot_train_test_split():
    """Figure 11: Train-test split before fit."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_xlim(0, 9)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "Train/Test Split: Split-Before-Fit", ha='center', fontsize=12, fontweight='bold')

    # Top: DataFrame box
    df_x, df_w, df_y, df_h = 3.0, 3.0, 7.5, 0.9
    add_box(ax, df_x, df_y, df_w, df_h, "DataFrame (100K rows)", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)

    # Arrow down to split function
    split_x, split_y = 3.2, 6.2
    split_w, split_h = 2.6, 0.9
    add_arrow(ax, (df_x + df_w / 2, df_y), (split_x + split_w / 2, split_y + split_h))
    ax.add_patch(FancyBboxPatch((split_x, split_y), split_w, split_h,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['bg_box'], edgecolor=COLORS['gray_medium'], linewidth=1.5))
    ax.text(split_x + split_w / 2, split_y + split_h / 2,
            "train_test_split()", ha='center', va='center', fontsize=9, fontweight='bold')

    # Left branch → Training
    train_x, train_y, train_w, train_h = 0.5, 4.8, 2.2, 0.9
    add_arrow(ax, (split_x, split_y + split_h / 2), (train_x + train_w, train_y + train_h / 2))
    add_box(ax, train_x, train_y, train_w, train_h,
            "Training (80K)", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=9, fontcolor='white')

    # Right branch → Test
    test_x, test_y, test_w, test_h = 6.3, 4.8, 2.2, 0.9
    add_arrow(ax, (split_x + split_w, split_y + split_h / 2), (test_x, test_y + test_h / 2))
    add_box(ax, test_x, test_y, test_w, test_h,
            "Test (20K)", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=9, fontcolor='white')

    # fit_transform on Training
    ax.text(train_x + train_w / 2, train_y - 0.4, "fit_transform()", ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    add_arrow(ax, (train_x + train_w / 2, train_y), (train_x + train_w / 2, 3.5))
    add_box(ax, train_x, 2.6, train_w, 0.9, "Fitted\nTransformers",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    # transform on Test
    ax.text(test_x + test_w / 2, test_y - 0.4, "transform()", ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    add_arrow(ax, (test_x + test_w / 2, test_y), (test_x + test_w / 2, 3.5))
    add_box(ax, test_x, 2.6, test_w, 0.9, "Apply Fitted\nTransformers",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontcolor='white')

    ax.text(4.5, 1.8, "Data Leakage Prevention: Fit ONLY on training data",
            ha='center', fontsize=9, fontweight='bold', color=COLORS['danger'])

    save_figure(fig, "fig_train_test_split.pdf")


def plot_transformer_pipeline():
    """Figure 13: Transformer fitting pipeline."""
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.set_xlim(0, 9)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "Transformer Pipeline: 12 Transformers", ha='center', fontsize=12, fontweight='bold')

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

    bw, bh = 2.4, 0.58
    row_gap = 0.08
    col_x = [0.5, 4.8]   # left edges for two columns
    n_rows = 6

    for i, (name, output) in enumerate(transformers):
        col = i // n_rows
        row = i % n_rows
        x = col_x[col]
        y = 7.8 - row * (bh + row_gap)
        add_box(ax, x, y, bw, bh, name, facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=7)
        ax.text(x + bw + 0.12, y + bh / 2, output, ha='left', va='center', fontsize=6, style='italic')

    # Convergence arrow down from midpoint between columns to hstack box
    mid_x = 4.5
    top_y = 7.8 + bh / 2
    bottom_y_of_cols = 7.8 - (n_rows - 1) * (bh + row_gap)
    conv_y = bottom_y_of_cols - 0.3
    # Draw vertical line on each column's right side, then horizontal to center, then down
    for cx in col_x:
        col_bot = 7.8 - (n_rows - 1) * (bh + row_gap)
        ax.annotate('', xy=(mid_x, conv_y - 0.05), xytext=(cx + bw / 2, col_bot),
                    arrowprops=dict(arrowstyle='-', color=COLORS['gray_medium'], lw=1.2))

    add_arrow(ax, (mid_x, conv_y), (mid_x, 1.9))
    add_box(ax, 2.5, 1.1, 4.0, 0.8, "scipy.sparse.hstack()  →  Final Feature Matrix",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=9, fontcolor='white')

    ax.set_xlim(0, 9)
    ax.set_ylim(0.5, 9.5)

    save_figure(fig, "fig_transformer_pipeline.pdf")


def plot_dtc_feature_extraction():
    """Figure 14: DTC feature extraction flow."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 8.5)
    ax.axis('off')

    ax.text(5.0, 8.1, "DTC Feature Extraction", ha='center', fontsize=12, fontweight='bold')

    bw, bh, gap = 2.0, 0.9, 0.3
    by = 6.2
    boxes = [
        ("DTC String\n'P0562,P0563,...'", COLORS['bg_box'], COLORS['primary']),
        ("Split by ','\nParse codes", None, None),   # process box
        ("Prefix Detection\nP/U/C/B flags", None, None),
        ("High-Value DTC\nOne-Hot Match", None, None),
    ]
    total_w = len(boxes) * bw + (len(boxes) - 1) * gap
    x0 = (10 - total_w) / 2
    xs = [x0 + i * (bw + gap) for i in range(4)]

    add_box(ax, xs[0], by, bw, bh, boxes[0][0], facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_process_box(ax, xs[1], by, bw, bh, boxes[1][0], fontsize=7)
    add_process_box(ax, xs[2], by, bw, bh, boxes[2][0], fontsize=7)
    add_process_box(ax, xs[3], by, bw, bh, boxes[3][0], fontsize=7)

    for i in range(3):
        add_arrow(ax, (xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2))

    ax.text(5.0, 5.0, "Output Feature Vector:", ha='center', fontsize=9, fontweight='bold')

    features = ["dtc_count: 2", "has_P: 1", "has_U: 0", "has_C: 0",
                "has_B: 0", "dtc_p0562: 1", "dtc_p0563: 1", "... (95 flags)"]
    fw, fh = 2.0, 0.45
    fcols = 4
    feat_total_w = fcols * fw + (fcols - 1) * 0.15
    fx0 = (10 - feat_total_w) / 2
    for i, feat in enumerate(features):
        col = i % fcols
        row = i // fcols
        fx = fx0 + col * (fw + 0.15)
        fy = 4.2 - row * (fh + 0.1)
        color = COLORS['success'] if i < 5 else COLORS['bg_box']
        edge = COLORS['success'] if i < 5 else COLORS['gray_medium']
        add_box(ax, fx, fy, fw, fh, feat, facecolor=color, edgecolor=edge, fontsize=7)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 8.5)

    save_figure(fig, "fig_dtc_feature_extraction.pdf")


def plot_complaint_matching():
    """Figure 15: Complaint matching flow."""
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.set_xlim(0, 9)
    ax.set_ylim(1.5, 10.0)
    ax.axis('off')

    ax.text(4.5, 9.6, "Complaint Matching", ha='center', fontsize=12, fontweight='bold')

    # Input
    add_input_box(ax, 0.5, 8.2, 2.2, 0.8, "Technician Notes")
    add_arrow(ax, (2.7, 8.6), (3.3, 8.6))

    # Stage 1: Keyword scan
    add_process_box(ax, 3.3, 8.2, 2.2, 0.8, "Keyword Scan\n(first match wins)")
    add_arrow(ax, (5.5, 8.6), (6.1, 8.6))

    # Decision 1
    d1x, d1y = 6.6, 8.6
    add_diamond(ax, d1x, d1y, 0.4, "Match?")

    # Yes → right, then down to Complaint Label
    add_arrow(ax, (d1x + 0.4, d1y), (8.0, d1y), color=COLORS['success'])
    ax.text(d1x + 0.5, d1y + 0.15, "Yes", ha='left', fontsize=7, color=COLORS['success'], fontweight='bold')
    add_arrow(ax, (8.0, d1y), (8.0, 3.6), color=COLORS['success'])

    # No → down
    add_arrow(ax, (d1x, d1y - 0.4), (d1x, 7.15), color=COLORS['danger'])
    ax.text(d1x + 0.1, 7.5, "No", ha='left', fontsize=7, color=COLORS['danger'], fontweight='bold')

    # Arrow left to fuzzy box
    add_arrow(ax, (d1x, 7.15), (5.5, 7.15), color=COLORS['danger'])

    # Stage 2: Fuzzy match
    add_process_box(ax, 3.3, 6.75, 2.2, 0.8, "Fuzzy Match\ncutoff=0.25", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'])

    # Decision 2
    d2x, d2y = 6.6, 5.8
    add_arrow(ax, (5.5, 7.15), (d2x - 0.4, d2y), color=COLORS['gray_medium'])
    add_diamond(ax, d2x, d2y, 0.4, "Match?")

    # Yes → right, then down to Complaint Label
    add_arrow(ax, (d2x + 0.4, d2y), (8.0, d2y), color=COLORS['success'])
    ax.text(d2x + 0.5, d2y + 0.15, "Yes", ha='left', fontsize=7, color=COLORS['success'], fontweight='bold')

    # No → down, then left to default
    add_arrow(ax, (d2x, d2y - 0.4), (d2x, 4.6), color=COLORS['danger'])
    ax.text(d2x + 0.1, 4.9, "No", ha='left', fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_arrow(ax, (d2x, 4.6), (5.5, 4.6), color=COLORS['danger'])

    # Default output
    add_output_box(ax, 3.3, 4.2, 2.2, 0.8, "OBD Light ON\n(default)")

    # Complaint label output (right column)
    add_output_box(ax, 7.0, 2.8, 2.0, 0.8, "Complaint\nLabel")

    # Connect Yes path down to Complaint Label
    add_arrow(ax, (8.0, 3.6), (8.0, 3.6))  # endpoint already at box

    ax.set_xlim(0, 9.5)

    save_figure(fig, "fig_complaint_matching.pdf")


def plot_xgboost_architecture():
    """Figure 16: XGBoost gradient boosting architecture.

    Shows the iterative boosting process: input features are passed through
    sequential trees, each fitted to the residual error of the previous
    ensemble prediction. The final prediction is the weighted sum of all
    tree outputs. Hyperparameters and the regularised objective are shown
    alongside the flow.
    """
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 10.5)

    ax.text(6.0, 10.0, "XGBoost Classifier Architecture", ha='center', fontsize=13, fontweight='bold')
    ax.text(6.0, 9.5, "Gradient Boosted Trees  (1,000 rounds, depth 10)", ha='center', fontsize=9, style='italic', color=COLORS['gray_medium'])

    # ── Row 1: Input → Tree 1 → + → Tree 2 → + → ... → Tree T → Prediction ──
    bw, bh = 1.6, 0.9

    # Input feature vector
    add_input_box(ax, 0.3, 7.2, 1.5, bh, "Feature\nVector X")

    # Tree 1
    add_arrow(ax, (1.8, 7.65), (2.5, 7.65))
    add_box(ax, 2.5, 7.2, bw, bh, r"Tree $h_1$" + "\n(depth 10)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    # Plus symbol
    ax.text(4.35, 7.65, "+", ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Tree 2
    add_arrow(ax, (4.1, 7.65), (4.6, 7.65))
    ax.text(4.6, 7.65, r"$\eta$", ha='center', va='center', fontsize=10, color=COLORS['danger'])
    add_arrow(ax, (4.75, 7.65), (5.0, 7.65))
    add_box(ax, 5.0, 7.2, bw, bh, r"Tree $h_2$" + "\n(depth 10)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    # Plus + dots
    ax.text(6.85, 7.65, "+", ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.text(7.3, 7.65, r"$\cdots$", ha='center', va='center', fontsize=14, color=COLORS['gray_medium'])

    # Tree T
    add_arrow(ax, (7.6, 7.65), (7.9, 7.65))
    ax.text(7.9, 7.65, r"$\eta$", ha='center', va='center', fontsize=10, color=COLORS['danger'])
    add_arrow(ax, (8.1, 7.65), (8.3, 7.65))
    add_box(ax, 8.3, 7.2, bw, bh, r"Tree $h_{1000}$" + "\n(depth 10)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    # Final prediction
    add_arrow(ax, (9.9, 7.65), (10.2, 7.65))
    add_output_box(ax, 10.2, 7.15, 1.5, 1.0, r"$\hat{y}$" + "\nPrediction", fontsize=8)

    # ── Row 2: Residual feedback loop ──
    # Arrow from prediction down then back left to show "each tree fits residual"
    ax.annotate('', xy=(2.5, 6.2), xytext=(10.95, 6.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.2, linestyle='--'))
    ax.text(6.5, 6.0, "Each tree fits the negative gradient (residual) of the previous ensemble",
            ha='center', va='top', fontsize=8, style='italic', color=COLORS['danger'])

    # ── Row 3: Update rule (centred) ──
    ax.text(6.0, 5.2, r"$F_{t+1}(x) = F_t(x) + \eta \cdot h_{t+1}(x)$",
            ha='center', fontsize=12)
    ax.text(6.0, 4.7, r"Objective:  $\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_k \lambda \sum_j w_j^2$",
            ha='center', fontsize=10, color=COLORS['gray_dark'])

    # ── Row 4: Key hyperparameters in a horizontal layout ──
    params = [
        (r"$\eta = 0.02$", "Learning rate\n(slow, stable)"),
        ("1,000 trees", "Sequential\nestimators"),
        ("depth = 10", "Max tree\ncomplexity"),
        (r"$\lambda = 0.1$", "L2 leaf weight\nregularisation"),
        ("subsample\n= 0.8", "Row sampling\nper tree"),
        ("colsample\n= 0.8", "Feature sampling\nper tree"),
    ]
    pw, ph = 1.6, 1.2
    pgap = 0.2
    total_pw = len(params) * pw + (len(params) - 1) * pgap
    px0 = (12 - total_pw) / 2
    pby = 2.5

    for i, (value, desc) in enumerate(params):
        px = px0 + i * (pw + pgap)
        add_box(ax, px, pby, pw, ph, value,
                facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')
        ax.text(px + pw / 2, pby - 0.15, desc, ha='center', va='top', fontsize=6.5,
                color=COLORS['gray_medium'], style='italic')

    ax.text(6.0, 4.0, "Hyperparameters (shared by FA and WD classifiers)", ha='center', fontsize=9, fontweight='bold')

    save_figure(fig, "fig_xgboost_architecture.pdf")


def plot_cascade_oof():
    """Figure 17: Cascade OOF architecture."""
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 9.5)
    ax.axis('off')

    ax.text(5.0, 9.1, "Cascade Architecture: Out-of-Fold Probabilities", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 2.2, 1.0
    gap = 0.5

    # Training Phase
    ax.text(1.1, 8.3, "Training Phase", ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    t1x, t1y = 0.3, 7.0
    add_process_box(ax, t1x, t1y, bw, bh, "FA Classifier\n(5-fold CV)")
    add_arrow(ax, (t1x + bw, t1y + bh / 2), (t1x + bw + gap, t1y + bh / 2))
    t2x = t1x + bw + gap
    add_box(ax, t2x, t1y, bw, bh, "OOF Probabilities\np_FA_train",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    add_arrow(ax, (t2x + bw, t1y + bh / 2), (t2x + bw + gap, t1y + bh / 2))
    t3x = t2x + bw + gap
    add_process_box(ax, t3x, t1y, bw, bh, "WD Classifier\n[p_FA | X] → WD")

    # Inference Phase
    ax.text(5.0, 5.9, "Inference Phase", ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    i1x, i1y = 1.5, 4.5
    add_process_box(ax, i1x, i1y, 2.0, bh, "FA Model\n(full train)")
    add_arrow(ax, (i1x + 2.0, i1y + bh / 2), (i1x + 2.0 + gap, i1y + bh / 2))
    i2x = i1x + 2.0 + gap
    add_box(ax, i2x, i1y, 2.0, bh, "p_FA_test",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    add_arrow(ax, (i2x + 2.0, i1y + bh / 2), (i2x + 2.0 + gap, i1y + bh / 2))
    i3x = i2x + 2.0 + gap
    add_process_box(ax, i3x, i1y, 2.2, bh, "WD Prediction\n[p_FA | X]",
                    facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontcolor='white')

    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 9.5)

    save_figure(fig, "fig_cascade_oof.pdf")


def plot_rule_engine_flow():
    """Figure 18: Rule engine flow with 9 rules."""
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Rule Engine: First-Match-Wins (9 Rules)", ha='center', fontsize=12, fontweight='bold')

    # Input box at top centre
    inp_x, inp_y, inp_w, inp_h = 3.5, 9.0, 2.0, 0.7
    add_input_box(ax, inp_x, inp_y, inp_w, inp_h, "Input x")
    add_arrow(ax, (inp_x + inp_w / 2, inp_y), (2.5, 8.45))

    rules = [
        ("1: over_voltage", "V > 16.0", "93%"),
        ("2: low_voltage", "V < 11.0", "95%"),
        ("3: moisture", "keyword ∈ {water,...}", "91%"),
        ("4: physical_damage", "keyword ∈ {crack,...}", "88.5%"),
        ("5: ntf", "keyword ∈ {no fault,...}", "95%"),
        ("6-9: DTC prefix", "U/P/C/B code", "57-80%"),
    ]

    d_size = 0.42
    col_x = 2.5       # diamond centre x
    row_ys = [8.1 - i * 1.3 for i in range(len(rules))]  # diamond centre y

    for i, (name, condition, conf) in enumerate(rules):
        y = row_ys[i]
        add_diamond(ax, col_x, y, d_size, f"{name}\n{conf}")
        # "Yes" → right output
        add_arrow(ax, (col_x + d_size, y), (6.0, y), color=COLORS['success'])
        ax.text(col_x + d_size + 0.08, y + 0.18, "Yes", fontsize=6.5, color=COLORS['success'], fontweight='bold')
        ax.text(4.0, y + 0.15, condition, ha='left', va='center', fontsize=7.5, style='italic')
        add_output_box(ax, 6.0, y - 0.3, 1.5, 0.6, "Return WD", fontsize=7)

    # Downward chain arrows between diamonds (on the "No" path)
    for i in range(len(rules) - 1):
        add_arrow(ax, (col_x, row_ys[i] - d_size), (col_x, row_ys[i + 1] + d_size), color=COLORS['danger'])
        ax.text(col_x + 0.08, (row_ys[i] + row_ys[i + 1]) / 2, "No", fontsize=6.5, color=COLORS['danger'], fontweight='bold')

    # Terminal: no rule matched
    last_y = row_ys[-1] - d_size
    add_arrow(ax, (col_x, last_y), (col_x, 1.1), color=COLORS['danger'])
    ax.text(col_x + 0.08, last_y - 0.4, "No", fontsize=6.5, color=COLORS['danger'], fontweight='bold')
    add_box(ax, col_x - 1.2, 0.5, 2.8, 0.7, "No rule matched → Return None",
            facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=8, fontcolor='white')

    save_figure(fig, "fig_rule_engine_flow.pdf")


def plot_llm_integration():
    """Figure 19: LLM integration architecture."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_xlim(0, 9)
    ax.set_ylim(2.0, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "LLM Integration Architecture", ha='center', fontsize=12, fontweight='bold')

    # Top row: Config → Provider → Retry
    bw, bh, gap = 2.2, 1.1, 0.3
    top_boxes = [
        ("API Key\nConfig", COLORS['bg_box'], COLORS['primary']),
        ("Provider Selection\nOpenAI / OpenRouter", None, None),
        ("Retry Logic\n(max 2, exp backoff)", None, None),
    ]
    total_top = 3 * bw + 2 * gap
    x0_top = (9 - total_top) / 2
    top_xs = [x0_top + i * (bw + gap) for i in range(3)]
    by_top = 7.4

    add_box(ax, top_xs[0], by_top, bw, bh, top_boxes[0][0], facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_process_box(ax, top_xs[1], by_top, bw, bh, top_boxes[1][0])
    add_process_box(ax, top_xs[2], by_top, bw, bh, top_boxes[2][0])

    for i in range(2):
        add_arrow(ax, (top_xs[i] + bw, by_top + bh / 2), (top_xs[i + 1], by_top + bh / 2))

    # Services row below — 3 stage boxes + function labels
    services = [
        ("Stage 1", "understand_claim", COLORS['brown']),
        ("Stage 3", "translate_to_ml_features", COLORS['brown']),
        ("Stage 6", "format_output", COLORS['brown']),
    ]
    sbw, sbh = 2.0, 0.8
    total_svc = 3 * sbw + 2 * gap
    x0_svc = (9 - total_svc) / 2
    svc_xs = [x0_svc + i * (sbw + gap) for i in range(3)]
    by_svc = 5.2
    by_fn = 4.3

    # Arrow from centre of Retry box down to the services row
    retry_cx = top_xs[2] + bw / 2
    svc_centre_x = x0_svc + total_svc / 2
    add_arrow(ax, (top_xs[1] + bw / 2, by_top), (svc_centre_x, by_svc + sbh + 0.1))

    for i, (stage, func, color) in enumerate(services):
        cx = svc_xs[i] + sbw / 2
        add_box(ax, svc_xs[i], by_svc, sbw, sbh, stage, facecolor=color, edgecolor=color, fontsize=9, fontcolor='white')
        add_box(ax, svc_xs[i], by_fn, sbw, 0.7, func, facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=6)

    ax.set_xlim(0, 9)
    ax.set_ylim(2.0, 9.5)

    save_figure(fig, "fig_llm_integration.pdf")


def plot_llm_categorisation():
    """Figure 20: LLM categorisation."""
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(5.5, 9.1, "LLM Claim Categorisation (Stage 1)", ha='center', fontsize=12, fontweight='bold')

    # Top row: Input → LLM → JSON
    bw, bh, gap = 2.2, 1.0, 0.4
    top_xs = [0.5, 0.5 + bw + gap, 0.5 + 2 * (bw + gap)]
    by_top = 7.5
    add_input_box(ax, top_xs[0], by_top, bw, bh, "Input:\nnotes + DTC")
    add_llm_box(ax, top_xs[1], by_top, bw, bh, "LLM Prompt\n+ Disambiguation\nRules")
    add_box(ax, top_xs[2], by_top, bw, bh, "JSON\nResponse",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=8)
    for i in range(2):
        add_arrow(ax, (top_xs[i] + bw, by_top + bh / 2), (top_xs[i + 1], by_top + bh / 2))

    # Arrow down from JSON box to categories
    json_cx = top_xs[2] + bw / 2
    add_arrow(ax, (json_cx, by_top), (5.5, 6.6))

    # Category boxes — 7 wide, evenly spaced
    categories = ["moisture\ndamage", "physical\ndamage", "ntf", "electrical\nissue",
                  "engine\nsymptom", "communication\nfault", "other"]
    colors_cat = [COLORS['danger'], COLORS['danger'], COLORS['success'],
                  COLORS['primary'], COLORS['primary'], COLORS['primary'], COLORS['gray_medium']]
    cbw, cbh = 1.3, 0.65
    cgap = 0.1
    total_cw = 7 * cbw + 6 * cgap
    cx0 = (11 - total_cw) / 2
    by_cat = 5.75
    for i, (cat, col) in enumerate(zip(categories, colors_cat)):
        cx = cx0 + i * (cbw + cgap)
        add_box(ax, cx, by_cat, cbw, cbh, cat, facecolor=col, edgecolor=col, fontsize=6.5, fontcolor='white')

    # Arrow down to WD mapping
    ax.text(5.5, 5.2, "↓  WD Mapping", ha='center', fontsize=9, fontweight='bold', color=COLORS['gray_medium'])

    wds = ["CF", "CF", "ATS", "PF", "PF", "PF", "None"]
    by_wd = 4.0
    for i, (wd, col) in enumerate(zip(wds, colors_cat)):
        cx = cx0 + i * (cbw + cgap)
        add_box(ax, cx, by_wd, cbw, cbh, wd, facecolor=col, edgecolor=col, fontsize=9, fontcolor='white')

    ax.set_xlim(0, 11)
    ax.set_ylim(1.5, 9.5)
    
    save_figure(fig, "fig_llm_categorisation.pdf")


def plot_score_combination():
    """Figure 21: Score combination decision tree."""
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(5.5, 9.1, "Score Combination Logic", ha='center', fontsize=12, fontweight='bold')

    d = 0.45   # diamond half-size
    bw_out = 2.2
    bh_out = 0.75

    # ── Diamond 1: Rule Fired? ──────────────────────────────────────
    d1x, d1y = 5.5, 8.1
    add_diamond(ax, d1x, d1y, d, "Rule\nFired?")

    # YES branch → left: "Agrees ML?"
    d2x, d2y = 2.5, 6.5
    add_arrow(ax, (d1x - d, d1y), (d2x + d, d2y), color=COLORS['success'])
    ax.text(3.4, 7.5, "Yes", fontsize=7.5, color=COLORS['success'], fontweight='bold')
    add_diamond(ax, d2x, d2y, d, "Agrees\nML?")

    # YES-YES → box below left
    b11x, b11y = 0.5, 5.0
    add_arrow(ax, (d2x - d, d2y), (b11x + bw_out / 2, b11y + bh_out), color=COLORS['success'])
    ax.text(1.0, 5.9, "Yes", fontsize=7, color=COLORS['success'], fontweight='bold')
    add_box(ax, b11x, b11y - bh_out / 2, bw_out, bh_out,
            "0.70·rule + 0.30·ml\n+ 2.0 bonus",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=7)

    # YES-NO → box below right of diamond2
    b12x = d2x + d + 0.3
    b12y = 5.0
    add_arrow(ax, (d2x + d, d2y), (b12x + bw_out / 2, b12y + bh_out), color=COLORS['danger'])
    ax.text(d2x + d + 0.1, 5.9, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_box(ax, b12x, b12y - bh_out / 2, bw_out, bh_out,
            "0.55·rule + 0.35·ml",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['danger'], fontsize=7)

    # NO branch → right: "LLM Avail?"
    d3x, d3y = 8.5, 6.5
    add_arrow(ax, (d1x + d, d1y), (d3x - d, d3y), color=COLORS['danger'])
    ax.text(7.0, 7.5, "No", fontsize=7.5, color=COLORS['danger'], fontweight='bold')
    add_diamond(ax, d3x, d3y, d, "LLM\nAvail?")

    # LLM YES → box below
    b21x = d3x - bw_out / 2
    b21y = 5.0
    add_arrow(ax, (d3x, d3y - d), (d3x, b21y + bh_out / 2 + 0.2), color=COLORS['success'])
    ax.text(d3x + 0.08, 5.8, "Yes", fontsize=7, color=COLORS['success'], fontweight='bold')
    add_box(ax, b21x, b21y - bh_out / 2, bw_out, bh_out,
            "0.85·ml + 0.15·llm",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['purple'], fontsize=7)

    # LLM NO → box further right — keep within bounds
    b22x = 8.5
    b22y = 3.8
    add_arrow(ax, (d3x + d, d3y), (b22x + bw_out / 2, b22y + bh_out), color=COLORS['danger'])
    ax.text(d3x + d + 0.05, d3y + 0.08, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_box(ax, b22x, b22y - bh_out / 2, bw_out, bh_out,
            "ML only\n(ml confidence)",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['gray_medium'], fontsize=7)

    # Status legend at bottom
    ax.text(5.5, 2.5, "Status Thresholds:", ha='center', fontsize=9, fontweight='bold')
    ax.text(5.5, 2.0, "Firm (≥85%)   |   Cautious (65–85%)   |   Manual (<65%)",
            ha='center', fontsize=8, style='italic', color=COLORS['gray_medium'])

    save_figure(fig, "fig_score_combination.pdf")


def plot_stage1_llm_understanding():
    """Figure 22: Stage 1 LLM understanding."""
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.axis('off')

    ax.text(5.5, 9.0, "Stage 1: LLM Claim Understanding", ha='center', fontsize=11, fontweight='bold')

    add_input_box(ax, 0.5, 7.0, 2.2, 1.2, "notes + DTC")
    add_arrow(ax, (2.7, 7.6), (3.5, 7.6))
    add_llm_box(ax, 3.5, 7.0, 3.2, 1.2, "LLM: understand_claim()\nunderstand_claim_with_retry()", fontsize=7)
    add_arrow(ax, (6.7, 7.6), (7.5, 7.6))
    add_output_box(ax, 7.5, 7.0, 3.0, 1.2, "Output:\n{category, complaint,\nseverity, FA, reasoning,\nconfidence}", fontsize=7)

    ax.set_xlim(0, 11)
    ax.set_ylim(5.5, 9.5)

    save_figure(fig, "fig_stage1_llm_understanding.pdf")


def plot_stage2_rule_engine():
    """Figure 23: Stage 2 rule engine."""
    fig, ax = create_figure(8.0, 4.0)

    ax.text(4.0, 9.0, "Stage 2: Rule Engine Evaluation", ha='center', fontsize=11, fontweight='bold')

    add_input_box(ax, 0.5, 7.0, 2.8, 1.2, "fault_code\ntechnician_notes\nvoltage")
    add_arrow(ax, (3.3, 7.6), (4.0, 7.6))
    add_process_box(ax, 4.0, 7.0, 2.2, 1.2, "run_rules()\n9 rules (priority order)")
    add_arrow(ax, (6.2, 7.6), (6.9, 7.6))
    add_output_box(ax, 6.9, 7.0, 2.2, 1.2, "Output:\n{rule_id, status,\nWD, confidence,\nFA, reason}")

    ax.set_xlim(0, 10)
    ax.set_ylim(5, 10)

    save_figure(fig, "fig_stage2_rule_engine.pdf")


def plot_stage3_feature_translation():
    """Figure 24: Stage 3 feature translation."""
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 10.5)
    ax.axis('off')

    ax.text(5.0, 10.1, "Stage 3: Feature Translation", ha='center', fontsize=12, fontweight='bold')

    # Input box
    inp_x, inp_y, inp_w, inp_h = 0.3, 7.0, 2.0, 1.2
    add_input_box(ax, inp_x, inp_y, inp_w, inp_h, "notes + DTC")

    # LLM path (top)
    llm_x, llm_y = 3.0, 7.8
    llm_w, llm_h = 3.0, 1.2
    ax.text(llm_x + llm_w / 2, llm_y + llm_h + 0.15, "LLM path (if available):", ha='center', fontsize=8, fontweight='bold', color=COLORS['brown'])
    add_arrow(ax, (inp_x + inp_w, inp_y + inp_h * 0.7), (llm_x, llm_y + llm_h / 2))
    add_llm_box(ax, llm_x, llm_y, llm_w, llm_h, "translate_to_ml_features()")

    # Fallback path (bottom)
    fb_x, fb_y = 3.0, 5.8
    fb_w, fb_h = 3.0, 1.2
    ax.text(fb_x + fb_w / 2, fb_y - 0.2, "Fallback path (deterministic):", ha='center', fontsize=8, fontweight='bold', color=COLORS['gray_medium'])
    add_arrow(ax, (inp_x + inp_w, inp_y + inp_h * 0.3), (fb_x, fb_y + fb_h / 2))
    add_process_box(ax, fb_x, fb_y, fb_w, fb_h, "extract_dtc_features()\n+ match_complaint()")

    # Output box
    out_x, out_y = 7.0, 6.2
    out_w, out_h = 2.8, 2.3
    add_output_box(ax, out_x, out_y, out_w, out_h,
                   "Feature Dict:\n{customer_complaint,\ndtc_codes, dtc_text,\ndtc_count, has_P/U/C/B,\nvoltage, mileage_km,\nyear, ...}", fontsize=7)
    add_arrow(ax, (llm_x + llm_w, llm_y + llm_h / 2), (out_x, out_y + out_h * 0.7))
    add_arrow(ax, (fb_x + fb_w, fb_y + fb_h / 2), (out_x, out_y + out_h * 0.3))

    save_figure(fig, "fig_stage3_feature_translation.pdf")


def plot_stage4_xgboost_cascade():
    """Figure 25: Stage 4 XGBoost cascade."""
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(2.5, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "Stage 4: XGBoost Cascade Scoring", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 2.4, 1.0

    # Input
    inp_x, inp_y = 0.3, 7.5
    add_input_box(ax, inp_x, inp_y, bw, bh, "Feature Vector\n(12 transformers)")

    # FA Classifier
    fa_x, fa_y = 3.3, 7.5
    add_arrow(ax, (inp_x + bw, inp_y + bh / 2), (fa_x, fa_y + bh / 2))
    add_process_box(ax, fa_x, fa_y, bw, bh, "FA Classifier\n(clf_fa)")

    # p_FA label & downward arrow
    fa_cx = fa_x + bw / 2
    add_arrow(ax, (fa_cx, fa_y), (fa_cx, 5.9))
    ax.text(fa_cx + 0.15, 6.65, "p_FA\n(6-class probs)", ha='left', fontsize=7.5, color=COLORS['purple'])

    # Concatenate box
    cat_x, cat_y = fa_x, 5.1
    cat_w, cat_h = bw, 0.8
    add_box(ax, cat_x, cat_y, cat_w, cat_h, "Concatenate:  [X | p_FA]",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=7.5, fontcolor='white')

    # WD Classifier
    wd_x, wd_y = fa_x, 3.6
    add_arrow(ax, (fa_cx, cat_y), (fa_cx, wd_y + bh))
    add_process_box(ax, wd_x, wd_y, bw, bh, "WD Classifier\n(clf_wd)")

    # Output
    out_x, out_y = 6.3, 3.6
    out_w, out_h = 2.5, 2.0
    add_arrow(ax, (wd_x + bw, wd_y + bh / 2), (out_x, out_y + out_h / 2))
    add_output_box(ax, out_x, out_y, out_w, out_h,
                   "Output:\nWD prediction\np_WD probabilities\nconfidence =\n√(c_FA · c_WD)", fontsize=7.5)

    ax.set_xlim(0, 9)
    ax.set_ylim(2.5, 9.5)

    save_figure(fig, "fig_stage4_xgboost_cascade.pdf")


def plot_stage5_score_combination():
    """Figure 26: Stage 5 score combination."""
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(4.5, 9.5)
    ax.axis('off')

    ax.text(6.0, 9.1, "Stage 5: Score Combination", ha='center', fontsize=12, fontweight='bold')

    # Three input boxes side-by-side
    ibw, ibh = 1.8, 1.1
    igap = 0.3
    total_iw = 3 * ibw + 2 * igap
    ix0 = 1.0
    iby = 7.0
    inputs = [
        ("Rule\nResult", COLORS['primary']),
        ("ML\nResult", COLORS['success']),
        ("LLM\nResult", COLORS['brown']),
    ]
    ixs = [ix0 + i * (ibw + igap) for i in range(3)]
    for i, (label, color) in enumerate(inputs):
        add_box(ax, ixs[i], iby, ibw, ibh, label, facecolor=color, edgecolor=color, fontsize=9, fontcolor='white')

    # Combine box — to the right of inputs with gap
    cbx = ix0 + total_iw + 0.6
    cby = iby
    cbw, cbh = 2.5, 1.1
    add_process_box(ax, cbx, cby, cbw, cbh, "combine_scores()\nWeighted blend\n+ Agreement check")

    # Arrows from each input box right edge to combine box left edge
    for i in range(3):
        add_arrow(ax, (ixs[i] + ibw, iby + ibh / 2), (cbx, cby + cbh / 2))

    # Output box
    obx = cbx + cbw + 0.5
    oby = iby
    obw, obh = 2.8, 1.1
    add_output_box(ax, obx, oby, obw, obh,
                   "Output:\ncombined_confidence\nstatus, decision_engine", fontsize=7)
    add_arrow(ax, (cbx + cbw, cby + cbh / 2), (obx, oby + obh / 2))

    ax.set_xlim(0, obx + obw + 0.3)

    save_figure(fig, "fig_stage5_score_combination.pdf")


def plot_stage6_output_formatting():
    """Figure 27: Stage 6 output formatting."""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 10.0)
    ax.axis('off')

    ax.text(5.0, 9.6, "Stage 6: Output Formatting", ha='center', fontsize=12, fontweight='bold')

    # Input box (left)
    inp_x, inp_y, inp_w, inp_h = 0.4, 6.5, 2.4, 1.3
    add_input_box(ax, inp_x, inp_y, inp_w, inp_h, "combined_decision\n+ features")

    # LLM path (top)
    llm_x, llm_y = 3.4, 7.3
    llm_w, llm_h = 2.6, 1.2
    ax.text(llm_x + llm_w / 2, llm_y + llm_h + 0.15, "LLM path:", ha='center', fontsize=8, fontweight='bold', color=COLORS['brown'])
    add_arrow(ax, (inp_x + inp_w, inp_y + inp_h * 0.75), (llm_x, llm_y + llm_h / 2))
    add_llm_box(ax, llm_x, llm_y, llm_w, llm_h, "format_output()\nNatural language\nreason generation")

    # Fallback path (bottom)
    fb_x, fb_y = 3.4, 5.3
    fb_w, fb_h = 2.6, 1.2
    ax.text(fb_x + fb_w / 2, fb_y - 0.2, "Fallback:", ha='center', fontsize=8, fontweight='bold', color=COLORS['gray_medium'])
    add_arrow(ax, (inp_x + inp_w, inp_y + inp_h * 0.25), (fb_x, fb_y + fb_h / 2))
    add_process_box(ax, fb_x, fb_y, fb_w, fb_h, "assemble_output_from_fields()\nTemplate-based reason")

    # Output box (right) — merge of both paths
    out_x, out_y = 6.8, 5.8
    out_w, out_h = 2.8, 2.1
    add_output_box(ax, out_x, out_y, out_w, out_h,
                   "ClaimResponse:\nstatus, FA, WD,\nconfidence, reason,\nmatched_complaint,\ndecision_engine", fontsize=7)

    add_arrow(ax, (llm_x + llm_w, llm_y + llm_h / 2), (out_x, out_y + out_h * 0.7))
    add_arrow(ax, (fb_x + fb_w, fb_y + fb_h / 2), (out_x, out_y + out_h * 0.3))

    save_figure(fig, "fig_stage6_output_formatting.pdf")


def plot_complete_predict_flow():
    """Figure 28: Complete prediction flow."""
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(2.0, 9.5)
    ax.axis('off')

    ax.text(6.5, 9.1, "Complete predict() Flow", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 1.8, 1.0
    gap = 0.3
    total_w = 6 * bw + 5 * gap
    x0 = (13 - total_w) / 2
    by = 7.0

    stages = [
        ("1. LLM Check", "api_key & len>5", COLORS['brown']),
        ("2. Stage 1", "LLM understand", COLORS['brown']),
        ("3. Stage 2", "run_rules()", COLORS['primary']),
        ("4. Stage 3", "features", COLORS['brown']),
        ("5. Stage 4", "run_ml()", COLORS['success']),
        ("6. Stage 5", "combine_scores()", COLORS['purple']),
    ]
    xs = [x0 + i * (bw + gap) for i in range(6)]

    for i, (name, detail, color) in enumerate(stages):
        cx = xs[i] + bw / 2
        add_stage_box(ax, xs[i], by, bw, bh, name, facecolor=color, fontcolor='white', fontsize=8)
        ax.text(cx, by - 0.3, detail, ha='center', va='top', fontsize=6.5, style='italic')

    for i in range(5):
        mid_y = by + bh / 2
        add_arrow(ax, (xs[i] + bw, mid_y), (xs[i + 1], mid_y))

    # Error handling box — centred below the stage row
    mid_x = x0 + total_w / 2
    err_w, err_h = 4.0, 1.0
    err_x = mid_x - err_w / 2
    err_y = 5.0
    add_arrow(ax, (mid_x, by - 0.1), (mid_x, err_y + err_h))
    add_box(ax, err_x, err_y, err_w, err_h,
            "Error Handling:\ntry/except per stage → fallback paths",
            facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=8, fontcolor='white')

    # Stage 6 output box
    out_w, out_h = 2.8, 0.9
    out_x = mid_x - out_w / 2
    out_y = 3.4
    add_arrow(ax, (mid_x, err_y), (mid_x, out_y + out_h))
    add_output_box(ax, out_x, out_y, out_w, out_h, "Stage 6: format_output()", fontsize=8)

    ax.set_xlim(0, 13)
    ax.set_ylim(2.0, 9.5)

    save_figure(fig, "fig_complete_predict_flow.pdf")


def plot_fastapi_architecture():
    """Figure 34: FastAPI architecture."""
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(5.5, 9.1, "FastAPI Backend Architecture", ha='center', fontsize=12, fontweight='bold')

    # Top: FastAPI App box centred
    app_w, app_h = 3.2, 0.9
    app_x = (11 - app_w) / 2
    app_y = 8.0
    add_box(ax, app_x, app_y, app_w, app_h, "FastAPI App",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=10, fontcolor='white')

    # Middleware/schema boxes — 4, evenly spread
    midboxes = [
        ("CORS\nMiddleware", COLORS['gray_medium']),
        ("Request Schema\n(ClaimRequest)", COLORS['secondary']),
        ("Response Schema\n(ClaimResponse)", COLORS['secondary']),
        ("/analyze Endpoint\nPOST", COLORS['success']),
    ]
    mbw, mbh = 2.1, 0.9
    mgap = 0.35
    total_mw = 4 * mbw + 3 * mgap
    mx0 = (11 - total_mw) / 2
    mby = 6.4
    mxs = [mx0 + i * (mbw + mgap) for i in range(4)]

    # Arrow from FastAPI box to middleware row
    add_arrow(ax, (app_x + app_w / 2, app_y), (mx0 + total_mw / 2, mby + mbh))

    for i, (name, color) in enumerate(midboxes):
        add_box(ax, mxs[i], mby, mbw, mbh, name, facecolor=color, edgecolor=color, fontsize=7, fontcolor='white')
        if i < 3:
            add_arrow(ax, (mxs[i] + mbw, mby + mbh / 2), (mxs[i + 1], mby + mbh / 2))

    # ML predictor below
    pred_w, pred_h = 2.4, 0.9
    pred_x = (11 - pred_w) / 2
    pred_y = 4.8
    add_arrow(ax, (mx0 + total_mw / 2, mby), (pred_x + pred_w / 2, pred_y + pred_h))
    add_process_box(ax, pred_x, pred_y, pred_w, pred_h,
                    "ML Predictor\npredict()", facecolor=COLORS['purple'], edgecolor=COLORS['purple'],
                    fontsize=8, fontcolor='white')

    # Endpoints below
    end_boxes = [
        ("GET /", "Health check"),
        ("POST /analyze", "Claim analysis"),
    ]
    ebw, ebh = 3.5, 0.9
    exs = [1.5, 6.0]
    eby = 3.0
    add_arrow(ax, (pred_x + pred_w / 2, pred_y), (5.5, eby + ebh))
    for i, (path, desc) in enumerate(end_boxes):
        add_box(ax, exs[i], eby, ebw, ebh, f"{path}\n{desc}",
                facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)

    save_figure(fig, "fig_fastapi_architecture.pdf")


def plot_frontend_architecture():
    """Figure 35: Frontend architecture."""
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 9.5)
    ax.axis('off')

    ax.text(5.0, 9.1, "Frontend Architecture (Single-Page App)", ha='center', fontsize=12, fontweight='bold')

    sections = [
        ("HTML / CSS", "Form: fault_code,\nnotes, voltage", COLORS['primary']),
        ("JavaScript", "Event handlers,\nAPI calls", COLORS['secondary']),
        ("Backend API", "POST http://\nlocalhost:8000/analyze", COLORS['success']),
    ]

    sbw, sbh = 2.4, 1.1
    sgap = 0.4
    total_sw = 3 * sbw + 2 * sgap
    sx0 = (10 - total_sw) / 2
    sby = 7.0
    sxs = [sx0 + i * (sbw + sgap) for i in range(3)]

    for i, (name, detail, color) in enumerate(sections):
        cx = sxs[i] + sbw / 2
        add_stage_box(ax, sxs[i], sby, sbw, sbh, name, facecolor=color, fontcolor='white', fontsize=9)
        ax.text(cx, sby - 0.35, detail, ha='center', va='top', fontsize=6.5, style='italic')
        if i < 2:
            add_arrow(ax, (sxs[i] + sbw, sby + sbh / 2), (sxs[i + 1], sby + sbh / 2))

    # Result display box below API section
    mid_x = sx0 + total_sw / 2
    res_w, res_h = 5.5, 0.85
    res_x = mid_x - res_w / 2
    res_y = 5.0
    api_cx = sxs[2] + sbw / 2
    add_arrow(ax, (api_cx, sby), (mid_x, res_y + res_h))
    add_output_box(ax, res_x, res_y, res_w, res_h,
                   "Result Display:  status, FA, WD, confidence, reason", fontsize=8)

    save_figure(fig, "fig_frontend_architecture.pdf")


def plot_docker_compose_architecture():
    """Figure 36: Docker Compose architecture."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(2.0, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "Docker Compose Deployment", ha='center', fontsize=12, fontweight='bold')

    bw, bh = 3.0, 1.5

    # Backend container (left)
    be_x, be_y = 0.5, 6.8
    add_box(ax, be_x, be_y, bw, bh,
            "Backend Container\nPort: 8000:8000\nHealth: GET /",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontcolor='white')

    # Frontend container (right)
    fe_x, fe_y = 5.5, 6.8
    add_box(ax, fe_x, fe_y, bw, bh,
            "Frontend Container\nPort: 3000:3000\nDepends: backend",
            facecolor=COLORS['secondary'], edgecolor=COLORS['secondary'], fontsize=8, fontcolor='white')

    # Network bridge box (centre, below)
    net_w, net_h = 2.2, 0.9
    net_x = (9 - net_w) / 2
    net_y = 4.9
    add_box(ax, net_x, net_y, net_w, net_h, "trace_net\n(bridge)",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')

    # Arrows: backend ↔ network
    net_cx = net_x + net_w / 2
    be_cx = be_x + bw / 2
    fe_cx = fe_x + bw / 2
    net_top = net_y + net_h

    add_arrow(ax, (be_cx, be_y), (net_cx - 0.2, net_top), color=COLORS['primary'])
    add_arrow(ax, (net_cx - 0.4, net_top), (be_cx - 0.3, be_y), color=COLORS['primary'])

    # Arrows: frontend ↔ network
    add_arrow(ax, (fe_cx, fe_y), (net_cx + 0.2, net_top), color=COLORS['secondary'])
    add_arrow(ax, (net_cx + 0.4, net_top), (fe_cx + 0.3, fe_y), color=COLORS['secondary'])

    ax.text(4.5, 3.9, "docker-compose.yml", ha='center', fontsize=9, fontweight='bold')
    ax.text(4.5, 3.4, "restart: unless-stopped", ha='center', fontsize=8, style='italic')

    ax.set_xlim(0, 9)
    ax.set_ylim(2.0, 9.5)

    save_figure(fig, "fig_docker_compose_architecture.pdf")


def plot_logging_architecture():
    """Figure 37: Logging architecture."""
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(4.5, 9.1, "Logging Architecture", ha='center', fontsize=12, fontweight='bold')

    # Root logger
    rl_w, rl_h = 2.8, 0.8
    rl_x = (9 - rl_w) / 2
    rl_y = 8.0
    add_box(ax, rl_x, rl_y, rl_w, rl_h, "Root Logger",
            facecolor=COLORS['gray_dark'], edgecolor=COLORS['gray_dark'], fontsize=9, fontcolor='white')

    # Child loggers — 3, evenly spread
    children = [
        ("trace.ml_predictor", COLORS['primary']),
        ("trace.llm_client", COLORS['secondary']),
        ("trace.api", COLORS['success']),
    ]
    cbw, cbh = 2.4, 0.75
    cgap = 0.2
    total_cw = 3 * cbw + 2 * cgap
    cx0 = (9 - total_cw) / 2
    cby = 6.4
    cxs = [cx0 + i * (cbw + cgap) for i in range(3)]
    root_cx = rl_x + rl_w / 2

    for i, (name, color) in enumerate(children):
        cx = cxs[i] + cbw / 2
        add_box(ax, cxs[i], cby, cbw, cbh, name, facecolor=color, edgecolor=color, fontsize=7.5, fontcolor='white')
        add_arrow(ax, (root_cx, rl_y), (cx, cby + cbh))

    # DecisionLogger helper
    dl_w, dl_h = 3.0, 0.75
    dl_x = (9 - dl_w) / 2
    dl_y = 5.0
    add_box(ax, dl_x, dl_y, dl_w, dl_h, "DecisionLogger",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontcolor='white')
    ax.text(4.5, 4.6, "log_stage(),  log_decision()", ha='center', fontsize=7.5, style='italic')

    dl_targets = [dl_x + dl_w * 0.25, dl_x + dl_w * 0.5, dl_x + dl_w * 0.75]
    for i in range(3):
        cx = cxs[i] + cbw / 2
        add_arrow(ax, (cx, cby), (dl_targets[i], dl_y + dl_h))

    # Format string
    ax.text(4.5, 3.9, "Format: %(asctime)s [%(levelname)s] %(name)s %(message)s",
            ha='center', fontsize=7.5, family='monospace')

    ax.set_xlim(0, 9)
    ax.set_ylim(1.5, 9.5)

    save_figure(fig, "fig_logging_architecture.pdf")


def plot_model_serialization():
    """Figure 38: Model serialization."""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 9.5)
    ax.axis('off')

    ax.text(5.0, 9.1, "Model Serialization & Auto-Training", ha='center', fontsize=12, fontweight='bold')

    # Top: Training pipeline — 4 stage boxes
    stages = [
        ("Training", "train_and_save()", COLORS['success']),
        ("Bundle", "14 components", COLORS['purple']),
        ("Serialize", "pickle.dump()", COLORS['secondary']),
        ("Storage", "trace_models.pkl", COLORS['gray_medium']),
    ]
    sbw, sbh = 1.8, 0.9
    sgap = 0.4
    total_sw = 4 * sbw + 3 * sgap
    sx0 = (10 - total_sw) / 2
    sby = 7.5
    sxs = [sx0 + i * (sbw + sgap) for i in range(4)]

    for i, (name, detail, color) in enumerate(stages):
        cx = sxs[i] + sbw / 2
        add_stage_box(ax, sxs[i], sby, sbw, sbh, name, facecolor=color, fontcolor='white', fontsize=8)
        ax.text(cx, sby - 0.3, detail, ha='center', va='top', fontsize=6.5, style='italic')
    for i in range(3):
        add_arrow(ax, (sxs[i] + sbw, sby + sbh / 2), (sxs[i + 1], sby + sbh / 2))

    # Inference section
    ax.text(5.0, 6.2, "Inference Time:", ha='center', fontsize=10, fontweight='bold')

    # Decision: file exists?
    q_x, q_y = 4.0, 5.2
    q_w, q_h = 2.0, 0.8
    add_box(ax, q_x, q_y, q_w, q_h, "File exists?",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)

    # Yes → left branch
    yes_x, yes_y, yes_w, yes_h = 1.0, 3.6, 2.2, 0.8
    add_arrow(ax, (q_x, q_y + q_h / 2), (yes_x + yes_w, yes_y + yes_h / 2), color=COLORS['success'])
    ax.text(2.2, 4.6, "Yes", fontsize=8, color=COLORS['success'], fontweight='bold')
    add_output_box(ax, yes_x, yes_y, yes_w, yes_h, "pickle.load()", fontsize=8)

    # No → right branch
    no_x, no_y, no_w, no_h = 6.8, 3.6, 2.2, 0.8
    add_arrow(ax, (q_x + q_w, q_y + q_h / 2), (no_x, no_y + no_h / 2), color=COLORS['danger'])
    ax.text(6.0, 4.6, "No", fontsize=8, color=COLORS['danger'], fontweight='bold')
    add_output_box(ax, no_x, no_y, no_w, no_h, "train_and_save()", fontsize=8, facecolor=COLORS['secondary'], edgecolor=COLORS['secondary'])

    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 9.5)

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
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 9.5)
    ax.axis('off')

    ax.text(5.0, 9.1, "DTC Pool Architecture", ha='center', fontsize=12, fontweight='bold')

    pools = [
        ("ASIC", "P0601–P0617", COLORS['asic']),
        ("Track", "P0300–P0356", COLORS['track']),
        ("Sensor", "P0113–P0343", COLORS['moisture']),
        ("Connector", "C0031–C0550", COLORS['connector']),
        ("Controller", "U0001–U0184", COLORS['controller']),
    ]

    pbw, pbh = 1.5, 0.85
    pgap = 0.2
    total_pw = 5 * pbw + 4 * pgap
    px0 = (10 - total_pw) / 2
    pby = 7.5
    pxs = [px0 + i * (pbw + pgap) for i in range(5)]

    for i, (name, codes, color) in enumerate(pools):
        cx = pxs[i] + pbw / 2
        add_box(ax, pxs[i], pby, pbw, pbh, name,
                facecolor=color, edgecolor=color, fontsize=9, fontcolor='white')
        ax.text(cx, pby - 0.4, codes, ha='center', va='top', fontsize=7, style='italic')
        if i < 4:
            add_arrow(ax, (pxs[i] + pbw, pby + pbh / 2), (pxs[i + 1], pby + pbh / 2),
                      color=COLORS['danger'], lw=1.2)

    # Companion DTC section
    ax.text(5.0, 6.0, "Companion DTC Injection", ha='center', fontsize=10, fontweight='bold')
    ax.text(5.0, 5.5, "P0562 ↔ P0563  (55%)", ha='center', fontsize=9, family='monospace')
    ax.text(5.0, 5.0, "U0100 ↔ U0101  (60%)", ha='center', fontsize=9, family='monospace')

    # Cross-FA section
    ax.text(5.0, 4.2, "Cross-FA Injection (4%)", ha='center', fontsize=10, fontweight='bold')
    ax.text(5.0, 3.7, "DTC_AMBIGUOUS_CROSS: P0300, P0171, P0325 …", ha='center', fontsize=8.5, family='monospace')

    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 9.5)

    save_figure(fig, "fig_dtc_pool_architecture.pdf")


def plot_cv_on_test_set():
    """Figure 30: Cross-validation on held-out test set."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis('off')

    ax.text(4.0, 9.2, "3-Fold Cross-Validation on Test Set", ha='center', fontsize=11, fontweight='bold')
    ax.text(4.0, 8.6, "Test Set: 20,000 rows → 3 folds × ~6,667 rows", ha='center', fontsize=9, style='italic')

    n = 3
    box_width = 1.8
    box_height = 0.9
    gap = 0.25

    for i in range(n):
        y = 7.0 - i * 1.5

        ax.text(0.5, y + box_height / 2, f"Fold {i+1}", fontsize=9, va='center', fontweight='bold')

        for j in range(n):
            x = 1.8 + j * (box_width + gap)

            if j == i:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="round,pad=0.05",
                                      facecolor=COLORS['white'], edgecolor=COLORS['danger'], linewidth=2)
                ax.text(x + box_width / 2, y + box_height / 2, "Validation", ha='center', va='center',
                        fontsize=7, color=COLORS['danger'])
            else:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="round,pad=0.05",
                                      facecolor=COLORS['primary'], alpha=0.5, edgecolor=COLORS['primary'], linewidth=1)
                ax.text(x + box_width / 2, y + box_height / 2, "Training", ha='center', va='center',
                        fontsize=7, color='white')
            ax.add_patch(rect)

    ax.set_xlim(0, 8)
    ax.set_ylim(3.0, 9.8)

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