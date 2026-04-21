#!/usr/bin/env python3
"""
Generate 44 Figures for Foundation_TRACE.tex
============================================
This script generates vector PDF figures for the TRACE technical documentation.
Each figure is saved to Documentation/figures/ directory.
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

def setup_style():
    """Configure matplotlib rcParams for academic style."""
    pass

def create_figure(width, height):
    """Create a new figure with given dimensions."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')
    return fig, ax

def save_figure(fig, filename):
    """Save figure as PDF with tight bounding box."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

def style_axes(ax):
    """Apply consistent axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

def add_box(ax, x, y, w, h, text, **kwargs):
    """Add a styled text box at position (x,y) with width w, height h."""
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
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=fontcolor, wrap=True)
    return box

def add_arrow(ax, start, end, **kwargs):
    """Add an arrow from start=(x1,y1) to end=(x2,y2)."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=kwargs.get('color', COLORS['gray_medium']),
                               lw=kwargs.get('lw', 1.5)))

def add_diamond(ax, x, y, size, text, **kwargs):
    """Add a diamond-shaped decision node."""
    diamond = Polygon(
        [[x, y+size], [x+size, y], [x, y-size], [x-size, y]],
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=7)
    return diamond


def plot_supervised_learning_pipeline():
    """Figure 1: Supervised learning pipeline block diagram."""
    fig, ax = create_figure(6.5, 3.0)
    
    ax.text(2.5, 9.0, "TRAINING PHASE", ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])
    ax.text(7.5, 9.0, "INFERENCE PHASE", ha='center', fontsize=10, fontweight='bold', color=COLORS['success'])
    
    add_box(ax, 0.5, 6.5, 2.8, 1.0, "Labeled Data\nD = {(x1,y1), ...}", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    add_box(ax, 3.8, 6.5, 2.2, 1.0, "Model Fitting", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)
    add_box(ax, 6.5, 6.5, 2.2, 1.0, "Trained Model\nf-hat", facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=9)
    
    add_arrow(ax, (3.45, 7.0), (3.65, 7.0))
    add_arrow(ax, (6.15, 7.0), (6.35, 7.0))
    
    add_box(ax, 5.0, 4.0, 1.8, 0.9, "New Input\nx", facecolor=COLORS['bg_box'], edgecolor=COLORS['secondary'], fontsize=9)
    add_box(ax, 7.2, 4.0, 1.8, 0.9, "Prediction\ny-hat", facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=9)
    
    ax.annotate('', xy=(6.95, 4.5), xytext=(6.75, 4.5), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))
    ax.annotate('', xy=(7.05, 4.5), xytext=(7.25, 4.5), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))
    ax.annotate('', xy=(7.2, 5.0), xytext=(7.2, 5.1), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0, ls='--'))
    
    ax.text(2.5, 2.5, "y = f(x) learned from\nlabeled examples", ha='center', fontsize=8, style='italic')
    ax.text(7.5, 2.5, "y-hat = f-hat(x) applied\nto new data", ha='center', fontsize=8, style='italic')
    
    save_figure(fig, "fig_supervised_learning_pipeline.pdf")


def plot_multiclass_decision_boundary():
    """Figure 2: 2D feature space with 3 class decision regions."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    np.random.seed(42)
    X, y = make_classification(n_classes=3, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, n_samples=200, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape + (3,))
    
    ax.contourf(xx, yy, Z[:,:,0], levels=[0, 0.33, 1], colors=[COLORS['secondary'], COLORS['primary']], alpha=0.25)
    ax.contourf(xx, yy, Z[:,:,1], levels=[0, 0.33, 1], colors=[COLORS['primary'], COLORS['success']], alpha=0.25)
    ax.contourf(xx, yy, Z[:,:,2], levels=[0, 0.33, 1], colors=[COLORS['success'], COLORS['secondary']], alpha=0.25)
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Class {chr(65+i)}', edgecolors='white', s=35, alpha=0.7)
    
    ax.contour(xx, yy, Z[:,:,0], levels=[0.33], colors='black', linestyles='--', linewidths=0.8)
    
    ax.set_xlabel('Feature X1', fontsize=10)
    ax.set_ylabel('Feature X2', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    style_axes(ax)
    
    save_figure(fig, "fig_multiclass_decision_boundary.pdf")


def plot_decision_tree_example():
    """Figure 3: Decision tree diagram with nodes and edges."""
    fig, ax = create_figure(5.0, 4.0)
    
    def add_node(ax, x, y, w, h, text, color=COLORS['bg_box'], border=COLORS['primary']):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=7, fontweight='bold')
        return box
    
    def add_leaf(ax, x, y, text, color):
        ellipse = Ellipse((x, y), 1.6, 0.9, facecolor=color, edgecolor=COLORS['gray_dark'], linewidth=1.5)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    
    add_node(ax, 3.0, 8.0, 2.2, 0.7, "Voltage > 14.5?")
    
    add_node(ax, 1.0, 6.0, 1.8, 0.7, "DTC starts\nwith 'P'?")
    add_node(ax, 5.0, 6.0, 1.8, 0.7, "Notes\n'moisture'?")
    
    add_leaf(ax, 1.2, 4.0, "Production\nFailure", COLORS['success'])
    add_leaf(ax, 2.8, 4.0, "Customer\nFailure", COLORS['secondary'])
    add_leaf(ax, 5.2, 4.0, "Rejected", COLORS['danger'])
    add_leaf(ax, 6.8, 4.0, "Needs\nReview", COLORS['gray_medium'])

    ax.annotate('', xy=(1.9, 6.7), xytext=(3.0, 8.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(2.2, 7.5, "Y", fontsize=7, color=COLORS['success'])
    ax.annotate('', xy=(5.9, 6.7), xytext=(5.2, 8.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(5.8, 7.5, "N", fontsize=7, color=COLORS['danger'])

    ax.annotate('', xy=(1.2, 4.5), xytext=(1.9, 6.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(1.1, 5.5, "Y", fontsize=7, color=COLORS['success'])
    ax.annotate('', xy=(2.8, 4.5), xytext=(2.8, 6.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(3.1, 5.5, "N", fontsize=7, color=COLORS['danger'])

    ax.annotate('', xy=(5.2, 4.5), xytext=(5.9, 6.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(5.0, 5.5, "Y", fontsize=7, color=COLORS['success'])
    ax.annotate('', xy=(6.8, 4.5), xytext=(6.8, 6.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
    ax.text(7.1, 5.5, "N", fontsize=7, color=COLORS['danger'])

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(2.8, 9.5)
    
    save_figure(fig, "fig_decision_tree_example.pdf")


def plot_random_forest_architecture():
    """Figure 4: Random Forest architecture diagram."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.axis('off')
    ax.set_xlim(0, 9.5)
    ax.set_ylim(2.0, 8.0)
    
    add_box(ax, 0.5, 5.0, 1.8, 0.9, "Training Data\nn samples", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    
    ax.annotate('', xy=(2.75, 5.45), xytext=(2.5, 5.45), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    ax.text(2.5, 5.7, "Bootstrap", ha='center', fontsize=6, color=COLORS['gray_medium'])
    
    add_box(ax, 3.0, 6.5, 1.3, 0.6, "Sample 1", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=7)
    add_box(ax, 3.0, 4.8, 1.3, 0.6, "Sample 2", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=7)
    add_box(ax, 3.0, 3.1, 1.3, 0.6, "Sample 3", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=7)
    
    ax.annotate('', xy=(3.65, 6.5), xytext=(3.65, 5.9), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1))
    ax.annotate('', xy=(3.65, 4.8), xytext=(3.65, 4.2), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1))
    ax.annotate('', xy=(3.65, 3.1), xytext=(3.65, 2.5), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1))
    
    add_box(ax, 5.0, 6.5, 1.1, 0.6, "Tree 1", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')
    add_box(ax, 5.0, 4.8, 1.1, 0.6, "Tree 2", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')
    add_box(ax, 5.0, 3.1, 1.1, 0.6, "Tree 3", facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')
    
    ax.annotate('', xy=(6.35, 5.45), xytext=(6.15, 5.45), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    ax.annotate('', xy=(6.35, 4.75), xytext=(6.15, 5.05), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    ax.annotate('', xy=(6.35, 3.6), xytext=(6.15, 4.2), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    
    add_box(ax, 6.8, 4.8, 2.0, 0.9, "Vote\nAggregation", facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontweight='bold')
    
    ax.text(7.8, 3.8, "n_estimators = T", ha='center', fontsize=7, style='italic')
    
    save_figure(fig, "fig_random_forest_architecture.pdf")


def plot_gradient_boosting_sequential():
    """Figure 5: Sequential gradient boosting flow diagram."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 9.5)

    ax.text(5.0, 9.0, "Gradient Boosting: Sequential Residual Fitting", ha='center', fontsize=11, fontweight='bold')

    # Step 1: Initial prediction
    add_box(ax, 0.5, 7.2, 2.5, 0.8, r"$F_0(x) = \mathrm{mean}(y)$",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    ax.annotate('', xy=(3.5, 7.6), xytext=(3.0, 7.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Step 2: Compute residual
    add_box(ax, 3.5, 7.2, 2.5, 0.8, r"$r_1 = y - F_0(x)$",
            facecolor=COLORS['danger'], edgecolor=COLORS['danger'], fontsize=8, fontweight='bold', fontcolor='white')
    ax.annotate('', xy=(6.5, 7.6), xytext=(6.0, 7.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Step 3: Fit tree to residual
    add_box(ax, 6.5, 7.2, 2.5, 0.8, r"Fit tree $h_1$ to $r_1$",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    # Arrow down to update
    ax.annotate('', xy=(5.0, 6.5), xytext=(5.0, 7.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Step 4: Update model
    add_box(ax, 2.5, 5.5, 5.0, 0.8, r"$F_1(x) = F_0(x) + \eta \cdot h_1(x)$",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=9, fontweight='bold')

    # Arrow down to next iteration
    ax.annotate('', xy=(5.0, 4.7), xytext=(5.0, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Step 5: Next residual
    add_box(ax, 3.0, 3.7, 4.0, 0.8, r"$r_2 = y - F_1(x)$   ...   repeat T times",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)

    ax.text(5.0, 2.5, r"Each tree corrects the errors of the previous ensemble",
            ha='center', fontsize=8, style='italic', color=COLORS['gray_medium'])
    ax.text(5.0, 1.8, r"$\eta$ (learning rate) controls each tree's contribution",
            ha='center', fontsize=8, style='italic', color=COLORS['danger'])

    save_figure(fig, "fig_gradient_boosting_sequential.pdf")


def plot_cascade_classifier():
    """Figure 6: Two-stage cascade classifier diagram."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 9.5)

    # Stage 1 label
    ax.text(1.0, 8.8, "Stage 1", ha='left', fontsize=9, color=COLORS['primary'], fontweight='bold')

    # Input x
    add_box(ax, 0.5, 7.2, 1.6, 0.9, "Input x", facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=9)

    # Arrow to f1
    ax.annotate('', xy=(2.8, 7.65), xytext=(2.1, 7.65),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Classifier f1
    add_box(ax, 2.8, 7.2, 2.4, 0.9, "Classifier f1\n(Failure Analysis)",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')

    # Arrow with "softmax" label
    ax.annotate('', xy=(6.0, 7.65), xytext=(5.2, 7.65),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    ax.text(5.6, 7.95, "softmax", ha='center', fontsize=7, style='italic')

    # Probability vector p
    add_box(ax, 6.0, 7.2, 1.2, 0.9, "p\n(6-class)", facecolor=COLORS['bg_box'], edgecolor=COLORS['purple'], fontsize=8)

    # Dashed augmentation box
    rect = FancyBboxPatch((0.3, 5.0), 7.1, 1.0, boxstyle="round,pad=0.08",
                          facecolor='none', edgecolor=COLORS['purple'], linewidth=1.2, linestyle='--')
    ax.add_patch(rect)
    ax.text(3.85, 5.4, "Augmented feature vector:  x' = [x | p]",
            ha='center', fontsize=8, color=COLORS['purple'])

    # Arrows down from x and p into augmented box
    ax.annotate('', xy=(1.3, 6.0), xytext=(1.3, 7.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.0, ls='--'))
    ax.annotate('', xy=(6.6, 6.0), xytext=(6.6, 7.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.0, ls='--'))

    # Arrow down from augmented to f2
    ax.annotate('', xy=(3.85, 4.0), xytext=(3.85, 5.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Stage 2 label
    ax.text(1.0, 4.2, "Stage 2", ha='left', fontsize=9, color=COLORS['success'], fontweight='bold')

    # Classifier f2
    add_box(ax, 2.8, 3.0, 2.4, 0.9, "Classifier f2\n(Warranty Decision)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    # Arrow to final prediction
    ax.annotate('', xy=(6.0, 3.45), xytext=(5.2, 3.45),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Final prediction
    add_box(ax, 6.0, 3.0, 2.0, 0.9, "Final\nPrediction",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    save_figure(fig, "fig_cascade_classifier.pdf")


def plot_kfold_cross_validation():
    """Figure 7: K-fold cross-validation diagram."""
    fig, ax = create_figure(5.0, 3.5)
    
    ax.text(2.5, 9.0, "K = 5", ha='center', fontsize=10, fontweight='bold')
    ax.text(2.5, 8.5, "Full Dataset", ha='center', fontsize=8)
    
    K = 5
    box_width = 1.3
    box_height = 0.45
    gap = 0.12
    
    for i in range(K):
        y = 7.0 - i * 1.2
        ax.text(0.3, y + 0.2, f"Iter {i+1}", fontsize=7, va='center')
        
        for j in range(K):
            x = 1.3 + j * (box_width + gap)
            
            if j == i:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="square", facecolor=COLORS['white'], edgecolor=COLORS['danger'], linewidth=2)
            else:
                rect = FancyBboxPatch((x, y), box_width, box_height, boxstyle="square", facecolor=COLORS['primary'], alpha=0.5, edgecolor=COLORS['primary'], linewidth=1)
            ax.add_patch(rect)
    
    ax.text(3.8, 1.2, "Shaded: Train", fontsize=7, color=COLORS['primary'])
    ax.text(3.8, 0.7, "White: Valid", fontsize=7, color=COLORS['danger'])
    
    save_figure(fig, "fig_kfold_cross_validation.pdf")


def plot_data_leakage_scenarios():
    """Figure 8: Correct vs incorrect preprocessing pipeline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax1.axis('off')
    ax2.axis('off')
    
    ax1.text(5, 9.0, "CORRECT", ha='center', fontsize=10, fontweight='bold', color=COLORS['success'])
    ax2.text(5, 9.0, "INCORRECT", ha='center', fontsize=10, fontweight='bold', color=COLORS['danger'])
    
    boxes1 = [("Data", 0.5, 6.5), ("Split", 2.3, 6.5), ("Train", 4.1, 6.5), ("Fit", 5.9, 6.5), ("Trans", 7.7, 6.5)]
    for label, x, y in boxes1:
        add_box(ax1, x, y, 1.2, 0.7, label, facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=7)
    
    for i in range(len(boxes1)-1):
        ax1.annotate('', xy=(boxes1[i+1][1], 6.85), xytext=(boxes1[i][1]+1.2, 6.85), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    
    ax2.text(5, 5.5, "DATA LEAKAGE!", ha='center', fontsize=11, fontweight='bold', color=COLORS['danger'])
    
    boxes2 = [("Data", 0.5, 3.5), ("Fit", 2.3, 3.5), ("Split", 4.1, 3.5), ("Train", 5.9, 3.5), ("Test", 7.7, 3.5)]
    for label, x, y in boxes2:
        color = COLORS['danger'] if label in ("Fit", "Split") else COLORS['bg_box']
        edge = COLORS['danger'] if label in ("Fit", "Split") else COLORS['primary']
        add_box(ax2, x, y, 1.2, 0.7, label, facecolor=color, edgecolor=edge, fontsize=7)
    
    for i in range(len(boxes2)-1):
        ax2.annotate('', xy=(boxes2[i+1][1], 3.85), xytext=(boxes2[i][1]+1.2, 3.85), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    
    ax2.axvline(x=5.0, ymin=0.15, ymax=0.85, color=COLORS['danger'], linestyle='--', linewidth=2)
    
    save_figure(fig, "fig_data_leakage_scenarios.pdf")


def plot_tfidf_computation():
    """Figure 9: TF-IDF computation pipeline."""
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 8.5)

    ax.text(5.0, 8.0, "TF-IDF Computation Pipeline", ha='center', fontsize=11, fontweight='bold')

    boxes = [
        (0.3, "Raw Text", "engine\noverheating"),
        (2.2, "Tokenize", "['engine',\n'overheating']"),
        (4.1, "TF", "count / total"),
        (6.0, "IDF", "log(N / df)"),
        (7.9, "TF-IDF", "TF x IDF"),
    ]

    bw, bh = 1.5, 1.0
    for x, title, content in boxes:
        add_box(ax, x, 5.5, bw, bh, title,
                facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')
        ax.text(x + bw / 2, 5.2, content, ha='center', fontsize=5.5, family='monospace', color=COLORS['gray_medium'])

    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1][0], 6.0), xytext=(boxes[i][0] + bw, 6.0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    ax.text(5.0, 4.2, r"$\mathrm{TF}(t,d) = \frac{\mathrm{count}(t,d)}{|d|}$        "
            r"$\mathrm{IDF}(t) = \log\frac{N}{df(t)} + 1$        "
            r"$\mathrm{TF\text{-}IDF} = \mathrm{TF} \times \mathrm{IDF}$",
            ha='center', fontsize=8)

    save_figure(fig, "fig_tfidf_computation.pdf")


def plot_one_hot_encoding():
    """Figure 10: One-hot encoding visual mapping."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.axis('off')
    ax.set_xlim(-0.2, 5.5)
    ax.set_ylim(0.5, 9.5)
    
    categories = ["red", "green", "blue"]
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    colors_list = [COLORS['danger'], COLORS['success'], COLORS['primary']]
    
    ax.text(1.0, 8.0, "Categorical", ha='center', fontsize=9, fontweight='bold')
    ax.text(4.0, 8.0, "One-Hot Encoded", ha='center', fontsize=9, fontweight='bold')
    ax.text(2.5, 7.5, "K = 3", ha='center', fontsize=8, style='italic')
    
    for i, (cat, vec, col) in enumerate(zip(categories, vectors, colors_list)):
        y = 6.0 - i * 1.5
        
        ax.add_patch(FancyBboxPatch((0.3, y-0.25), 1.2, 0.7, boxstyle="round,pad=0.05", facecolor=col, alpha=0.3, edgecolor=col, linewidth=1.2))
        ax.text(0.9, y + 0.1, cat, ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.annotate('', xy=(2.2, y + 0.1), xytext=(1.5, y + 0.1), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
        
        for j, v in enumerate(vec):
            x = 2.8 + j * 0.6
            if v == 1:
                rect = FancyBboxPatch((x, y-0.25), 0.5, 0.7, boxstyle="round,pad=0.05", facecolor=col, edgecolor=col, linewidth=1.2)
            else:
                rect = FancyBboxPatch((x, y-0.25), 0.5, 0.7, boxstyle="round,pad=0.05", facecolor=COLORS['bg_light'], edgecolor=COLORS['gray_medium'], linewidth=1)
            ax.add_patch(rect)
            ax.text(x + 0.25, y + 0.1, str(v), ha='center', va='center', fontsize=8, fontweight='bold' if v else 'normal')
    
    ax.text(2.5, 1.5, "exactly one '1'", ha='center', fontsize=8, style='italic')
    
    save_figure(fig, "fig_one_hot_encoding.pdf")


def plot_feature_scaling_comparison():
    """Figure 11: Before/after histograms for feature scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    
    np.random.seed(42)
    data = np.random.lognormal(mean=2, sigma=0.8, size=2000)
    
    ax1.hist(data, bins=25, color=COLORS['primary'], alpha=0.5, edgecolor=COLORS['border'])
    ax1.axvline(np.mean(data), color=COLORS['danger'], linestyle='-', linewidth=2, label=f'Mean = {np.mean(data):.1f}')
    ax1.axvline(np.mean(data) - np.std(data), color=COLORS['gray_medium'], linestyle='--', linewidth=1)
    ax1.axvline(np.mean(data) + np.std(data), color=COLORS['gray_medium'], linestyle='--', linewidth=1)
    ax1.set_xlabel('Value', fontsize=9)
    ax1.set_ylabel('Frequency', fontsize=9)
    ax1.set_title('Raw Distribution (skewed)', fontsize=10)
    ax1.legend(fontsize=7)
    style_axes(ax1)
    
    standardized = (data - np.mean(data)) / np.std(data)
    ax2.hist(standardized, bins=25, color=COLORS['primary'], alpha=0.5, edgecolor=COLORS['border'])
    ax2.axvline(0, color=COLORS['danger'], linestyle='-', linewidth=2, label='Mean = 0')
    ax2.axvline(-1, color=COLORS['gray_medium'], linestyle='--', linewidth=1)
    ax2.axvline(1, color=COLORS['gray_medium'], linestyle='--', linewidth=1, label='+/-1 sigma')
    ax2.set_xlabel('Value', fontsize=9)
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.set_title('Standardized (mu=0, sigma=1)', fontsize=10)
    ax2.legend(fontsize=7)
    style_axes(ax2)
    
    save_figure(fig, "fig_feature_scaling_comparison.pdf")


def plot_feature_binning():
    """Figure 12: Continuous distribution with bin partitioning."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    np.random.seed(42)
    data = np.random.normal(loc=14, scale=2, size=3000)
    
    ax.hist(data, bins=20, color=COLORS['primary'], alpha=0.5, edgecolor=COLORS['border'])
    
    bin_edges = [10, 12, 14, 16, 18]
    for edge in bin_edges:
        ax.axvline(edge, color=COLORS['danger'], linestyle='-', linewidth=2, alpha=0.7)
    
    for i in range(len(bin_edges)-1):
        ax.axvspan(bin_edges[i], bin_edges[i+1], alpha=0.1, color=COLORS['secondary'] if i % 2 == 0 else COLORS['primary'])
    
    ymax = ax.get_ylim()[1]
    ax.text(11, ymax * 0.85, "Bin 0", ha='center', fontsize=7)
    ax.text(13, ymax * 0.85, "Bin 1", ha='center', fontsize=7)
    ax.text(15, ymax * 0.85, "Bin 2", ha='center', fontsize=7)
    ax.text(17, ymax * 0.85, "Bin 3", ha='center', fontsize=7)
    
    ax.set_xlabel('Voltage', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Voltage Distribution with Bin Boundaries', fontsize=10)
    style_axes(ax)

    from matplotlib.lines import Line2D as _L2D
    legend_elements = [
        _L2D([0], [0], color=COLORS['danger'], lw=2, label='Bin edges'),
        _L2D([0], [0], color=COLORS['primary'], lw=6, alpha=0.3, label='Histogram'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    save_figure(fig, "fig_feature_binning.pdf")


def plot_interaction_features():
    """Figure 13: Decision boundary before/after interaction feature."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)
    y = ((x1 * x2) > 0).astype(int)
    
    colors = [COLORS['primary'], COLORS['danger']]
    for i in range(2):
        mask = y == i
        ax1.scatter(x1[mask], x2[mask], c=colors[i], label=f'Class {i}', alpha=0.6, edgecolors='white')
    
    x_line = np.linspace(-1, 1, 100)
    ax1.plot(x_line, np.zeros_like(x_line), 'k--', linewidth=1.5, label='Linear boundary')
    ax1.plot(np.zeros_like(x_line), x_line, 'k--', linewidth=1.5)
    
    ax1.set_xlabel('Feature X1', fontsize=9)
    ax1.set_ylabel('Feature X2', fontsize=9)
    ax1.set_title('Linear Boundary (no interaction)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper right')
    style_axes(ax1)
    
    for i in range(2):
        mask = y == i
        ax2.scatter(x1[mask], x2[mask], c=colors[i], label=f'Class {i}', alpha=0.6, edgecolors='white')
    
    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Z = (xx * yy).reshape(xx.shape)
    ax2.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=[COLORS['primary'], COLORS['danger']], alpha=0.2)
    ax2.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    ax2.set_xlabel('Feature X1', fontsize=9)
    ax2.set_ylabel('Feature X2', fontsize=9)
    ax2.set_title('Non-linear (with X1 x X2)', fontsize=9)
    ax2.legend(fontsize=7, loc='upper right')
    style_axes(ax2)
    
    save_figure(fig, "fig_interaction_features.pdf")


def plot_csr_format():
    """Figure 14: CSR sparse matrix format diagram."""
    fig, ax = create_figure(6.5, 3.5)
    
    ax.text(1.5, 8.5, "Dense Matrix (4x5)", ha='center', fontsize=9, fontweight='bold')
    
    dense = np.array([[0, 3, 0, 1, 0], [0, 0, 0, 4, 0], [2, 0, 0, 0, 5], [0, 0, 0, 0, 0]])
    for i in range(4):
        for j in range(5):
            x = 0.5 + j * 0.5
            y = 6.5 - i * 0.5
            if dense[i, j] != 0:
                rect = FancyBboxPatch((x, y), 0.4, 0.4, boxstyle="square", facecolor=COLORS['primary'], edgecolor=COLORS['primary'], alpha=0.5)
            else:
                rect = FancyBboxPatch((x, y), 0.4, 0.4, boxstyle="square", facecolor=COLORS['bg_light'], edgecolor=COLORS['gray_light'])
            ax.add_patch(rect)
            if dense[i, j] != 0:
                ax.text(x + 0.2, y + 0.2, str(dense[i, j]), ha='center', va='center', fontsize=6, fontweight='bold')
    
    ax.text(5.5, 8.5, "CSR Arrays", ha='center', fontsize=9, fontweight='bold')
    
    data = [3, 1, 4, 2, 5]
    indices = [1, 3, 0, 2, 4]
    indptr = [0, 2, 2, 4, 5]
    arrays = [("data", data), ("indices", indices), ("indptr", indptr)]
    
    for arr_idx, (name, arr) in enumerate(arrays):
        y = 6.5 - arr_idx * 1.2
        ax.text(4.8, y + 0.2, name + ":", ha='right', fontsize=8, fontweight='bold')
        
        for j, v in enumerate(arr):
            x = 5.0 + j * 0.4
            rect = FancyBboxPatch((x, y), 0.35, 0.4, boxstyle="square", facecolor=COLORS['bg_box'], edgecolor=COLORS['border'], linewidth=1)
            ax.add_patch(rect)
            ax.text(x + 0.175, y + 0.2, str(v), ha='center', va='center', fontsize=6)
    
    arrows = [(1.5, 6.7, 5.0, 6.7), (2.0, 6.2, 5.0, 5.5), (2.5, 5.7, 5.0, 4.3)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=0.8))
    
    ax.text(3.0, 3.5, "non-zero values", ha='center', fontsize=7, color=COLORS['primary'])
    ax.text(3.0, 2.8, "column indices", ha='center', fontsize=7, color=COLORS['gray_medium'])
    ax.text(3.0, 2.1, "row pointers", ha='center', fontsize=7, color=COLORS['secondary'])
    
    save_figure(fig, "fig_csr_format.pdf")


def plot_fuzzy_matching():
    """Figure 15: Edit distance visualization."""
    fig, ax = create_figure(5.0, 3.0)
    
    pairs = [("engine", "engne", "transposition"), ("voltage", "votage", "deletion"), ("moisture", "moister", "substitution")]
    colors = [COLORS['success'], COLORS['secondary'], COLORS['danger']]
    
    ax.text(2.5, 8.5, "Edit Distance", ha='center', fontsize=10, fontweight='bold')
    
    for idx, (s1, s2, op) in enumerate(pairs):
        y = 6.5 - idx * 1.8
        
        ax.text(0.3, y + 0.4, s1, ha='left', fontsize=8, fontweight='bold', family='monospace')
        
        for j, c in enumerate(s1[:min(len(s1),6)]):
            x = 1.8 + j * 0.35
            rect = FancyBboxPatch((x, y), 0.3, 0.4, boxstyle="square", facecolor=COLORS['success'], alpha=0.3, edgecolor=COLORS['success'])
            ax.add_patch(rect)
            ax.text(x + 0.15, y + 0.2, c, ha='center', va='center', fontsize=7)
        
        ax.annotate('', xy=(3.5, y + 0.2), xytext=(2.8, y + 0.2), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
        ax.text(3.15, y + 0.4, "d=1", ha='center', fontsize=7, color=colors[idx])
        
        for j, c in enumerate(s2[:min(len(s2),6)]):
            x = 4.0 + j * 0.35
            if j < len(s1) and c == s1[j]:
                color = COLORS['success']
                alpha = 0.3
            else:
                color = colors[idx]
                alpha = 0.3
            rect = FancyBboxPatch((x, y), 0.3, 0.4, boxstyle="square", facecolor=color, alpha=alpha, edgecolor=color)
            ax.add_patch(rect)
            ax.text(x + 0.15, y + 0.2, c, ha='center', va='center', fontsize=7)
        
        ax.text(4.0, y - 0.3, f"({op})", ha='left', fontsize=6, style='italic', color=COLORS['gray_medium'])
    
    ax.text(2.5, 1.0, "Similarity = 1 - lev/max(|a|,|b|)", ha='center', fontsize=7, style='italic')
    
    save_figure(fig, "fig_fuzzy_matching.pdf")


def plot_rule_engine_flow():
    """Figure 16: Rule engine flowchart."""
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0.3, 10.0)

    ax.text(4.0, 9.6, "Rule Engine Flow", ha='center', fontsize=11, fontweight='bold')

    # Input diamond at top
    add_diamond(ax, 3.0, 8.8, 0.5, "Input")

    rules = [
        ("V > 16V?", "CF"),
        ("V < 11V?", "CF"),
        ("moisture?", "CF"),
        ("DTC = P?", "PF"),
    ]

    d_size = 0.45
    col_x = 3.0
    row_ys = [7.5 - i * 1.7 for i in range(len(rules))]

    # Arrow from Input to first rule
    ax.annotate('', xy=(col_x, row_ys[0] + d_size), xytext=(col_x, 8.8 - 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))

    for i, (rule_text, outcome) in enumerate(rules):
        y = row_ys[i]
        add_diamond(ax, col_x, y, d_size, rule_text)

        # "Yes" → right to output box
        ax.annotate('', xy=(5.2, y), xytext=(col_x + d_size, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.2))
        ax.text(col_x + d_size + 0.1, y + 0.2, "Yes", fontsize=7, color=COLORS['success'], fontweight='bold')
        add_box(ax, 5.2, y - 0.3, 1.5, 0.6, f"Return {outcome}",
                facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')

        # "No" → down to next rule
        if i < len(rules) - 1:
            ax.annotate('', xy=(col_x, row_ys[i + 1] + d_size), xytext=(col_x, y - d_size),
                        arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.2))
            ax.text(col_x + 0.1, (y + row_ys[i + 1]) / 2, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')

    # "No" from last rule → Default/ML
    last_y = row_ys[-1]
    ax.annotate('', xy=(col_x, 1.1), xytext=(col_x, last_y - d_size),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.2))
    ax.text(col_x + 0.1, last_y - d_size - 0.5, "No", fontsize=7, color=COLORS['danger'], fontweight='bold')
    add_box(ax, col_x - 1.0, 0.5, 2.0, 0.6, "Default / ML",
            facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=8, fontweight='bold')

    save_figure(fig, "fig_rule_engine_flow.pdf")


def plot_confidence_thresholds():
    """Figure 17: Confidence threshold regions."""
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    
    ax.axvspan(0, 65, alpha=0.3, color=COLORS['danger'], label='Manual Review')
    ax.axvspan(65, 85, alpha=0.3, color=COLORS['secondary'], label='Cautious')
    ax.axvspan(85, 100, alpha=0.3, color=COLORS['success'], label='Firm')
    
    ax.axvline(65, color=COLORS['danger'], linestyle='--', linewidth=2)
    ax.axvline(85, color=COLORS['success'], linestyle='--', linewidth=2)
    
    ax.text(32.5, 7.5, "Manual Review", ha='center', fontsize=9, fontweight='bold', color=COLORS['danger'])
    ax.text(75, 7.5, "Cautious", ha='center', fontsize=9, fontweight='bold', color=COLORS['secondary'])
    ax.text(92.5, 7.5, "Firm", ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    
    ax.text(65, 1.5, "tau_m = 65%", ha='center', fontsize=7, color=COLORS['danger'])
    ax.text(85, 1.5, "tau_f = 85%", ha='center', fontsize=7, color=COLORS['success'])
    
    ax.scatter([40, 70, 90], [4.5, 4.5, 4.5], c=[COLORS['danger'], COLORS['secondary'], COLORS['success']], s=80, zorder=5)
    
    ax.set_xlabel('Confidence Score (%)', fontsize=9)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    save_figure(fig, "fig_confidence_thresholds.pdf")


def plot_weighted_blending():
    """Figure 18: Weighted score blending bar chart."""
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    
    sources = ['Rule', 'ML', 'LLM', 'Bonus']
    agree = [30, 50, 20, 5]
    disagree = [20, 30, 15, 0]
    
    x = np.arange(2)
    width = 0.3
    
    bottom1 = np.zeros(2)
    for i, (s, c) in enumerate(zip(sources[:3], [COLORS['primary'], COLORS['success'], COLORS['purple']])):
        ax.bar(x - width/2, agree[i], width, bottom=bottom1, label=s, color=c)
        bottom1 += agree[i]
    
    ax.bar(x - width/2, agree[3], width, bottom=bottom1, label='Agreement Bonus', color=COLORS['secondary'])
    
    bottom2 = np.zeros(2)
    for i, (s, c) in enumerate(zip(sources[:3], [COLORS['primary'], COLORS['success'], COLORS['purple']])):
        ax.bar(x + width/2, disagree[i], width, bottom=bottom2, color=c)
        bottom2 += disagree[i]
    
    ax.text(0, 108, "85%", ha='center', fontsize=9, fontweight='bold')
    ax.text(1, 68, "65%", ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Sources Agree', 'Sources Disagree'], fontsize=8)
    ax.set_ylabel('Confidence (%)', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(0, 120)
    style_axes(ax)
    
    save_figure(fig, "fig_weighted_blending.pdf")


def plot_geometric_vs_arithmetic_mean():
    """Figure 19: Geometric vs arithmetic mean comparison."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    x = np.linspace(0.1, 0.9, 100)
    arithmetic = (x + 0.5) / 2
    geometric = np.sqrt(x * 0.5)
    
    ax.plot(x * 100, arithmetic * 100, color=COLORS['primary'], linewidth=2, label='Arithmetic Mean')
    ax.plot(x * 100, geometric * 100, color=COLORS['danger'], linewidth=2, label='Geometric Mean')
    ax.fill_between(x * 100, geometric * 100, arithmetic * 100, alpha=0.2, color=COLORS['secondary'], label='AM-GM Gap')
    
    ax.axvline(50, color=COLORS['gray_medium'], linestyle='--', linewidth=1)
    ax.text(55, 72, 'Max gap here', ha='left', fontsize=7, style='italic')
    
    ax.set_xlabel('Input a (%)', fontsize=9)
    ax.set_ylabel('Mean Value (%)', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    style_axes(ax)
    
    ax.text(0.5, 0.02, 'G(a,b) = sqrt(a*b)', ha='center', fontsize=7, transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.95, 'A(a,b) = (a+b)/2', ha='center', fontsize=7, transform=ax.transAxes, style='italic')
    
    save_figure(fig, "fig_geometric_vs_arithmetic_mean.pdf")


def plot_transformer_architecture():
    """Figure 20: Transformer architecture block diagram.

    Shows a single encoder block with Multi-Head Attention, Feed Forward,
    Add & Norm sublayers, and residual (skip) connections. The '× N' bracket
    indicates that this block is repeated N times.
    """
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)

    ax.text(4.0, 11.5, "Transformer Encoder Architecture", ha='center', fontsize=12, fontweight='bold')

    bx, bw = 1.5, 3.5  # box x and width
    bh = 0.7  # box height
    cx = bx + bw / 2   # centre x

    # Input
    add_box(ax, bx, 0.8, bw, bh, "Input + Positional Encoding",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], fontsize=8)
    ax.annotate('', xy=(cx, 2.0), xytext=(cx, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # === Encoder block (inside bracket) ===
    block_bot = 2.0
    block_top = 9.0

    # Layer 1: Multi-Head Attention
    mha_y = 2.2
    add_box(ax, bx, mha_y, bw, bh, "Multi-Head Attention",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')

    # Residual arrow 1 (skip around MHA + Add&Norm)
    ax.annotate('', xy=(bx - 0.15, mha_y + bh / 2), xytext=(bx - 0.15, mha_y + bh + 1.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.0, ls='--'))
    ax.text(bx - 0.35, mha_y + bh + 0.3, "residual", ha='center', fontsize=5, rotation=90,
            color=COLORS['secondary'], style='italic')

    ax.annotate('', xy=(cx, mha_y + bh + 0.3), xytext=(cx, mha_y + bh),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))

    # Add & Norm 1
    an1_y = 3.5
    add_box(ax, bx, an1_y, bw, bh, "Add & Norm",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontweight='bold')

    ax.annotate('', xy=(cx, an1_y + bh + 0.3), xytext=(cx, an1_y + bh),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))

    # Layer 2: Feed Forward
    ff_y = 4.8
    add_box(ax, bx, ff_y, bw, bh, "Feed-Forward Network",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    # Residual arrow 2 (skip around FFN + Add&Norm)
    ax.annotate('', xy=(bx - 0.15, ff_y + bh / 2), xytext=(bx - 0.15, ff_y + bh + 1.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.0, ls='--'))
    ax.text(bx - 0.35, ff_y + bh + 0.3, "residual", ha='center', fontsize=5, rotation=90,
            color=COLORS['secondary'], style='italic')

    ax.annotate('', xy=(cx, ff_y + bh + 0.3), xytext=(cx, ff_y + bh),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))

    # Add & Norm 2
    an2_y = 6.1
    add_box(ax, bx, an2_y, bw, bh, "Add & Norm",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=8, fontweight='bold')

    # Upward arrow to sublayer descriptions
    ax.annotate('', xy=(cx, an2_y + bh + 0.3), xytext=(cx, an2_y + bh),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0))

    # Dropout / layer output
    drop_y = 7.4
    add_box(ax, bx, drop_y, bw, bh, "Dropout",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['gray_medium'], fontsize=8)

    # === "× N" bracket on right side ===
    bracket_x = bx + bw + 0.4
    ax.plot([bracket_x, bracket_x + 0.2, bracket_x + 0.2, bracket_x],
            [block_bot, block_bot, block_top - 0.5, block_top - 0.5],
            color=COLORS['gray_medium'], lw=1.5)
    ax.text(bracket_x + 0.5, (block_bot + block_top - 0.5) / 2, "× N",
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['gray_medium'])

    # Arrow up to output
    ax.annotate('', xy=(cx, drop_y + bh + 0.5), xytext=(cx, drop_y + bh),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Output
    add_box(ax, bx, 8.7, bw, bh, "Output Representations",
            facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], fontsize=8)

    # Sublayer descriptions on the right
    ax.text(bx + bw + 0.4, mha_y + bh / 2, r"$\mathrm{Attn}(Q,K,V)$",
            ha='left', va='center', fontsize=7, color=COLORS['primary'], style='italic')
    ax.text(bx + bw + 0.4, ff_y + bh / 2, r"$\mathrm{FFN}(x) = W_2 \sigma(W_1 x)$",
            ha='left', va='center', fontsize=7, color=COLORS['success'], style='italic')

    save_figure(fig, "fig_transformer_architecture.pdf")


def plot_prompt_structure():
    """Figure 21: Prompt structure anatomy.

    Shows the five sections of an LLM prompt used in TRACE, stacked
    vertically. Each section has a title on the left and example content
    on the right, all inside the same box so nothing is hidden.
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0.0, 11.0)

    ax.text(5.0, 10.5, "LLM Prompt Structure", ha='center', fontsize=13, fontweight='bold')

    sections = [
        ("1. System Role", "You are an automotive warranty analyst.\nAnalyse the claim and return a structured JSON.", COLORS['primary']),
        ("2. Input Data", "fault_code: P0562\ntechnician_notes: \"engine overheating\"\nvoltage: 15.8", COLORS['success']),
        ("3. Constraints", "category must be one of:\n{moisture_damage, physical_damage, ntf,\n electrical_issue, engine_symptom, other}", COLORS['secondary']),
        ("4. Disambiguation", "If notes mention overheat AND DTC starts\nwith P -> classify as electrical_issue", COLORS['purple']),
        ("5. Output Format", '{"category": "electrical_issue",\n "confidence": 0.92, "reasoning": "..."}', COLORS['brown']),
    ]

    bh = 1.55
    gap = 0.15
    y = 9.0
    label_w = 2.2
    content_w = 6.8

    for title, content, color in sections:
        # Title column (left)
        add_box(ax, 0.5, y - bh, label_w, bh, title,
                facecolor=color, edgecolor=color, fontsize=8, fontweight='bold', fontcolor='white')
        # Content column (right)
        add_box(ax, 0.5 + label_w + 0.1, y - bh, content_w, bh, "",
                facecolor=COLORS['bg_box'], edgecolor=color, fontsize=7)
        ax.text(0.5 + label_w + 0.35, y - bh / 2, content,
                ha='left', va='center', fontsize=7.5, family='monospace')
        y -= bh + gap

    # Down arrow on the left side to show order
    ax.annotate('', xy=(1.6, 0.8), xytext=(1.6, 9.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.5, ls='--'))
    ax.text(0.2, 5.0, "Top\nto\nBottom", ha='center', va='center', fontsize=7,
            color=COLORS['gray_medium'], style='italic', rotation=90)

    save_figure(fig, "fig_prompt_structure.pdf")


def plot_temperature_effect():
    """Figure 22: Temperature effect on probability distribution."""
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 3.0))
    
    logits = np.array([3.0, 1.0, 0.5, 0.2])
    tokens = ['A', 'B', 'C', 'D']
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    for ax, T in zip(axes, temperatures):
        probs = np.exp(logits / T) / np.sum(np.exp(logits / T))
        
        colors_bar = [COLORS['primary'] if p == max(probs) else COLORS['gray_medium'] for p in probs]
        ax.bar(tokens, probs, color=colors_bar, alpha=0.7, edgecolor=COLORS['border'])
        
        ax.set_title(f'T = {T}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_ylim(0, 1.1)
        style_axes(ax)
    
    fig.text(0.5, 0.02, 'T=0.1 (Greedy) to T=2.0 (Random)', ha='center', fontsize=8, style='italic')
    
    save_figure(fig, "fig_temperature_effect.pdf")


def plot_api_integration_flow():
    """Figure 23: API integration flow diagram."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.axis('off')
    ax.set_xlim(0, 9)
    ax.set_ylim(1.5, 7.5)

    ax.text(4.5, 7.0, "LLM API Integration Flow", ha='center', fontsize=11, fontweight='bold')

    # Client App
    add_box(ax, 0.5, 4.8, 1.8, 0.9, "Client App",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')

    # Arrow with label
    ax.annotate('', xy=(3.0, 5.25), xytext=(2.3, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))
    ax.text(2.65, 5.65, "POST /v1/chat/completions", ha='center', fontsize=6, color=COLORS['secondary'])
    ax.text(2.65, 4.55, "Authorization: Bearer ...", ha='center', fontsize=5.5, color=COLORS['gray_medium'], style='italic')

    # HTTP POST box
    add_box(ax, 3.0, 4.8, 2.2, 0.9, "HTTP POST\n/chat/completions",
            facecolor=COLORS['gray_medium'], edgecolor=COLORS['gray_medium'], fontsize=7, fontweight='bold')

    # Arrow to server
    ax.annotate('', xy=(5.8, 5.25), xytext=(5.2, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # API Server
    add_box(ax, 5.8, 4.8, 1.8, 0.9, "API Server\n(LLM)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    # Response arrow down
    ax.annotate('', xy=(5.0, 3.5), xytext=(6.7, 4.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.2, ls='--'))

    # JSON Response
    add_box(ax, 3.2, 2.5, 2.5, 0.9, "JSON Response\n{category, confidence, ...}",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=7, fontweight='bold')

    save_figure(fig, "fig_api_integration_flow.pdf")


def plot_exponential_backoff():
    """Figure 24: Exponential backoff timeline."""
    fig, ax = create_figure(6.5, 2.5)
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 10)
    
    ax.plot([0, 6.5], [7.5, 7.5], 'k-', linewidth=2)
    
    attempts = [(0, 'Attempt 0', COLORS['danger']), (1, 'Attempt 1', COLORS['danger']), (3, 'Attempt 2', COLORS['danger']), (6, 'Attempt 3', COLORS['success'])]
    
    for x, label, color in attempts:
        ax.scatter([x], [7.5], c=color, s=150, zorder=5)
        ax.text(x, 8.8, label, ha='center', fontsize=7, fontweight='bold')
    
    delays = [(0.5, '1s'), (2, '2s'), (4.5, '4s')]
    for x, label in delays:
        ax.text(x, 6.7, label, ha='center', fontsize=6, color=COLORS['secondary'], fontweight='bold')
    
    ax.text(3.0, 1.5, "delay = base * 2^k", ha='center', fontsize=8, style='italic')
    ax.text(3.0, 1.0, "Total: 7s", ha='center', fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    save_figure(fig, "fig_exponential_backoff.pdf")


def plot_fallback_chain():
    """Figure 25: Fallback chain decision tree."""
    fig, ax = create_figure(5.0, 4.0)
    
    levels = [
        (8.0, "Primary: LLM API", COLORS['primary']),
        (6.0, "Fallback 1: Rules", COLORS['secondary']),
        (4.0, "Fallback 2: ML", COLORS['success']),
        (2.0, "Default: Safe", COLORS['gray_medium']),
    ]
    
    for y, label, color in levels:
        add_box(ax, 1.3, y - 0.4, 2.6, 0.7, label, facecolor=color, edgecolor=color, fontsize=8, fontweight='bold')
        
        if y > 2.0:
            ax.annotate('', xy=(2.65, y - 0.55), xytext=(2.65, y - 0.85), arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.2))
            ax.text(2.85, y - 0.7, "fail", ha='left', fontsize=6, color=COLORS['danger'])
    
    ax.text(0.3, 5.0, "Most capable", ha='center', fontsize=6, style='italic', color=COLORS['primary'])
    ax.text(0.3, 1.0, "Most reliable", ha='center', fontsize=6, style='italic', color=COLORS['gray_medium'])
    
    ax.annotate('', xy=(0.9, 5.0), xytext=(0.9, 2.0), arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=0.8, ls='--'))
    ax.text(0.2, 3.5, "Cap↓", ha='center', fontsize=5, rotation=90, color=COLORS['gray_medium'])
    ax.text(0.2, 3.0, "Rel↑", ha='center', fontsize=5, rotation=90, color=COLORS['gray_medium'])
    
    save_figure(fig, "fig_fallback_chain.pdf")


def plot_precision_recall_f1():
    """Figure 26: Precision, Recall, F1 bar chart."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    metrics = ['Precision', 'Recall', 'F1']
    values = [0.88, 0.82, 0.85]
    colors_bar = [COLORS['primary'], COLORS['success'], COLORS['purple']]
    
    bars = ax.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor=COLORS['border'], linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    ax.axhline(0.85, color=COLORS['purple'], linestyle='--', linewidth=1.5, label='F1 threshold')

    ax.set_ylabel('Score', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title('Precision, Recall, F1 Scores', fontsize=10, fontweight='bold')
    style_axes(ax)
    ax.legend(fontsize=7, loc='lower right')

    ax.text(1.5, 0.15, r'$F_1 = \frac{2 \cdot P \cdot R}{P + R}$', ha='center', fontsize=9)

    save_figure(fig, "fig_precision_recall_f1.pdf")


def plot_confusion_matrix():
    """Figure 27: Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    
    matrix = np.array([
        [142, 5, 3, 2],
        [8, 89, 12, 5],
        [4, 10, 95, 6],
        [3, 7, 8, 82]
    ])
    
    labels = ['Prod Fail', 'Cust Fail', 'NTF', 'Electrical']
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    for i in range(4):
        for j in range(4):
            color = 'white' if matrix[i, j] > 70 else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha='right')
    ax.set_yticklabels(labels, fontsize=7)
    
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    
    save_figure(fig, "fig_confusion_matrix.pdf")


def plot_feature_importance():
    """Figure 28: Feature importance horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    
    features = ['dtc_prefix_P', 'voltage', 'notes_len', 'tfidf_eng', 'dtc_count', 'ohe_supplier', 'volt_bin', 'notes_mst']
    importance = [0.28, 0.19, 0.14, 0.11, 0.09, 0.07, 0.06, 0.06]
    
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, importance, color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['border'])
    
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=7)
    ax.set_xlabel('Gini Importance', fontsize=9)
    ax.invert_yaxis()
    style_axes(ax)
    
    ax2 = ax.twiny()
    cumulative = np.cumsum(importance)
    ax2.plot(cumulative, y_pos, color=COLORS['danger'], linewidth=2, marker='o', markersize=3, label='Cumulative')
    ax2.set_xlabel('Cumulative %', fontsize=7)
    ax2.set_xlim(0, 1.2)

    from matplotlib.lines import Line2D as _L2D
    legend_elements = [
        _L2D([0], [0], color=COLORS['primary'], lw=6, alpha=0.7, label='Gini importance'),
        _L2D([0], [0], color=COLORS['danger'], lw=2, marker='o', markersize=3, label='Cumulative'),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc='lower right')

    save_figure(fig, "fig_feature_importance.pdf")


def plot_calibration_curve():
    """Figure 29: Calibration curve."""
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    
    mean_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    frac_pos = np.array([0.08, 0.18, 0.32, 0.42, 0.55, 0.68, 0.78, 0.88, 0.94])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect')
    ax.plot(mean_pred, frac_pos, 'o-', color=COLORS['primary'], linewidth=2, markersize=6, label='Model')
    
    ax.fill_between(mean_pred, mean_pred, frac_pos, alpha=0.2, color=COLORS['secondary'])
    
    ax.set_xlabel('Mean Predicted Prob', fontsize=9)
    ax.set_ylabel('Fraction of Positives', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)
    
    ax.text(0.6, 0.15, 'ECE = 0.042', ha='center', fontsize=8, fontweight='bold', color=COLORS['purple'])
    
    save_figure(fig, "fig_calibration_curve.pdf")


def plot_distributions():
    """Figure 30: Distribution comparison."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    x = np.linspace(-4, 6, 500)
    
    normal = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    truncnorm = np.where((x >= -2) & (x <= 4), (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2), 0)
    truncnorm = truncnorm / (truncnorm.sum() * (x[1] - x[0]) + 1e-10)
    lognorm = (1 / (np.maximum(x, 0.01) * np.sqrt(2 * np.pi) * 0.8)) * np.exp(-0.5 * ((np.log(np.maximum(x, 0.01)) - 2) / 0.8)**2)
    lognorm = np.where(x > 0, lognorm, 0)
    
    ax.plot(x, normal, color=COLORS['primary'], linewidth=2, label='Normal')
    ax.plot(x, truncnorm, color=COLORS['success'], linewidth=2, label='Truncated')
    ax.plot(x, lognorm, color=COLORS['secondary'], linewidth=2, label='Log-normal')
    
    ax.fill_between(x, normal, alpha=0.2, color=COLORS['primary'])
    ax.fill_between(x, truncnorm, alpha=0.2, color=COLORS['success'])
    
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('PDF', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    style_axes(ax)
    
    save_figure(fig, "fig_distributions.pdf")


def plot_weighted_sampling():
    """Figure 31: Weighted sampling comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    
    categories = ['A', 'B', 'C', 'D', 'E']
    weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    ax1.bar(categories, weights, color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['border'])
    ax1.set_ylabel('Probability', fontsize=9)
    ax1.set_title('Target Weights', fontsize=10)
    style_axes(ax1)
    
    np.random.seed(42)
    samples = np.random.choice(categories, size=1000, p=weights)
    actual = [np.sum(samples == c) / 1000 for c in categories]
    
    ax2.bar(categories, actual, color=COLORS['success'], alpha=0.7, edgecolor=COLORS['border'])
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.set_title('Actual (n=1000)', fontsize=10)
    style_axes(ax2)
    
    save_figure(fig, "fig_weighted_sampling.pdf")


def plot_mixture_distribution():
    """Figure 32: Mixture distribution."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    x = np.linspace(0, 25, 500)
    
    mu1, sigma1 = 10, 2
    mu2, sigma2 = 18, 3
    
    comp1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    comp2 = (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    mixture = 0.6 * comp1 + 0.4 * comp2
    
    ax.plot(x, comp1, color=COLORS['primary'], linewidth=1.5, linestyle='--', label=f'N({mu1},{sigma1})')
    ax.plot(x, comp2, color=COLORS['secondary'], linewidth=1.5, linestyle='--', label=f'N({mu2},{sigma2})')
    ax.plot(x, mixture, color=COLORS['danger'], linewidth=2.5, label='Mixture')
    
    ax.fill_between(x, comp1, alpha=0.2, color=COLORS['primary'])
    ax.fill_between(x, comp2, alpha=0.2, color=COLORS['secondary'])
    
    ax.axvline(mu1, color=COLORS['primary'], linestyle=':', linewidth=1.5)
    ax.axvline(mu2, color=COLORS['secondary'], linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    style_axes(ax)
    
    ax.text(0.5, 0.95, 'f(x) = 0.6*N1 + 0.4*N2', transform=ax.transAxes, ha='center', fontsize=8, style='italic')
    
    save_figure(fig, "fig_mixture_distribution.pdf")


def plot_correlated_features():
    """Figure 33: Correlated features scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    
    np.random.seed(42)
    correlations = [0.8, 0, -0.8]
    titles = ['rho = +0.8', 'rho = 0', 'rho = -0.8']
    
    for ax, rho, title in zip(axes, correlations, titles):
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 200).T
        
        ax.scatter(x, y, c=COLORS['primary'], alpha=0.5, s=15)
        
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x), p(sorted(x)), color=COLORS['danger'], linewidth=1.5)
        
        ax.text(0.05, 0.9, title, transform=ax.transAxes, fontsize=9, fontweight='bold')
        ax.set_xlabel('X1', fontsize=8)
        ax.set_ylabel('X2', fontsize=8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        style_axes(ax)
    
    save_figure(fig, "fig_correlated_features.pdf")


def plot_pipeline_architecture():
    """Figure 34: Pipeline architecture."""
    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    ax.axis('off')
    ax.set_xlim(0, 11)
    ax.set_ylim(3.5, 7.5)

    ax.text(5.5, 7.0, "ML Pipeline Architecture", ha='center', fontsize=11, fontweight='bold')

    stages = [
        ("1. Input", "Raw data", COLORS['primary']),
        ("2. Preprocess", "Clean", COLORS['primary']),
        ("3. Features", "Vector", COLORS['success']),
        ("4. Inference", "Preds", COLORS['purple']),
        ("5. Post-proc", "Output", COLORS['secondary']),
        ("6. Output", "JSON", COLORS['success']),
    ]

    bw, bh = 1.4, 0.8
    gap = 0.25
    total_w = len(stages) * bw + (len(stages) - 1) * gap
    x0 = (11 - total_w) / 2

    for i, (label, fmt, color) in enumerate(stages):
        x = x0 + i * (bw + gap)
        add_box(ax, x, 5.0, bw, bh, label, facecolor=COLORS['bg_box'], edgecolor=color, fontsize=7, fontweight='bold')
        ax.text(x + bw / 2, 4.7, fmt, ha='center', fontsize=6, family='monospace', color=COLORS['gray_medium'])
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + bw + gap, 5.4), xytext=(x + bw, 5.4),
                        arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    save_figure(fig, "fig_pipeline_architecture.pdf")


def plot_hybrid_engine():
    """Figure 35: Hybrid engine diagram."""
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 8.5)

    ax.text(5.0, 8.0, "Hybrid Decision Engine", ha='center', fontsize=11, fontweight='bold')

    sources = [
        (0.5, 6.2, "Rule Engine", "f_r(x)", COLORS['primary']),
        (0.5, 4.7, "ML Classifier", "f_ml(x)", COLORS['success']),
        (0.5, 3.2, "LLM Layer", "f_llm(x)", COLORS['purple']),
    ]

    combiner_x = 4.5
    for x, y, label, eq, color in sources:
        add_box(ax, x, y, 2.0, 0.7, label, facecolor=color, edgecolor=color, fontsize=8, fontweight='bold')
        ax.text(x + 1.0, y - 0.3, eq, ha='center', fontsize=6, family='monospace')
        ax.annotate('', xy=(combiner_x, 4.85), xytext=(x + 2.0, y + 0.35),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    add_box(ax, combiner_x, 4.3, 2.0, 1.1, "Weighted\nCombiner",
            facecolor=COLORS['secondary'], edgecolor=COLORS['secondary'], fontsize=9, fontweight='bold')
    ax.text(combiner_x + 1.0, 3.95, "Agreement Logic", ha='center', fontsize=6.5,
            style='italic', color=COLORS['gray_medium'])

    ax.annotate('', xy=(7.2, 4.85), xytext=(combiner_x + 2.0, 4.85),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    add_box(ax, 7.2, 4.3, 2.0, 1.1, "Final\nDecision",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=9, fontweight='bold')

    save_figure(fig, "fig_hybrid_engine.pdf")


def plot_model_serialization():
    """Figure 36: Model serialization flow."""
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.axis('off')
    ax.set_xlim(0, 9)
    ax.set_ylim(2.0, 7.5)

    ax.text(4.5, 7.0, "Model Serialization (pickle)", ha='center', fontsize=11, fontweight='bold')

    # Python Objects
    add_box(ax, 0.5, 4.5, 2.2, 1.0, "Python Objects\n(classifiers, transformers)",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=7, fontweight='bold')

    # Arrow: pickle.dump()
    ax.annotate('', xy=(3.5, 5.0), xytext=(2.7, 5.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.text(3.1, 5.65, "pickle.dump()", ha='center', fontsize=7, color=COLORS['success'], fontweight='bold')

    # .pkl File
    add_box(ax, 3.5, 4.5, 1.8, 1.0, ".pkl File",
            facecolor=COLORS['white'], edgecolor=COLORS['gray_dark'], fontsize=9, fontweight='bold')
    ax.text(4.4, 4.1, "trace_models.pkl", ha='center', fontsize=7, style='italic', color=COLORS['gray_dark'])

    # Arrow: pickle.load()
    ax.annotate('', xy=(6.1, 5.0), xytext=(5.3, 5.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.5))
    ax.text(5.7, 5.65, "pickle.load()", ha='center', fontsize=7, color=COLORS['danger'], fontweight='bold')

    # Restored Objects
    add_box(ax, 6.1, 4.5, 2.2, 1.0, "Restored Objects\n(classifiers, transformers)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')

    ax.text(4.5, 3.0, "14 components: 2 classifiers + 12 transformers",
            ha='center', fontsize=8, style='italic', color=COLORS['gray_medium'])

    save_figure(fig, "fig_model_serialization.pdf")


def plot_rest_api():
    """Figure 37: REST API diagram."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 8.5)

    ax.text(5.0, 8.0, "REST API: Request / Response", ha='center', fontsize=13, fontweight='bold')

    # Client box
    add_box(ax, 0.5, 5.0, 2.0, 1.2, "Client\n(Browser)",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=10, fontweight='bold', fontcolor='white')

    # Request arrow (top, left to right)
    ax.annotate('', xy=(4.0, 6.8), xytext=(1.5, 6.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2.0))
    ax.text(2.75, 7.1, "POST /analyze", ha='center', fontsize=9, fontweight='bold', color=COLORS['secondary'])
    ax.text(2.75, 6.45, "Content-Type: application/json", ha='center', fontsize=7,
            color=COLORS['gray_medium'])

    # API Server box
    add_box(ax, 4.0, 5.0, 2.5, 1.2, "API Server\n(FastAPI)",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=10, fontweight='bold', fontcolor='white')

    # Response arrow (bottom, right to left)
    ax.annotate('', xy=(1.5, 4.2), xytext=(5.25, 4.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2.0))
    ax.text(3.5, 3.85, "200 OK  +  JSON body", ha='center', fontsize=8,
            color=COLORS['purple'], fontweight='bold')

    # JSON Response box
    add_box(ax, 7.2, 5.0, 2.3, 1.2, "JSON Response\n{status, FA, WD,\nconfidence}",
            facecolor=COLORS['purple'], edgecolor=COLORS['purple'], fontsize=7, fontweight='bold', fontcolor='white')

    # Arrow from server to JSON response
    ax.annotate('', xy=(7.2, 5.6), xytext=(6.5, 5.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2.0))

    # Flow labels
    ax.text(2.75, 7.55, "REQUEST", ha='center', fontsize=7, color=COLORS['secondary'],
            fontweight='bold', style='italic')
    ax.text(3.5, 3.5, "RESPONSE", ha='center', fontsize=7, color=COLORS['purple'],
            fontweight='bold', style='italic')

    save_figure(fig, "fig_rest_api.pdf")


def plot_file_upload_flow():
    """Figure 38: File upload flow."""
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.axis('off')
    ax.set_xlim(0, 9)
    ax.set_ylim(3.0, 7.5)

    ax.text(4.5, 7.0, "File Upload Flow", ha='center', fontsize=11, fontweight='bold')

    steps = [
        (0.5, "Client\nBrowser", COLORS['primary']),
        (2.5, "multipart/\nform-data", COLORS['secondary']),
        (4.5, "Server\nParser", COLORS['success']),
        (6.5, "Processing", COLORS['purple']),
    ]

    for i, (x, label, color) in enumerate(steps):
        add_box(ax, x, 4.8, 1.6, 1.0, label, facecolor=color, edgecolor=color, fontsize=8, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(steps[i+1][0], 5.3), xytext=(x + 1.6, 5.3),
                        arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    ax.text(1.3, 4.5, "boundary=...", ha='center', fontsize=6, family='monospace', color=COLORS['gray_medium'])
    ax.text(3.5, 4.5, "file stream", ha='center', fontsize=7, style='italic')
    ax.text(5.5, 4.5, "result", ha='center', fontsize=7, style='italic')

    save_figure(fig, "fig_file_upload_flow.pdf")


def plot_ocr_pipeline():
    """Figure 39: OCR pipeline."""
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 8.0)

    ax.text(5.0, 7.5, "OCR Pipeline", ha='center', fontsize=11, fontweight='bold')

    # Input Image box (outer frame)
    outer = FancyBboxPatch((0.2, 4.6), 2.2, 1.8, boxstyle="round,pad=0.05",
                           facecolor=COLORS['bg_box'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.add_patch(outer)
    ax.text(1.3, 6.15, "Input Image", ha='center', fontsize=7, color=COLORS['gray_medium'])
    # Bounding boxes inside the image box (well within bounds)
    for xi, txt in [(0.45, "P0562"), (1.45, "14.2V")]:
        rect = FancyBboxPatch((xi, 5.0), 0.8, 0.5, boxstyle="square",
                              facecolor='white', edgecolor=COLORS['primary'], linewidth=1.2)
        ax.add_patch(rect)
        ax.text(xi + 0.4, 5.25, txt, ha='center', va='center', fontsize=7, fontweight='bold', family='monospace')

    ax.annotate('', xy=(2.9, 5.5), xytext=(2.4, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.5))

    # Text Detection
    add_box(ax, 2.9, 4.9, 1.8, 1.1, "Text\nDetection",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=9, fontweight='bold', fontcolor='white')

    ax.annotate('', xy=(5.2, 5.5), xytext=(4.7, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.5))

    # Character Recognition
    add_box(ax, 5.2, 4.9, 1.8, 1.1, "Character\nRecognition",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=9, fontweight='bold', fontcolor='white')

    ax.annotate('', xy=(7.5, 5.5), xytext=(7.0, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.5))

    # Output Text
    outer2 = FancyBboxPatch((7.5, 4.9), 1.8, 1.1, boxstyle="round,pad=0.05",
                            facecolor=COLORS['bg_box'], edgecolor=COLORS['success'], linewidth=1.5)
    ax.add_patch(outer2)
    ax.text(8.4, 5.55, '"P0562, 14.2V"', ha='center', va='center', fontsize=7, fontweight='bold', family='monospace')
    ax.text(8.4, 5.15, "Output Text", ha='center', fontsize=6.5, color=COLORS['gray_medium'])

    # Labels below
    ax.text(1.3, 4.3, "Input", ha='center', fontsize=7, color=COLORS['gray_medium'])
    ax.text(3.8, 4.3, r"$f_{detect}$", ha='center', fontsize=8, color=COLORS['primary'])
    ax.text(6.1, 4.3, r"$f_{recognize}$", ha='center', fontsize=8, color=COLORS['success'])

    save_figure(fig, "fig_ocr_pipeline.pdf")


def plot_logging_architecture():
    """Figure 40: Logging architecture hierarchy."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(1.0, 9.5)

    ax.text(4.0, 9.0, "Logging Architecture", ha='center', fontsize=11, fontweight='bold')

    # root Logger
    add_box(ax, 2.8, 7.5, 2.4, 0.7, "root Logger",
            facecolor=COLORS['gray_dark'], edgecolor=COLORS['gray_dark'], fontsize=8, fontweight='bold', fontcolor='white')

    # trace
    add_box(ax, 2.8, 6.0, 2.4, 0.7, "trace",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')
    ax.annotate('', xy=(4.0, 7.5), xytext=(4.0, 6.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Child loggers
    loggers = [(1.0, 4.5, "trace.ml"), (4.5, 4.5, "trace.llm")]
    for x, y, name in loggers:
        add_box(ax, x, y, 2.0, 0.7, name, facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=7)
        ax.annotate('', xy=(x + 1.0, 5.2), xytext=(4.0, 6.0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=0.8))

    # Handlers row
    ax.annotate('', xy=(4.0, 3.7), xytext=(4.0, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=0.8))

    handlers = [
        (0.5, 2.8, "StreamHandler", COLORS['success']),
        (3.0, 2.8, "FileHandler", COLORS['success']),
        (5.5, 2.8, "Formatter", COLORS['purple']),
    ]
    for x, y, name, color in handlers:
        add_box(ax, x, y, 2.0, 0.6, name, facecolor=color, edgecolor=color, fontsize=7, fontweight='bold')

    # Arrow from FileHandler to Formatter
    ax.annotate('', xy=(5.5, 3.1), xytext=(5.0, 3.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=0.8))

    save_figure(fig, "fig_logging_architecture.pdf")


def plot_docker_layers():
    """Figure 41: Docker layers diagram."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 9.5)

    ax.text(4.0, 9.0, "Docker Image Layers", ha='center', fontsize=11, fontweight='bold')

    layers = [
        (7.5, "ubuntu:22.04", "Base OS", COLORS['primary']),
        (6.3, "Python 3.11", "Runtime", COLORS['success']),
        (5.1, "pip install", "Dependencies", COLORS['secondary']),
        (3.9, "COPY . /app", "App Code", COLORS['purple']),
        (2.7, "CMD uvicorn", "Entrypoint", COLORS['brown']),
    ]

    bx, bw = 1.5, 3.0
    for y, cmd, label, color in layers:
        rect = FancyBboxPatch((bx, y - 0.35), bw, 0.7, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=color, alpha=0.3, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(bx + bw / 2, y, cmd, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(bx + bw + 0.3, y, label, ha='left', va='center', fontsize=8)

    # Image boundary box
    rect = FancyBboxPatch((1.3, 2.0), 3.4, 6.6, boxstyle="round,pad=0.05",
                          facecolor='none', edgecolor=COLORS['border'], linewidth=2)
    ax.add_patch(rect)
    ax.text(3.0, 2.25, "Read-only Image", ha='center', fontsize=7, style='italic', color=COLORS['gray_medium'])

    # Container layer
    rect = FancyBboxPatch((1.3, 1.2), 3.4, 0.6, boxstyle="round,pad=0.05",
                          facecolor=COLORS['success'], edgecolor=COLORS['success'], alpha=0.3, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(3.0, 1.5, "Container (writable layer)", ha='center', va='center', fontsize=7, fontweight='bold')

    save_figure(fig, "fig_docker_layers.pdf")


def plot_health_check_cycle():
    """Figure 42: Health check cycle."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0.5, 9.0)

    ax.text(4.0, 8.5, "Health Check Cycle (interval = 30s)", ha='center', fontsize=11, fontweight='bold')

    # Send Probe (top centre)
    add_box(ax, 2.5, 7.0, 2.0, 0.7, "Send Probe",
            facecolor=COLORS['primary'], edgecolor=COLORS['primary'], fontsize=8, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(3.5, 6.3), xytext=(3.5, 7.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Evaluate Response
    add_box(ax, 2.5, 5.3, 2.0, 0.7, "Evaluate\nResponse",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=8, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(3.5, 4.6), xytext=(3.5, 5.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.2))

    # Healthy? (decision)
    add_box(ax, 2.5, 3.6, 2.0, 0.7, "Healthy?",
            facecolor=COLORS['secondary'], edgecolor=COLORS['secondary'], fontsize=8, fontweight='bold')

    # Yes → left: Wait interval
    ax.annotate('', xy=(1.5, 3.95), xytext=(2.5, 3.95),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.2))
    ax.text(2.0, 4.15, "Yes", ha='center', fontsize=7, color=COLORS['success'], fontweight='bold')

    add_box(ax, 0.2, 3.6, 1.3, 0.7, "Wait\n30s",
            facecolor=COLORS['success'], edgecolor=COLORS['success'], fontsize=7, fontweight='bold')

    # Loop back up from Wait to Send Probe (left side)
    ax.annotate('', xy=(0.85, 7.0), xytext=(0.85, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0, ls='--'))
    ax.plot([0.85, 2.5], [7.35, 7.35], color=COLORS['gray_medium'], lw=1.0, ls='--')

    # No → right: Restart
    ax.annotate('', xy=(5.5, 3.95), xytext=(4.5, 3.95),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.2))
    ax.text(5.0, 4.15, "No", ha='center', fontsize=7, color=COLORS['danger'], fontweight='bold')

    add_box(ax, 5.5, 3.6, 1.5, 0.7, "Restart",
            facecolor=COLORS['danger'], edgecolor=COLORS['danger'], fontsize=7, fontweight='bold')

    # Loop back up from Restart to Send Probe (right side)
    ax.annotate('', xy=(6.25, 7.0), xytext=(6.25, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_medium'], lw=1.0, ls='--'))
    ax.plot([6.25, 4.5], [7.35, 7.35], color=COLORS['gray_medium'], lw=1.0, ls='--')

    save_figure(fig, "fig_health_check_cycle.pdf")


def plot_correlation_scatter():
    """Figure 43: Correlation scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    
    np.random.seed(42)
    correlations = [0.9, 0, -0.9]
    titles = ['r = +0.9', 'r = 0', 'r = -0.9']
    
    for ax, rho, title in zip(axes, correlations, titles):
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 200).T
        
        ax.scatter(x, y, c=COLORS['primary'], alpha=0.5, s=15)
        
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(x), p(sorted(x)), color=COLORS['danger'], linewidth=1.5)
        
        ax.text(0.05, 0.9, title, transform=ax.transAxes, fontsize=9, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        style_axes(ax)
    
    save_figure(fig, "fig_correlation_scatter.pdf")


def plot_box_plot():
    """Figure 44: Annotated box plot."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    
    np.random.seed(42)
    data = np.random.lognormal(mean=2.5, sigma=0.6, size=200)
    
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][0].set_alpha(0.5)
    bp['medians'][0].set_color(COLORS['danger'])
    bp['medians'][0].set_linewidth(2)
    
    for flier in bp['fliers']:
        flier.set(marker='o', color=COLORS['danger'], alpha=0.5)
    
    stats = np.percentile(data, [25, 50, 75])
    q1, median, q3 = stats
    
    iqr = q3 - q1
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    
    ax.annotate('Q1', xy=(0.85, q1), xytext=(0.7, q1), fontsize=7, color=COLORS['gray_medium'])
    ax.annotate('Q2', xy=(0.85, median), xytext=(0.7, median), fontsize=7, color=COLORS['danger'], fontweight='bold')
    ax.annotate('Q3', xy=(0.85, q3), xytext=(0.7, q3), fontsize=7, color=COLORS['gray_medium'])
    
    ax.annotate('', xy=(0.82, q1), xytext=(0.82, q3), arrowprops=dict(arrowstyle='<->', color=COLORS['gray_medium'], lw=0.8))
    ax.text(0.65, (q1+q3)/2, f'IQR', fontsize=7, rotation=90, va='center')
    
    ax.set_ylabel('Value', fontsize=9)
    ax.set_title('Box Plot with Annotations', fontsize=10)
    ax.set_xticks([])
    style_axes(ax)
    
    save_figure(fig, "fig_box_plot.pdf")


FIGURE_REGISTRY = [
    ("fig_supervised_learning_pipeline", plot_supervised_learning_pipeline, 6.5, 3.0),
    ("fig_multiclass_decision_boundary", plot_multiclass_decision_boundary, 4.0, 3.5),
    ("fig_decision_tree_example", plot_decision_tree_example, 5.0, 4.0),
    ("fig_random_forest_architecture", plot_random_forest_architecture, 6.5, 3.5),
    ("fig_gradient_boosting_sequential", plot_gradient_boosting_sequential, 6.5, 3.0),
    ("fig_cascade_classifier", plot_cascade_classifier, 6.5, 3.0),
    ("fig_kfold_cross_validation", plot_kfold_cross_validation, 5.0, 3.5),
    ("fig_data_leakage_scenarios", plot_data_leakage_scenarios, 6.5, 3.0),
    ("fig_tfidf_computation", plot_tfidf_computation, 6.5, 2.5),
    ("fig_one_hot_encoding", plot_one_hot_encoding, 5.0, 3.0),
    ("fig_feature_scaling_comparison", plot_feature_scaling_comparison, 6.5, 2.5),
    ("fig_feature_binning", plot_feature_binning, 5.0, 3.0),
    ("fig_interaction_features", plot_interaction_features, 6.5, 3.0),
    ("fig_csr_format", plot_csr_format, 6.5, 3.5),
    ("fig_fuzzy_matching", plot_fuzzy_matching, 5.0, 3.0),
    ("fig_rule_engine_flow", plot_rule_engine_flow, 5.0, 4.0),
    ("fig_confidence_thresholds", plot_confidence_thresholds, 6.5, 2.0),
    ("fig_weighted_blending", plot_weighted_blending, 5.0, 3.5),
    ("fig_geometric_vs_arithmetic_mean", plot_geometric_vs_arithmetic_mean, 5.0, 3.0),
    ("fig_transformer_architecture", plot_transformer_architecture, 5.0, 5.0),
    ("fig_prompt_structure", plot_prompt_structure, 6.5, 3.5),
    ("fig_temperature_effect", plot_temperature_effect, 6.5, 3.0),
    ("fig_api_integration_flow", plot_api_integration_flow, 6.5, 2.5),
    ("fig_exponential_backoff", plot_exponential_backoff, 6.5, 2.5),
    ("fig_fallback_chain", plot_fallback_chain, 5.0, 4.0),
    ("fig_precision_recall_f1", plot_precision_recall_f1, 5.0, 3.0),
    ("fig_confusion_matrix", plot_confusion_matrix, 4.0, 3.5),
    ("fig_feature_importance", plot_feature_importance, 5.0, 3.5),
    ("fig_calibration_curve", plot_calibration_curve, 4.0, 3.5),
    ("fig_distributions", plot_distributions, 5.0, 3.0),
    ("fig_weighted_sampling", plot_weighted_sampling, 6.5, 3.0),
    ("fig_mixture_distribution", plot_mixture_distribution, 5.0, 3.0),
    ("fig_correlated_features", plot_correlated_features, 6.5, 2.5),
    ("fig_pipeline_architecture", plot_pipeline_architecture, 6.5, 2.5),
    ("fig_hybrid_engine", plot_hybrid_engine, 6.5, 3.5),
    ("fig_model_serialization", plot_model_serialization, 6.5, 2.5),
    ("fig_rest_api", plot_rest_api, 6.5, 3.0),
    ("fig_file_upload_flow", plot_file_upload_flow, 6.5, 3.0),
    ("fig_ocr_pipeline", plot_ocr_pipeline, 6.5, 2.5),
    ("fig_logging_architecture", plot_logging_architecture, 5.0, 4.0),
    ("fig_docker_layers", plot_docker_layers, 5.0, 4.0),
    ("fig_health_check_cycle", plot_health_check_cycle, 4.0, 4.0),
    ("fig_correlation_scatter", plot_correlation_scatter, 6.5, 2.5),
    ("fig_box_plot", plot_box_plot, 5.0, 3.0),
]


def main():
    """Generate all 44 figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()
    np.random.seed(42)
    
    print(f"Generating {len(FIGURE_REGISTRY)} figures...")
    
    results = []
    for i, (name, func, w, h) in enumerate(FIGURE_REGISTRY):
        try:
            func()
            results.append((name, "OK"))
            print(f"  [{len(results)}/{len(FIGURE_REGISTRY)}] {name} OK")
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            print(f"  [{len(results)}/{len(FIGURE_REGISTRY)}] {name} FAIL: {e}")
    
    ok = sum(1 for _, s in results if s == "OK")
    print(f"\nGenerated {ok}/{len(FIGURE_REGISTRY)} figures successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()