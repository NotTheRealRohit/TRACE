# Implementation Plan: Generate 44 Figures for Foundation_TRACE.tex

## 1. Architecture Overview

### Script Location
- **File**: `/mnt/d/study/git/capProj-2/Documentation/generate_figures.py`
- **Output Directory**: `/mnt/d/study/git/capProj-2/Documentation/figures/`

### Approach
- Single monolithic Python script with one function per figure
- All figures saved as vector PDF files (no rasterization)
- Orchestration via a `main()` function that calls each figure generator
- Progress reporting with figure count and status

### Output File Naming Convention
Each figure is saved as `{label}.pdf` matching the LaTeX `\label{}`:
```
figures/fig_supervised_learning_pipeline.pdf
figures/fig_multiclass_decision_boundary.pdf
...
figures/fig_box_plot.pdf
```

### LaTeX Integration
After generation, replace each `\fbox{...}` placeholder with:
```latex
\includegraphics[width=\textwidth]{figures/fig_name.pdf}
```

---

## 2. Global Style Configuration

### Color Palette (Color-Blind Friendly, Academic)
```python
COLORS = {
    "primary":      "#1f77b4",  # Blue
    "secondary":    "#ff7f0e",  # Orange
    "success":      "#2ca02c",  # Green
    "danger":       "#d62728",  # Red
    "purple":       "#9467bd",  # Purple
    "brown":        "#8c564b",  # Brown
    "pink":         "#e377c2",  # Pink
    "gray_light":   "#bcbd22",  # Yellow-green
    "gray_medium":  "#7f7f7f",  # Gray
    "gray_dark":    "#1a1a2e",  # Near-black
    "bg_light":     "#f8f9fa",  # Light gray background
    "bg_box":       "#e8eaf6",  # Light indigo for boxes
    "border":       "#37474f",  # Dark slate for borders
    "text":         "#212121",  # Near-black text
    "grid":         "#e0e0e0",  # Light grid lines
    "white":        "#ffffff",
    "black":        "#000000",
}
```

### Matplotlib rcParams
```python
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
```

### Figure Dimensions
| Category | Width (in) | Height (in) | Use Case |
|----------|------------|-------------|----------|
| Single-column plot | 3.5 | 2.5 | Simple statistical plots |
| Wide plot | 6.5 | 3.5 | Multi-panel comparisons |
| Architecture diagram | 6.5 | 3.0 | Flow diagrams, pipelines |
| Tall diagram | 5.0 | 4.0 | Decision trees, hierarchies |
| Square plot | 4.0 | 4.0 | Scatter plots, confusion matrix |

---

## 3. Per-Figure Specifications

### Section 1: Machine Learning Algorithms

---

#### Figure 1: `fig:supervised-learning-pipeline`
- **Function**: `plot_supervised_learning_pipeline()`
- **Type**: Block flow diagram (two-phase)
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - Two large rounded rectangles: "TRAINING PHASE" (left) and "INFERENCE PHASE" (right)
  - Training: "Labeled Data" → "Model Fitting" → "Trained Model (θ)"
  - Inference: "New Input (x)" → "Trained Model (θ)" → "Prediction (ŷ)"
  - Dashed arrow from "Trained Model" in training to "Trained Model" in inference
- **Colors**: Training phase boxes in `bg_box` with `primary` border; inference in `bg_box` with `success` border
- **Implementation**: `patches.FancyBboxPatch` for boxes, `ax.annotate()` for arrows, `ax.text()` for labels
- **Key Labels**: "Training Phase", "Inference Phase", "Labeled Data D={(xᵢ,yᵢ)}", "Model Fitting", "Trained Model f̂", "New Input x", "Prediction ŷ"

---

#### Figure 2: `fig:multiclass-decision-boundary`
- **Function**: `plot_multiclass_decision_boundary()`
- **Type**: 2D feature space with decision regions
- **Dimensions**: 4.0 x 3.5 inches
- **Data**: Generate 3-class data using `make_classification(n_classes=3, n_features=2, n_informative=2, n_redundant=0)`
- **Visual Elements**:
  - Colored decision regions (contour fill) for 3 classes
  - Scatter points for training data
  - Decision boundary lines (contour at 0.5 probability)
- **Colors**: Class 0 = `primary`, Class 1 = `secondary`, Class 2 = `success`
- **Implementation**: Train a small `RandomForestClassifier`, use `meshgrid` + `contourf` for regions, `scatter` for points
- **Key Labels**: "Feature X₁", "Feature X₂", legend with "Class A", "Class B", "Class C"

---

#### Figure 3: `fig:decision-tree-example`
- **Function**: `plot_decision_tree_example()`
- **Type**: Tree diagram with nodes and edges
- **Dimensions**: 5.0 x 4.0 inches
- **Visual Elements**:
  - Root node: "Voltage > 14.5?"
  - Left child: "DTC starts with P?" → leaves "Production Failure" / "Customer Failure"
  - Right child: "Notes contain 'moisture'?" → leaves "Rejected" / "Needs Review"
  - Rectangular test nodes, elliptical leaf nodes
  - Edges labeled "Yes"/"No"
- **Colors**: Test nodes = `bg_box` with `primary` border; leaf nodes = `success` (approve), `danger` (reject), `gray_medium` (review)
- **Implementation**: Manual positioning with `patches.FancyBboxPatch` and `patches.Ellipse`, `ax.plot()` for edges, `ax.text()` for labels
- **Key Labels**: Feature tests at internal nodes, class labels at leaves, "Yes"/"No" on edges

---

#### Figure 4: `fig:random-forest-architecture`
- **Function**: `plot_random_forest_architecture()`
- **Type**: Architecture diagram (parallel processing)
- **Dimensions**: 6.5 x 3.5 inches
- **Visual Elements**:
  - Left: "Training Data" box
  - Middle-top: "Bootstrap Sample 1" → "Tree 1", "Bootstrap Sample 2" → "Tree 2", "Bootstrap Sample 3" → "Tree 3" (3 parallel paths)
  - Right: "Vote Aggregation" → "Final Prediction"
  - Arrows showing data flow
- **Colors**: Data = `primary`, Trees = `success`, Aggregator = `purple`
- **Implementation**: `patches.FancyBboxPatch` for boxes, `ax.annotate()` with `arrowprops` for connections
- **Key Labels**: "Bootstrap Sampling", "Parallel Tree Training", "Majority Vote", "n_estimators = T"

---

#### Figure 5: `fig:gradient-boosting-sequential`
- **Function**: `plot_gradient_boosting_sequential()`
- **Type**: Sequential flow diagram with residual correction
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - Row 1: "Initial Prediction F₀(x) = mean(y)" → "Residuals r₁ = y - F₀(x)"
  - Row 2: "Tree h₁ fitted to r₁" → "Update: F₁(x) = F₀(x) + η·h₁(x)" → "Residuals r₂"
  - Row 3: "Tree h₂ fitted to r₂" → "Update: F₂(x) = F₁(x) + η·h₂(x)" → "..."
  - Vertical arrows showing sequential dependency
- **Colors**: Predictions = `primary`, Residuals = `danger`, Trees = `success`, Updates = `purple`
- **Implementation**: `patches.FancyBboxPatch` for boxes, vertical and horizontal arrows, equation labels
- **Key Labels**: "F₀(x)", "r₁", "h₁(x)", "F₁(x) = F₀(x) + η·h₁(x)", "Learning rate η"

---

#### Figure 6: `fig:cascade-classifier`
- **Function**: `plot_cascade_classifier()`
- **Type**: Two-stage pipeline diagram
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - Stage 1: "Input Features x" → "Classifier f₁ (Failure Analysis)" → "Probability Vector p⁽¹⁾"
  - Concatenation: "x" + "p⁽¹⁾" → "Augmented Features x' = [x | p⁽¹⁾]"
  - Stage 2: "x'" → "Classifier f₂ (Warranty Decision)" → "Final Prediction ŷ"
  - Dashed box around concatenation step
- **Colors**: Stage 1 = `primary`, Stage 2 = `success`, Concatenation = `purple`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()` for arrows, `patches.Rectangle` for dashed grouping box
- **Key Labels**: "Stage 1: Failure Analysis", "Stage 2: Warranty Decision", "x' = [x | p⁽¹⁾]", "softmax(f₁(x))"

---

#### Figure 7: `fig:kfold-cross-validation`
- **Function**: `plot_kfold_cross_validation()`
- **Type**: K-fold partitioning diagram
- **Dimensions**: 5.0 x 3.5 inches
- **Visual Elements**:
  - K=5 rows, each showing 5 blocks
  - In each row, 4 blocks shaded (train) and 1 block white (validation)
  - Validation block shifts right each row
  - Labels: "Iteration 1" through "Iteration 5"
  - Top bar showing "Full Dataset"
- **Colors**: Train = `primary` (60% opacity), Validation = `white` with `danger` border
- **Implementation**: `patches.Rectangle` for blocks, loop over folds, `ax.text()` for labels
- **Key Labels**: "Training (80%)", "Validation (20%)", "K=5", "Iteration k"

---

#### Figure 8: `fig:data-leakage-scenarios`
- **Function**: `plot_data_leakage_scenarios()`
- **Type**: Side-by-side comparison (correct vs incorrect)
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - Left panel (✓ CORRECT): "Full Data" → "Split" → "Train" → "Fit Scaler" → "Transform Test"
  - Right panel (✗ INCORRECT): "Full Data" → "Fit Scaler" → "Split" → "Train" / "Test"
  - Green checkmark on left, red X on right
  - Dashed divider between panels
- **Colors**: Correct path = `success`, Incorrect path = `danger`, Data boxes = `bg_box`
- **Implementation**: Two subplots (`ax1`, `ax2`), `patches.FancyBboxPatch`, `ax.annotate()`, `ax.axvline()` for divider
- **Key Labels**: "CORRECT: Fit on Train Only", "INCORRECT: Fit on Full Data", "Data Leakage!"

---

### Section 2: Feature Engineering

---

#### Figure 9: `fig:tfidf-computation`
- **Function**: `plot_tfidf_computation()`
- **Type**: Pipeline flow diagram
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - 5 sequential boxes: "Raw Text" → "Tokenization" → "TF Computation" → "IDF Weighting" → "Sparse Vector"
  - Example text shown in first box: "engine overheating voltage low"
  - Example output in last box: "[0.42, 0.31, 0.0, 0.28, ...]"
  - Arrows with operation labels
- **Colors**: Boxes = `bg_box` with `primary` border, arrows = `gray_medium`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()`, monospace font for examples
- **Key Labels**: "TF(t,d) = count/total", "IDF(t) = log(N/df) + 1", "TF-IDF = TF × IDF"

---

#### Figure 10: `fig:one-hot-encoding`
- **Function**: `plot_one_hot_encoding()`
- **Type**: Visual mapping diagram
- **Dimensions**: 5.0 x 3.0 inches
- **Visual Elements**:
  - Left column: categorical values "red", "green", "blue"
  - Right column: binary vectors [1,0,0], [0,1,0], [0,0,1] shown as colored grids
  - Arrows mapping each value to its vector
  - Highlight the "hot" (1) position in each vector
- **Colors**: red=`danger`, green=`success`, blue=`primary`; hot cells highlighted with darker shade
- **Implementation**: `patches.Rectangle` for vector cells, `ax.annotate()` for mappings
- **Key Labels**: "Categorical", "One-Hot Encoded", "K=3", "exactly one 1"

---

#### Figure 11: `fig:feature-scaling-comparison`
- **Function**: `plot_feature_scaling_comparison()`
- **Type**: Before/after histograms
- **Dimensions**: 6.5 x 2.5 inches
- **Data**: Generate skewed data: `np.random.lognormal(mean=2, sigma=0.8, size=2000)`
- **Visual Elements**:
  - Left subplot: histogram of raw data (right-skewed)
  - Right subplot: histogram after z-score standardization (centered at 0, σ=1)
  - Vertical lines showing mean and ±1σ
- **Colors**: Histogram bars = `primary` (50% opacity), mean line = `danger`, σ lines = `gray_medium` (dashed)
- **Implementation**: `ax.hist()`, `ax.axvline()`, two subplots
- **Key Labels**: "Raw Distribution (skewed)", "Standardized (μ=0, σ=1)", "z = (x - μ) / σ"

---

#### Figure 12: `fig:feature-binning`
- **Function**: `plot_feature_binning()`
- **Type**: Distribution with bin partitioning
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: `np.random.normal(loc=14, scale=2, size=3000)`
- **Visual Elements**:
  - Histogram of continuous distribution
  - Vertical lines at bin boundaries (e.g., 10, 12, 14, 16, 18)
  - Colored regions for each bin
  - Bin labels below x-axis: "Bin 0", "Bin 1", etc.
- **Colors**: Alternating bin colors = `primary`/`secondary` (20% opacity), boundary lines = `danger`
- **Implementation**: `ax.hist()`, `ax.axvline()`, `ax.fill_between()` for bin shading
- **Key Labels**: "Voltage Distribution", "Bin Boundaries", "K=5 equal-width bins"

---

#### Figure 13: `fig:interaction-features`
- **Function**: `plot_interaction_features()`
- **Type**: Before/after decision boundary comparison
- **Dimensions**: 6.5 x 3.0 inches
- **Data**: XOR-like pattern with interaction
- **Visual Elements**:
  - Left: 2D scatter with linear decision boundary (poor fit)
  - Right: Same data with non-linear boundary after adding x₁×x₂ feature (good fit)
  - Contour fill for decision regions
- **Colors**: Class 0 = `primary`, Class 1 = `danger`, boundary = `black`
- **Implementation**: Two subplots, `contourf` for regions, `scatter` for points
- **Key Labels**: "Linear Boundary (no interaction)", "Non-linear Boundary (with x₁×x₂)", "Poor Fit" vs "Good Fit"

---

#### Figure 14: `fig:csr-format`
- **Function**: `plot_csr_format()`
- **Type**: Matrix decomposition diagram
- **Dimensions**: 6.5 x 3.5 inches
- **Visual Elements**:
  - Top-left: 4×5 dense matrix with some zeros highlighted
  - Below: Three arrays shown as horizontal grids:
    - `data`: [3, 1, 4, 2, 5] (non-zero values)
    - `indices`: [1, 3, 0, 2, 4] (column indices)
    - `indptr`: [0, 2, 2, 4, 5] (row pointers)
  - Arrows from matrix non-zeros to corresponding array positions
- **Colors**: Non-zero cells = `primary` (highlighted), zero cells = `bg_light`, array cells = `bg_box`
- **Implementation**: `patches.Rectangle` for matrix cells and array cells, `ax.annotate()` for arrows
- **Key Labels**: "Dense Matrix (4×5)", "data: non-zero values", "indices: column indices", "indptr: row pointers"

---

#### Figure 15: `fig:fuzzy-matching`
- **Function**: `plot_fuzzy_matching()`
- **Type**: Edit distance visualization
- **Dimensions**: 5.0 x 3.0 inches
- **Visual Elements**:
  - Three string pairs with edit operations shown:
    - "engine" → "engne" (transposition: distance=1)
    - "voltage" → "votage" (deletion: distance=1)
    - "moisture" → "moister" (substitution: distance=1)
  - Character-by-character alignment with color coding:
    - Green = match, Red = mismatch, Orange = insertion/deletion
  - Similarity ratio displayed for each pair
- **Colors**: Match = `success`, Mismatch = `danger`, Insert/Delete = `secondary`
- **Implementation**: `patches.Rectangle` for character cells, color-coded, `ax.text()` for labels
- **Key Labels**: "Edit Distance", "Similarity = 1 - lev/max(|a|,|b|)", "Levenshtein Distance"

---

### Section 3: Rule-Based Systems

---

#### Figure 16: `fig:rule-engine-flow`
- **Function**: `plot_rule_engine_flow()`
- **Type**: Flowchart
- **Dimensions**: 5.0 x 4.0 inches
- **Visual Elements**:
  - Diamond: "Input (fault_code, notes, voltage)"
  - Sequential diamonds: "Rule 1: Voltage > 16V?", "Rule 2: Voltage < 11V?", "Rule 3: 'moisture' in notes?", "Rule 4: DTC prefix = P?"
  - Each rule diamond has "Yes" → rectangular output box, "No" → next rule
  - Bottom: "No rule matched → Default/ML fallback"
- **Colors**: Decision diamonds = `bg_box` with `primary` border, Output boxes = `success`/`danger`/`gray_medium`
- **Implementation**: `patches.Polygon` for diamonds, `patches.FancyBboxPatch` for outputs, `ax.annotate()` for flow arrows
- **Key Labels**: "First Match Wins", "Priority Order", "Approved", "Rejected", "Manual Review"

---

#### Figure 17: `fig:confidence-thresholds`
- **Function**: `plot_confidence_thresholds()`
- **Type**: Number line with threshold regions
- **Dimensions**: 6.5 x 2.0 inches
- **Visual Elements**:
  - Horizontal number line from 0 to 100
  - Three colored regions:
    - 0-65: Red zone labeled "Manual Review"
    - 65-85: Orange zone labeled "Cautious"
    - 85-100: Green zone labeled "Firm"
  - Vertical dashed lines at 65 and 85
  - Example confidence markers on the line
- **Colors**: Manual Review = `danger` (30% opacity), Cautious = `secondary` (30%), Firm = `success` (30%)
- **Implementation**: `ax.fill_between()` for regions, `ax.axvline()` for thresholds, `ax.text()` for labels
- **Key Labels**: "τ_manual = 65%", "τ_firm = 85%", "Confidence Score (%)"

---

### Section 4: Score Combination & Decision Fusion

---

#### Figure 18: `fig:weighted-blending`
- **Function**: `plot_weighted_blending()`
- **Type**: Stacked bar chart
- **Dimensions**: 5.0 x 3.5 inches
- **Visual Elements**:
  - Two stacked bars side by side:
    - Bar 1 (Agree): Rule (30%) + ML (50%) + LLM (20%) + Bonus (5%)
    - Bar 2 (Disagree): Rule (20%) + ML (30%) + LLM (15%) (no bonus)
  - Legend showing source contributions
  - Total confidence labeled on top of each bar
- **Colors**: Rule = `primary`, ML = `success`, LLM = `purple`, Bonus = `secondary`
- **Implementation**: `ax.bar()` with `bottom` parameter for stacking, `ax.text()` for labels
- **Key Labels**: "Sources Agree (85%)", "Sources Disagree (65%)", "Rule", "ML", "LLM", "Agreement Bonus"

---

#### Figure 19: `fig:geometric-vs-arithmetic-mean`
- **Function**: `plot_geometric_vs_arithmetic_mean()`
- **Type**: Line plot comparison
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: x = np.linspace(0.1, 0.9, 100); arithmetic = (x + 0.5)/2; geometric = sqrt(x * 0.5)
- **Visual Elements**:
  - Two lines: arithmetic mean (straight) and geometric mean (curved below)
  - Shaded region between the two lines (AM-GM gap)
  - Vertical line showing maximum gap at x ≠ 0.5
- **Colors**: Arithmetic = `primary`, Geometric = `danger`, Gap = `secondary` (20% opacity)
- **Implementation**: `ax.plot()`, `ax.fill_between()`, `ax.axvline()`
- **Key Labels**: "Arithmetic Mean", "Geometric Mean", "AM-GM Gap", "G(a,b) = √(a·b)", "A(a,b) = (a+b)/2"

---

### Section 5: Large Language Models

---

#### Figure 20: `fig:transformer-architecture`
- **Function**: `plot_transformer_architecture()`
- **Type**: Neural network block diagram
- **Dimensions**: 5.0 x 5.0 inches
- **Visual Elements**:
  - Input at bottom: "Input Embeddings + Positional Encoding"
  - Stacked blocks (repeat N times):
    - "Multi-Head Attention" → "Add & Norm" → "Feed Forward" → "Add & Norm"
  - Output at top: "Output Representations"
  - Residual connection arrows bypassing each sub-layer
  - "N×" label on the side indicating repetition
- **Colors**: Attention = `primary`, Add&Norm = `purple`, FFN = `success`, Residual = `gray_medium` (dashed)
- **Implementation**: `patches.FancyBboxPatch` for blocks, `ax.annotate()` for data flow and residual arrows
- **Key Labels**: "Multi-Head Attention", "Add & Norm", "Feed Forward Network", "Residual Connection", "N×"

---

#### Figure 21: `fig:prompt-structure`
- **Function**: `plot_prompt_structure()`
- **Type**: Structured prompt anatomy diagram
- **Dimensions**: 6.5 x 3.5 inches
- **Visual Elements**:
  - Large rectangle representing full prompt, divided into labeled sections:
    - Top: "Role Framing" — "You are an expert automotive warranty analyst"
    - Middle-top: "Input Data" — fault_code, notes, voltage
    - Middle: "Constraints" — enumerated categories
    - Middle-bottom: "Disambiguation Rules" — priority rules
    - Bottom: "Output Format" — JSON schema
  - Color-coded sections with labels on the right
- **Colors**: Role = `primary`, Input = `success`, Constraints = `secondary`, Rules = `purple`, Format = `brown`
- **Implementation**: `patches.FancyBboxPatch` for sections, `ax.text()` for content (small font)
- **Key Labels**: "Role Framing", "Input Data", "Enumerated Constraints", "Disambiguation Rules", "JSON Output Format"

---

#### Figure 22: `fig:temperature-effect`
- **Function**: `plot_temperature_effect()`
- **Type**: Probability distribution comparison
- **Dimensions**: 6.5 x 3.0 inches
- **Data**: Fixed logits z = [3.0, 1.0, 0.5, 0.2]; compute softmax at T = 0.1, 0.5, 1.0, 2.0
- **Visual Elements**:
  - Four subplots (1×4), each showing a bar chart of token probabilities
  - T=0.1: nearly deterministic (one bar at ~1.0)
  - T=0.5: peaked distribution
  - T=1.0: moderate spread
  - T=2.0: nearly uniform
- **Colors**: Bars gradient from `primary` (highest prob) to lighter shades
- **Implementation**: Four subplots, `ax.bar()`, `ax.set_title()` for temperature labels
- **Key Labels**: "T=0.1 (Greedy)", "T=0.5", "T=1.0", "T=2.0 (Random)", "P(token)", "Token"

---

#### Figure 23: `fig:api-integration-flow`
- **Function**: `plot_api_integration_flow()`
- **Type**: Client-server flow diagram
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - Left: "Client Application" box
  - Middle: "HTTP POST /v1/chat/completions" with JSON body shown
  - Right: "API Server (OpenRouter/OpenAI)" → "LLM Inference"
  - Return arrow: "JSON Response" → "Parsed Output"
  - Authentication header shown
- **Colors**: Client = `primary`, HTTP = `gray_medium`, Server = `success`, Response = `purple`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()` for arrows, monospace for JSON
- **Key Labels**: "POST /v1/chat/completions", "Authorization: Bearer KEY", "model, messages, temperature", "choices[0].message.content"

---

#### Figure 24: `fig:exponential-backoff`
- **Function**: `plot_exponential_backoff()`
- **Type**: Timeline diagram
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - Horizontal timeline from t=0 to t=7 seconds
  - Attempt markers: "Attempt 0" (t=0), "Attempt 1" (t=1), "Attempt 2" (t=3), "Attempt 3" (t=7)
  - Delay annotations between attempts: "1s", "2s", "4s"
  - X marks for failed attempts, checkmark for success
  - Formula annotation: "delay_k = base_delay × 2^k"
- **Colors**: Failed attempts = `danger`, Success = `success`, Delays = `secondary`
- **Implementation**: Horizontal line for timeline, `ax.plot()` markers, `ax.annotate()` for delays
- **Key Labels**: "Attempt 0 (fail)", "Attempt 1 (fail)", "Attempt 2 (fail)", "Attempt 3 (success)", "Total: 7s"

---

#### Figure 25: `fig:fallback-chain`
- **Function**: `plot_fallback_chain()`
- **Type**: Decision tree / flow diagram
- **Dimensions**: 5.0 x 4.0 inches
- **Visual Elements**:
  - Top: "Primary: LLM API Call"
  - If fails → "Fallback 1: Rule Engine"
  - If fails → "Fallback 2: ML Model"
  - If fails → "Default: Safe Defaults"
  - Each level shows capability decreasing but availability increasing
  - Side annotations: "Most capable" at top, "Most reliable" at bottom
- **Colors**: Primary = `primary`, Fallback 1 = `secondary`, Fallback 2 = `success`, Default = `gray_medium`
- **Implementation**: Vertical flow with `patches.FancyBboxPatch`, `ax.annotate()` for decision arrows
- **Key Labels**: "Try LLM", "Try Rules", "Try ML", "Use Defaults", "Capability ↓", "Reliability ↑"

---

### Section 6: Model Evaluation Metrics

---

#### Figure 26: `fig:precision-recall-f1`
- **Function**: `plot_precision_recall_f1()`
- **Type**: Bar chart with relationship diagram
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: Example values: Precision=0.88, Recall=0.82, F1=0.85
- **Visual Elements**:
  - Three bars side by side: Precision, Recall, F1
  - Horizontal dashed line at F1 showing harmonic mean relationship
  - Formula annotation: "F1 = 2·P·R / (P+R)"
  - Venn-style overlap diagram below showing TP, FP, FN regions
- **Colors**: Precision = `primary`, Recall = `success`, F1 = `purple`
- **Implementation**: `ax.bar()`, `ax.axhline()`, `patches.Circle` or `patches.Ellipse` for Venn
- **Key Labels**: "Precision = TP/(TP+FP)", "Recall = TP/(TP+FN)", "F1 = 2·P·R/(P+R)"

---

#### Figure 27: `fig:confusion-matrix`
- **Function**: `plot_confusion_matrix()`
- **Type**: Heatmap
- **Dimensions**: 4.0 x 3.5 inches
- **Data**: 4×4 confusion matrix with realistic values:
  ```
  [[142,  5,  3,  2],
   [  8, 89, 12,  5],
   [  4, 10, 95,  6],
   [  3,  7,  8, 82]]
  ```
- **Visual Elements**:
  - Color-coded heatmap with values in each cell
  - Row labels: "Production Failure", "Customer Failure", "NTF", "Electrical"
  - Column labels: same as rows
  - Diagonal cells highlighted
- **Colors**: Use sequential colormap (Blues or similar), diagonal = darkest
- **Implementation**: `seaborn.heatmap()` or `ax.imshow()` with `ax.text()` for values
- **Key Labels**: "True Labels (rows)", "Predicted Labels (columns)", cell values

---

#### Figure 28: `fig:feature-importance`
- **Function**: `plot_feature_importance()`
- **Type**: Horizontal bar chart
- **Dimensions**: 5.0 x 3.5 inches
- **Data**: 8 features with importance scores (descending):
  - dtc_prefix_P: 0.28, voltage: 0.19, notes_length: 0.14, tfidf_engine: 0.11, dtc_count: 0.09, ohe_supplier: 0.07, voltage_bin: 0.06, notes_moisture: 0.06
- **Visual Elements**:
  - Horizontal bars sorted by importance
  - Values labeled at end of each bar
  - Cumulative importance line (secondary axis)
- **Colors**: Bars = `primary` gradient (darker = more important)
- **Implementation**: `ax.barh()`, `ax.text()` for values, `ax.twinx()` for cumulative line
- **Key Labels**: "Feature", "Gini Importance", "Cumulative %"

---

#### Figure 29: `fig:calibration-curve`
- **Function**: `plot_calibration_curve()`
- **Type**: Calibration/reliability plot
- **Dimensions**: 4.0 x 3.5 inches
- **Data**: Generate binned calibration data:
  - Perfect: diagonal y=x
  - Model curve: slightly overconfident (above diagonal at low probs, below at high)
- **Visual Elements**:
  - Diagonal dashed line (perfect calibration)
  - Model calibration curve (step function or smooth)
  - Histogram of predicted probabilities at bottom (inset or secondary plot)
  - ECE value annotation
- **Colors**: Perfect = `gray_medium` (dashed), Model = `primary`, Histogram = `secondary`
- **Implementation**: `ax.plot()` for curves, `ax.axhline()`/`ax.axvline()` for reference, inset with `fig.add_axes()`
- **Key Labels**: "Mean Predicted Probability", "Fraction of Positives", "Perfect Calibration", "ECE = 0.042"

---

### Section 7: Synthetic Data Generation

---

#### Figure 30: `fig:distributions`
- **Function**: `plot_distributions()`
- **Type**: Overlaid PDF curves
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: x = np.linspace(-4, 8, 500); normal = scipy.stats.norm.pdf(x); truncated = scipy.stats.truncnorm.pdf(x); lognormal = scipy.stats.lognorm.pdf(x)
- **Visual Elements**:
  - Three overlaid PDF curves on same axes
  - Normal: symmetric bell centered at 0
  - Truncated normal: bell cut off at bounds [−2, 4]
  - Log-normal: right-skewed, positive only
  - Legend identifying each distribution
- **Colors**: Normal = `primary`, Truncated = `success`, Log-normal = `secondary`
- **Implementation**: `ax.plot()` for PDFs, `ax.fill_between()` for shading under curves
- **Key Labels**: "Normal N(0,1)", "Truncated Normal", "Log-Normal", "PDF", "x"

---

#### Figure 31: `fig:weighted-sampling`
- **Function**: `plot_weighted_sampling()`
- **Type**: Dual bar chart (weights vs results)
- **Dimensions**: 6.5 x 3.0 inches
- **Data**: Categories A-E with weights [0.35, 0.25, 0.20, 0.12, 0.08]; simulate 1000 samples
- **Visual Elements**:
  - Left subplot: target weights as bars
  - Right subplot: actual sample distribution from weighted random sampling
  - Dashed lines showing expected proportions
- **Colors**: Bars = `primary` (left), `success` (right)
- **Implementation**: Two subplots, `ax.bar()`, `np.random.choice()` with p=weights
- **Key Labels**: "Target Weights", "Actual Distribution (n=1000)", "Category", "Probability"

---

#### Figure 32: `fig:mixture-distribution`
- **Function**: `plot_mixture_distribution()`
- **Type**: Bimodal distribution with components
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: π₁=0.6, μ₁=10, σ₁=2; π₂=0.4, μ₂=18, σ₂=3; mixture = π₁·N₁ + π₂·N₂
- **Visual Elements**:
  - Component 1: N(10, 2²) curve
  - Component 2: N(18, 3²) curve
  - Mixture: weighted sum curve (bold)
  - Vertical lines at μ₁ and μ₂
  - Mixing weights annotated
- **Colors**: Component 1 = `primary`, Component 2 = `secondary`, Mixture = `danger` (bold)
- **Implementation**: `ax.plot()` for curves, `ax.axvline()` for means, `ax.fill_between()` for component areas
- **Key Labels**: "π₁=0.6, μ₁=10", "π₂=0.4, μ₂=18", "Mixture", "f(x) = π₁·N₁(x) + π₂·N₂(x)"

---

#### Figure 33: `fig:correlated-features`
- **Function**: `plot_correlated_features()`
- **Type**: Three scatter plots
- **Dimensions**: 6.5 x 2.5 inches
- **Data**: Generate bivariate normal with ρ=0.8, ρ=0, ρ=−0.8 (n=200 each)
- **Visual Elements**:
  - Three subplots side by side
  - Each: scatter plot with regression line
  - Correlation coefficient displayed in each subplot
  - Ellipse showing covariance region
- **Colors**: Points = `primary` (50% opacity), Regression line = `danger`
- **Implementation**: Three subplots, `ax.scatter()`, `ax.plot()` for regression line, `patches.Ellipse`
- **Key Labels**: "ρ = +0.8", "ρ = 0", "ρ = −0.8", "X₁", "X₂"

---

### Section 8: System Architecture Patterns

---

#### Figure 34: `fig:pipeline-architecture`
- **Function**: `plot_pipeline_architecture()`
- **Type**: Horizontal flow diagram
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - Sequential boxes: "Input" → "Preprocessing" → "Feature Extraction" → "Model Inference" → "Post-processing" → "Output"
  - Arrows between each stage
  - Stage numbers (1-5) above boxes
  - Data format annotations below each box
- **Colors**: Boxes = `bg_box` with `primary` border, arrows = `gray_medium`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()` for arrows
- **Key Labels**: "Raw Input", "Cleaned Data", "Feature Vector", "Predictions", "Formatted Output"

---

#### Figure 35: `fig:hybrid-engine`
- **Function**: `plot_hybrid_engine()`
- **Type**: Three-input combiner diagram
- **Dimensions**: 6.5 x 3.5 inches
- **Visual Elements**:
  - Three parallel boxes on left: "Rule Engine", "ML Classifier", "LLM Layer"
  - Arrows converging to center: "Weighted Combiner"
  - Output on right: "Final Decision"
  - Weight annotations on each arrow (w_rule, w_ml, w_llm)
  - Agreement/disagreement logic shown
- **Colors**: Rules = `primary`, ML = `success`, LLM = `purple`, Combiner = `secondary`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()` for converging arrows
- **Key Labels**: "f_rules(x)", "f_ML(x)", "f_LLM(x)", "Combine(·)", "Decision + Confidence"

---

#### Figure 36: `fig:model-serialization`
- **Function**: `plot_model_serialization()`
- **Type**: Serialization flow diagram
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - Left: "Python Objects" (model weights, transformers, metadata icons)
  - Arrow: "pickle.dump()" → "Binary File (.pkl)"
  - Arrow: "pickle.load()" → "Restored Objects"
  - File icon in center
  - Bidirectional arrows
- **Colors**: Objects = `primary`, File = `gray_dark`, Arrows = `success`/`danger`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()`, file icon with `patches.Rectangle`
- **Key Labels**: "Serialize", "Deserialize", "trace_models.pkl", "θ, φ, ψ"

---

### Section 9: API & Web Architecture

---

#### Figure 37: `fig:rest-api`
- **Function**: `plot_rest_api()`
- **Type**: Request/response cycle diagram
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - Left: "Client" (browser icon)
  - Right: "Server" (API icon)
  - Top arrow (left→right): "POST /analyze" with JSON body shown
  - Bottom arrow (right→left): "200 OK" with response JSON
  - Headers shown on request arrow
- **Colors**: Client = `primary`, Server = `success`, Request = `secondary`, Response = `purple`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()` for arrows, monospace for JSON
- **Key Labels**: "POST /analyze", "Content-Type: application/json", "200 OK", "ClaimResponse JSON"

---

#### Figure 38: `fig:file-upload-flow`
- **Function**: `plot_file_upload_flow()`
- **Type**: Multipart upload flow diagram
- **Dimensions**: 6.5 x 3.0 inches
- **Visual Elements**:
  - "Client Browser" → "multipart/form-data POST" → "Server Parser" → "File Processing" → "JSON Response"
  - Multipart boundary shown in request
  - File icon flowing through pipeline
- **Colors**: Client = `primary`, Server = `success`, Data = `secondary`
- **Implementation**: `patches.FancyBboxPatch`, `ax.annotate()`, file icon
- **Key Labels**: "multipart/form-data", "boundary=----WebKitFormBoundary", "file stream", "processed result"

---

### Section 10: OCR & Image Processing

---

#### Figure 39: `fig:ocr-pipeline`
- **Function**: `plot_ocr_pipeline()`
- **Type**: OCR processing flow
- **Dimensions**: 6.5 x 2.5 inches
- **Visual Elements**:
  - "Input Image" (rectangle with simulated text regions) → "Text Detection" (bounding boxes drawn) → "Character Recognition" → "Output Text" (monospace string)
  - Simulated image with text-like rectangles
  - Bounding boxes around detected regions
- **Colors**: Image = `gray_medium`, Detection = `primary` (boxes), Recognition = `success`, Text = `gray_dark`
- **Implementation**: `patches.Rectangle` for image and text regions, `patches.Rectangle` with colored borders for bounding boxes
- **Key Labels**: "Input Image", "f_detect(I)", "f_recognize(I[box])", "Output: 'P0562 14.2V'"

---

### Section 11: Logging & Observability

---

#### Figure 40: `fig:logging-architecture`
- **Function**: `plot_logging_architecture()`
- **Type**: Logger hierarchy tree
- **Dimensions**: 5.0 x 4.0 inches
- **Visual Elements**:
  - Root: "root Logger" at top
  - Children: "trace" → "trace.ml_predictor", "trace.llm_client", "trace.main"
  - Each module logger has handlers: "ConsoleHandler", "FileHandler"
  - Handlers connect to "Formatter"
  - Log level labels on each logger
- **Colors**: Root = `gray_dark`, Module loggers = `primary`, Handlers = `success`, Formatters = `purple`
- **Implementation**: `patches.FancyBboxPatch` for nodes, `ax.annotate()` for tree connections
- **Key Labels**: "root", "trace", "trace.ml_predictor", "StreamHandler", "FileHandler", "Formatter"

---

### Section 12: Containerization & Deployment

---

#### Figure 41: `fig:docker-layers`
- **Function**: `plot_docker_layers()`
- **Type**: Stacked layer diagram
- **Dimensions**: 5.0 x 4.0 inches
- **Visual Elements**:
  - Stacked rectangles (bottom to top):
    - "ubuntu:22.04" (base OS)
    - "Python 3.11 runtime"
    - "pip install -r requirements.txt"
    - "COPY . /app"
    - "CMD uvicorn main:app"
  - Each layer labeled with size estimate
  - "Image" bracket on the side
  - "Container" overlay showing writable top layer
- **Colors**: Alternating layer colors from palette, container overlay = `success` (semi-transparent)
- **Implementation**: `patches.FancyBboxPatch` stacked vertically, `ax.text()` for labels
- **Key Labels**: "Layer 1: Base OS", "Layer 2: Runtime", "Layer 3: Dependencies", "Layer 4: App Code", "Layer 5: Entrypoint"

---

#### Figure 42: `fig:health-check-cycle`
- **Function**: `plot_health_check_cycle()`
- **Type**: Circular cycle diagram
- **Dimensions**: 4.0 x 4.0 inches
- **Visual Elements**:
  - Circular flow: "Send Probe" → "Evaluate Response" → "Healthy?" → (Yes) "Wait interval" → loop back
  - Branch: (No) → "Increment failure count" → "≥ retries?" → (Yes) "Restart Container"
  - Clock/timer icon for interval
- **Colors**: Healthy path = `success`, Unhealthy path = `danger`, Decision = `secondary`
- **Implementation**: `patches.FancyBboxPatch` for steps, `ax.annotate()` for circular arrows, `patches.Arc` for cycle
- **Key Labels**: "HTTP GET /health", "Status 200?", "failures < retries", "Restart", "interval=30s"

---

### Section 13: Statistical Concepts

---

#### Figure 43: `fig:correlation-scatter`
- **Function**: `plot_correlation_scatter()`
- **Type**: Three scatter plots (positive, negative, zero)
- **Dimensions**: 6.5 x 2.5 inches
- **Data**: Generate with np.random.multivariate_normal for ρ=0.9, ρ=−0.9, ρ=0
- **Visual Elements**:
  - Three subplots side by side
  - Scatter points with regression line
  - Correlation coefficient and formula in each
  - Tight cluster for high |ρ|, scattered for ρ=0
- **Colors**: Points = `primary` (50% opacity), Regression = `danger`
- **Implementation**: Three subplots, `ax.scatter()`, `ax.plot()` for regression line
- **Key Labels**: "r = +0.9 (Strong Positive)", "r = −0.9 (Strong Negative)", "r = 0 (No Correlation)"

---

#### Figure 44: `fig:box-plot`
- **Function**: `plot_box_plot()`
- **Type**: Annotated box plot
- **Dimensions**: 5.0 x 3.0 inches
- **Data**: Generate skewed data: np.random.lognormal(mean=2.5, sigma=0.6, size=200)
- **Visual Elements**:
  - Box plot with full annotations:
    - Q1 (25th percentile) line
    - Median (Q2) line (thicker)
    - Q3 (75th percentile) line
    - IQR bracket showing Q3-Q1
    - Whiskers to 1.5×IQR
    - Outlier points beyond whiskers
  - Labels for each component
- **Colors**: Box = `primary` (fill), Median = `danger` (line), Whiskers = `gray_dark`, Outliers = `danger` (dots)
- **Implementation**: `ax.boxplot()` with custom styling, `ax.annotate()` for component labels, `ax.axhline()` for quartiles
- **Key Labels**: "Q1 (25%)", "Median (Q2)", "Q3 (75%)", "IQR = Q3-Q1", "Whiskers (1.5×IQR)", "Outliers"

---

## 4. Code Organization

### Module Structure
```
generate_figures.py
├── Imports
├── Constants (COLORS, OUTPUT_DIR, FIGURE_REGISTRY)
├── Helper Functions
│   ├── setup_style()
│   ├── create_figure(width, height)
│   ├── save_figure(fig, filename)
│   ├── style_axes(ax)
│   ├── add_arrow(ax, start, end, **kwargs)
│   ├── add_box(ax, x, y, w, h, text, **kwargs)
│   └── add_diamond(ax, x, y, size, text, **kwargs)
├── Figure Generation Functions (44 functions)
│   ├── Section 1: plot_supervised_learning_pipeline() ... plot_data_leakage_scenarios()
│   ├── Section 2: plot_tfidf_computation() ... plot_fuzzy_matching()
│   ├── ... (all 44)
│   └── Section 13: plot_correlation_scatter(), plot_box_plot()
├── main()
│   ├── Create output directory
│   ├── Setup matplotlib style
│   ├── Loop through FIGURE_REGISTRY
│   │   ├── Call figure function
│   │   ├── Save to PDF
│   │   └── Report progress
│   └── Print summary
└── if __name__ == "__main__": main()
```

### Helper Functions Detail

```python
def setup_style():
    """Configure matplotlib rcParams for academic style."""

def create_figure(width, height):
    """Create a new figure with given dimensions."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax

def save_figure(fig, filename):
    """Save figure as PDF with tight bounding box."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def style_axes(ax):
    """Apply consistent axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

def add_box(ax, x, y, w, h, text, **kwargs):
    """Add a styled text box at position (x,y) with width w, height h."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=kwargs.get('linewidth', 1.0)
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=kwargs.get('fontsize', 9),
            fontweight=kwargs.get('fontweight', 'normal'))
    return box

def add_arrow(ax, start, end, **kwargs):
    """Add an arrow from start=(x1,y1) to end=(x2,y2)."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=kwargs.get('color', COLORS['gray_medium']),
                               lw=kwargs.get('lw', 1.5)))

def add_diamond(ax, x, y, size, text, **kwargs):
    """Add a diamond-shaped decision node."""
    diamond = patches.Polygon(
        [[x, y+size], [x+size, y], [x, y-size], [x-size, y]],
        facecolor=kwargs.get('facecolor', COLORS['bg_box']),
        edgecolor=kwargs.get('edgecolor', COLORS['primary']),
        linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=8)
    return diamond
```

### Figure Registry
```python
FIGURE_REGISTRY = [
    # Section 1: Machine Learning Algorithms
    ("fig_supervised_learning_pipeline", plot_supervised_learning_pipeline, 6.5, 3.0),
    ("fig_multiclass_decision_boundary", plot_multiclass_decision_boundary, 4.0, 3.5),
    ...
    # Section 13: Statistical Concepts
    ("fig_correlation_scatter", plot_correlation_scatter, 6.5, 2.5),
    ("fig_box_plot", plot_box_plot, 5.0, 3.0),
]
```

### Error Handling
```python
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()
    results = []
    for name, func, w, h in FIGURE_REGISTRY:
        try:
            func()  # Each function creates and saves its own figure
            results.append((name, "OK"))
            print(f"  [{len(results)}/44] {name} ✓")
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            print(f"  [{len(results)}/44] {name} ✗ {e}")
    # Print summary
    ok = sum(1 for _, s in results if s == "OK")
    print(f"\nGenerated {ok}/44 figures successfully.")
```

---

## 5. Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `matplotlib` | Primary plotting library | `pip install matplotlib` |
| `numpy` | Data generation, arrays | `pip install numpy` |
| `seaborn` | Confusion matrix heatmap only | `pip install seaborn` |
| `scipy` | Distribution PDFs (norm, truncnorm, lognorm) | `pip install scipy` |

No other dependencies required. All diagrams are drawn using matplotlib primitives (patches, annotations, text).

---

## 6. Execution

### Run the Script
```bash
cd /mnt/d/study/git/capProj-2/Documentation
python generate_figures.py
```

### Expected Output
```
Generating 44 figures...
  [1/44] fig_supervised_learning_pipeline ✓
  [2/44] fig_multiclass_decision_boundary ✓
  ...
  [44/44] fig_box_plot ✓

Generated 44/44 figures successfully.
Output directory: /mnt/d/study/git/capProj-2/Documentation/figures/
```

### Output Files
44 PDF files in `figures/` directory:
```
figures/
├── fig_supervised_learning_pipeline.pdf
├── fig_multiclass_decision_boundary.pdf
├── fig_decision_tree_example.pdf
...
└── fig_box_plot.pdf
```

### LaTeX Integration
Replace each `\fbox{...}` placeholder in `Foundation_TRACE.tex` with:
```latex
\includegraphics[width=\textwidth]{figures/fig_name.pdf}
```

Example replacement for Figure 1:
```latex
% BEFORE:
\fbox{\parbox{0.85\textwidth}{\centering\vspace{1cm}FIGURE: ...\vspace{1cm}}}

% AFTER:
\includegraphics[width=0.85\textwidth]{figures/fig_supervised_learning_pipeline.pdf}
```

---

## 7. Quality Checklist

- [ ] All 44 figures generated without errors
- [ ] All PDFs are vector graphics (no rasterization — verify with `pdfinfo` or by zooming)
- [ ] Consistent font family (serif) across all figures
- [ ] Consistent color palette used throughout
- [ ] Text readable at document scale (minimum 8pt font)
- [ ] Colors distinguishable in grayscale print (avoid red/green only distinctions)
- [ ] No overlapping labels or annotations
- [ ] All axes properly labeled with units where applicable
- [ ] Legends present where multiple series are shown
- [ ] Figure aspect ratios appropriate for single-column or full-width placement
- [ ] PDF file sizes reasonable (< 500KB each for vector graphics)
- [ ] All mathematical notation uses proper Unicode or LaTeX-compatible symbols
- [ ] Decision boundaries and scatter plots use sufficient data points (n ≥ 200)
- [ ] Histograms use appropriate bin counts (15-30 bins)
- [ ] Flow diagrams have clear directional arrows
- [ ] Architecture diagrams maintain consistent box sizes and spacing

---

## 8. Implementation Notes

### Coordinate System
All flow/architecture diagrams use a normalized 10×10 coordinate system for consistent positioning:
- `x` ranges from 0 (left) to 10 (right)
- `y` ranges from 0 (bottom) to 10 (top)
- Boxes are typically 1.5-2.5 units wide, 0.8-1.2 units tall
- Arrows connect box centers or edges

### Text Sizing
- Titles: 12pt
- Axis labels: 10pt
- Box/annotation text: 9pt
- Small labels/formulas: 8pt
- Monospace (code/JSON): 8pt

### Mathematical Notation
Use Unicode characters for mathematical symbols:
- θ (theta), μ (mu), σ (sigma), π (pi)
- ŷ (y-hat), x̄ (x-bar)
- → (arrow), × (multiplication), ± (plus-minus)
- Subscripts: use Unicode subscripts where possible (₀, ₁, ₂) or regular text

### Seeded Randomness
For reproducibility, set `np.random.seed(42)` at the start of `main()` for all data generation.

### Performance
Expected generation time: 5-15 seconds for all 44 figures. Statistical plots (confusion matrix, calibration curve) may take slightly longer due to data generation.
