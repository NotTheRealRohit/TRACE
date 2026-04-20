# Methodology_TRACE — Materials, Methods, and Architecture Chapter Writing Plan

> **Target:** ~45–50 pages | **Style:** Technical methodology (TRACE-specific details, exact values, code references) | **Adapted from:** Reference document Chapter 4 pattern (Research → Data → Preparation → Models → Results)

---

## Document-Level Guidelines

### LaTeX Preamble Packages Required

```latex
\usepackage{amsmath, amssymb, mathtools}   % equations
\usepackage{graphicx}                       % figures
\usepackage{booktabs}                       % professional tables
\usepackage{multirow, multicol}             % complex table layouts
\usepackage{algorithm, algorithmic}         % pseudocode
\usepackage{tikz}                           % diagrams
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}
\usepackage{float}                          % [H] placement
\usepackage{hyperref}                       % cross-references
\usepackage{xcolor}                         % colored elements
\usepackage{listings}                       % code blocks
\usepackage{pythonhighlight}                % Python syntax highlighting (from NotesTeX style)
\usepackage{longtable}                      % multi-page tables
\usepackage{tabularx}                       % auto-width tables
```

### Writing Style Rules

| Rule | Specification |
|------|--------------|
| **Tone** | Technical methodology — precise, implementation-focused. Use "the TRACE system," "the prediction pipeline," "the rule engine" |
| **Scope** | TRACE-specific: exact hyperparameter values, file paths, line numbers, function names, rule definitions, feature lists, probability distributions |
| **Structure per subsection** | (1) Definition paragraph → (2) Mathematical formula(s) with exact TRACE values → (3) Figure placeholder → (4) TRACE-specific context (code references, exact values, design rationale) |
| **Cross-references** | Use `\ref{}` and `\label{}` for internal linking |
| **Equation numbering** | All equations numbered; use `align` for multi-line |
| **Figures** | Every figure gets `\caption{}` and `\label{fig:...}` |
| **Tables** | Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) for this chapter |
| **Page target** | 45–50 pages total at 10pt article class |
| **Code references** | Format as `\texttt{file.py:line}` with escaped underscores |
| **Constants** | Always include exact numerical values from the codebase |

### Figure Convention

Every figure placeholder follows this pattern:

```latex
\begin{figure}[H]
    \centering
    % \includegraphics[width=0.85\textwidth]{figures/figure_name.pdf}
    \fbox{\parbox{0.85\textwidth}{\centering\vspace{2cm}FIGURE: [description]\vspace{2cm}}}
    \caption{[Descriptive caption]}
    \label{fig:[label]}
\end{figure}
```

### Table Convention

Tables use `booktabs` with clear column headers. Multi-page tables use `longtable`. All tables include a source reference to the codebase.

---

## Section Hierarchy and Page Estimates

| Section | Subsections | Est. Pages |
|---------|------------|------------|
| **4.1 Research Methodology** | 4.1.1–4.1.4 | 3.5 |
| **4.2 Dataset Architecture** | 4.2.1–4.2.7 | 8.0 |
| **4.3 Data Preparation Pipeline** | 4.3.1–4.3.6 | 7.5 |
| **4.4 Model Architecture** | 4.4.1–4.4.6 | 10.0 |
| **4.5 Inference Pipeline** | 4.5.1–4.5.7 | 8.0 |
| **4.6 Evaluation Methodology** | 4.6.1–4.6.5 | 4.5 |
| **4.7 System Architecture \& Deployment** | 4.7.1–4.7.5 | 4.0 |
| **TOTAL** | **38 subsections** | **~45.5 pages** |

---

## 4.1 Research Methodology (~3.5 pages)

### 4.1.1 Hybrid Decision Engine Design (~1.0 pages)

**Definition:** The TRACE system employs a hybrid decision engine that integrates three complementary inference paradigms — a deterministic rule engine, a gradient-boosted ML cascade, and a large language model — into a unified warranty claim analysis pipeline. The architecture follows a six-stage processing flow where each stage contributes structured signals to the final decision.

**Formulas:**
- Hybrid decision function:
  $$\text{decision}(x) = \text{Combine}\big(f_{\text{LLM}}(x),\; f_{\text{rules}}(x),\; f_{\text{ML}}(x)\big)$$
- Decision engine tag assignment:
  $$\text{engine} = \begin{cases} \text{"LLM+Rule+ML"} & \text{if rule fires } \land \text{ LLM has signal} \\ \text{"Rule+ML"} & \text{if rule fires } \land \text{ no LLM signal} \\ \text{"LLM+ML"} & \text{if no rule fires } \land \text{ LLM has signal} \\ \text{"ML"} & \text{otherwise} \end{cases}$$
  where LLM has signal $\iff$ $\text{confidence}_{\text{LLM}} \geq 0.5$ (source: \texttt{ml\_predictor.py:652}).

**Figure:** `fig:hybrid-decision-engine-overview` — High-level architecture diagram showing the six-stage pipeline: (1) LLM Understanding → (2) Rule Engine → (3) Feature Translation → (4) XGBoost Cascade → (5) Score Combination → (6) Output Formatting, with data flow arrows between stages

**TRACE-specific context:** The hybrid engine is implemented in \texttt{ml\_predictor.py} (941 lines) with the main entry point at \texttt{predict()} (line 836). The design philosophy prioritizes deterministic rules for clear-cut cases (voltage extremes, keyword matches, DTC prefix patterns) while delegating ambiguous cases to the ML cascade. The LLM provides semantic understanding at Stage 1 and output formatting at Stage 6, with graceful degradation to rule-based and ML-only modes when API keys are unavailable. The system is served via FastAPI at \texttt{main.py:112} through the \texttt{/analyze} endpoint.

---

### 4.1.2 Six-Stage Pipeline Architecture (~1.0 pages)

**Definition:** The prediction pipeline processes each warranty claim through six sequential stages, each performing a specific transformation. Stages 1, 3, and 6 are LLM-enhanced with deterministic fallbacks; Stages 2, 4, and 5 are always active.

**Formulas:**
- Pipeline composition:
  $$y = f_6\Big(f_5\big(f_4(f_3(f_2(f_1(x))))\big)\Big)$$
  where:
  - $f_1$: LLM claim understanding (\texttt{llm\_client.py:346}, \texttt{understand\_claim()})
  - $f_2$: Rule engine evaluation (\texttt{ml\_predictor.py:465}, \texttt{run\_rules()})
  - $f_3$: Feature translation (\texttt{llm\_client.py:437} or fallback \texttt{extract\_dtc\_features()})
  - $f_4$: XGBoost cascade (\texttt{ml\_predictor.py:495}, \texttt{run\_ml()})
  - $f_5$: Score combination (\texttt{ml\_predictor.py:664}, \texttt{combine\_scores()})
  - $f_6$: Output formatting (\texttt{llm\_client.py:273} or fallback \texttt{assemble\_output\_from\_fields()})

**Figure:** `fig:six-stage-pipeline` — Detailed flow diagram with each stage showing: input types, processing logic, output types, and fallback paths

**TRACE-specific context:** The pipeline is orchestrated in the \texttt{predict()} function (\texttt{ml\_predictor.py:836–925}). Each stage has explicit error handling with \texttt{try/except} blocks that trigger fallbacks. The \texttt{DecisionLogger} class (\texttt{logging\_config.py:38}) logs structured output at each stage. LLM availability is checked at line 848: \texttt{api\_key\_available = bool(os.getenv("OPENROUTER\_API\_KEY"))}.

---

### 4.1.3 Research Problem Formulation (~0.75 pages)

**Definition:** The TRACE system addresses the warranty claim analysis problem as a dual classification task: (1) Failure Analysis — identifying the root cause of a component failure among 6 possible classes, and (2) Warranty Decision — determining financial responsibility among 3 possible classes (Production Failure, Customer Failure, According to Specification).

**Formulas:**
- Failure Analysis (6-class):
  $$\text{FA}: \mathcal{X} \to \{\text{NTF}, \text{Track burnt due to EOS}, \text{ASIC CJ327 failure due to EOS}, \text{Sensor short due to moisture}, \text{Connector damage}, \text{controller failure due to supplier production failure}\}$$
- Warranty Decision (3-class):
  $$\text{WD}: \mathcal{X} \to \{\text{Production Failure}, \text{Customer Failure}, \text{According to Specification}\}$$
- Input space:
  $$\mathcal{X} = \mathcal{D}_{\text{code}} \times \mathcal{T}_{\text{notes}} \times \mathbb{R}_{\text{voltage}}$$
  where $\mathcal{D}_{\text{code}}$ is the set of DTC codes, $\mathcal{T}_{\text{notes}}$ is free-text technician notes, and $\mathbb{R}_{\text{voltage}}$ is the measured voltage.

**TRACE-specific context:** The dual classification is handled by a cascade architecture where the FA classifier's probability vector is concatenated with the original features and fed to the WD classifier (\texttt{ml\_predictor.py:427}). This allows the WD classifier to leverage the FA model's confidence distribution as an additional feature. The 6 FA classes and 3 WD classes are defined in the synthetic dataset generator (\texttt{generate\_dataset\_v9(1).py}).

---

### 4.1.4 Evaluation Strategy Overview (~0.75 pages)

**Definition:** The evaluation methodology assesses the TRACE system at three levels: (1) isolated ML classifier performance on held-out test data, (2) end-to-end pipeline accuracy including rule engine overrides, and (3) cross-validation variance estimation on unseen data.

**Formulas:**
- Test set evaluation:
  $$\text{Acc}_{\text{test}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i \in \mathcal{D}_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)$$
- Cross-validation on held-out test set (3-fold):
  $$\text{CV}_{\text{acc}} = \frac{1}{3} \sum_{k=1}^{3} \text{Acc}(\mathcal{D}_{\text{test}}^{(k)})$$
- End-to-end pipeline evaluation:
  $$\text{Acc}_{\text{pipeline}} = \frac{1}{N_{\text{sample}}} \sum_{i=1}^{N_{\text{sample}}} \mathbb{I}(\text{predict}(x_i).\text{warranty\_decision} = y_i^{\text{WD}})$$

**TRACE-specific context:** Evaluation is implemented in \texttt{evaluate\_model.py} (528 lines). The script addresses five documented fixes from the original evaluator: data leakage prevention (FIX 1), correct cross-validation on held-out test set only (FIX 2), cascade calibration checking (FIX 3), end-to-end pipeline evaluation (FIX 4), and preprocessing consistency (FIX 5). The train/test split uses \texttt{test\_size=0.2, random\_state=42} (\texttt{ml\_predictor.py:297}).

---

## 4.2 Dataset Architecture (~8.0 pages)

### 4.2.1 Synthetic Dataset Overview (~0.75 pages)

**Definition:** The TRACE system is trained on a synthetically generated dataset of 100,000 warranty claims spanning model years 2019–2025, designed to mimic real-world automotive warranty data with realistic noise levels, feature correlations, and temporal drift.

**Formulas:**
- Dataset size:
  $$|\mathcal{D}| = 100{,}000 \text{ rows}, \quad 11 \text{ columns}$$
- Column schema:
  $$\text{Columns} = [\text{Customer}, \text{Year}, \text{Date}, \text{QC\_Number}, \text{Customer Complaint}, \text{DTC}, \text{Voltage}, \text{Failure Analysis}, \text{Warranty Decision}, \text{Supplier}, \text{Mileage\_km}]$$
- Pattern correlation target:
  $$\rho_{\text{patterns}} \approx 93\text{--}96\% \quad \text{(not 100\%, reflecting real-world noise)}$$

**Figure:** `fig:dataset-overview` — Table showing the 11 columns with example values, data types, and distributions

**TRACE-specific context:** The dataset is generated by \texttt{generate\_dataset\_v9(1).py} (779 lines) and saved as \texttt{synthetic\_warranty\_claims\_v9.csv}. The dataset uses \texttt{rng = np.random.default\_rng(42)} for reproducibility. The generator implements 10 named improvements (C4, C5, C6, W5, W6, W7, DTC1–DTC4) over previous versions. The dataset is loaded at \texttt{ml\_predictor.py:94}: \texttt{DATA\_PATH = os.path.join(BASE\_DIR, "synthetic\_warranty\_claims\_v9.csv")}.

---

### 4.2.2 Failure Analysis Class Distribution (~1.0 pages)

**Definition:** The dataset contains 6 Failure Analysis classes with year-aware proportional allocation and temporal drift modeling, where class proportions shift linearly across model years.

**Formulas:**
- Base FA proportions:
  $$\begin{aligned} P(\text{NTF}) &= 0.300, & P(\text{Track burnt}) &= 0.200, & P(\text{Connector}) &= 0.150 \\ P(\text{ASIC}) &= 0.120, & P(\text{Moisture}) &= 0.120, & P(\text{Controller}) &= 0.110 \end{aligned}$$
- Temporal drift per year (from base year 2019):
  $$\begin{aligned} \Delta_{\text{NTF}} &= +0.002/\text{yr}, & \Delta_{\text{Track}} &= -0.003/\text{yr}, & \Delta_{\text{Connector}} &= +0.004/\text{yr} \\ \Delta_{\text{ASIC}} &= -0.003/\text{yr}, & \Delta_{\text{Moisture}} &= +0.003/\text{yr}, & \Delta_{\text{Controller}} &= -0.003/\text{yr} \end{aligned}$$
- Normalized proportion for year $t$:
  $$P_{\text{norm}}(\text{FA}_i, t) = \frac{\max(0.01, P_{\text{base}}(\text{FA}_i) + \Delta_i \cdot (t - 2019))}{\sum_j \max(0.01, P_{\text{base}}(\text{FA}_j) + \Delta_j \cdot (t - 2019))}$$

**Figure:** `fig:fa-class-distribution` — Stacked bar chart showing FA class proportions across years 2019–2025 with drift visualization

**TRACE-specific context:** Implemented in \texttt{generate\_dataset\_v9(1).py:277–308} via \texttt{compute\_year\_counts()}. The drift models realistic shifts: NTF claims increase over time (improved diagnostics), connector damage increases (aging fleet), while track burnt and ASIC failures decrease (design improvements). Year weights follow \texttt{YEAR\_WEIGHTS = [0.04, 0.07, 0.10, 0.13, 0.18, 0.23, 0.25]} (line 30), reflecting increasing production volumes.

---

### 4.2.3 Voltage Distribution by Failure Class (~1.0 pages)

**Definition:** Each FA class is assigned a distinct voltage distribution using truncated normal distributions, creating separable voltage signatures that reflect real-world electrical failure modes.

**Formulas:**
- Truncated normal distribution:
  $$V \sim \mathcal{TN}(\mu, \sigma^2, a, b) \quad \text{with PDF:} \quad f(v) = \frac{\phi\left(\frac{v-\mu}{\sigma}\right)}{\sigma\left[\Phi\left(\frac{b-\mu}{\sigma}\right) - \Phi\left(\frac{a-\mu}{\sigma}\right)\right]}$$
- Class-specific parameters:
  $$\begin{array}{lcccc} \text{Class} & \mu & \sigma & [a, b] \\ \hline \text{ASIC CJ327} & 15.3 & 0.45 & [13.8, 16.5] \\ \text{Track burnt} & 17.8 & 1.10 & [15.5, 21.0] \\ \text{Controller} & 10.4 & 0.60 & [8.5, 12.5] \\ \text{Sensor moisture} & 12.7 & 0.55 & [10.5, 14.2] \\ \text{NTF} & 13.2 & 0.55 & [11.8, 14.8] \\ \text{Connector damage} & 13.3 & 0.65 & [11.5, 15.0] \end{array}$$

**Figure:** `fig:voltage-distributions` — Overlaid KDE plots showing voltage distributions for all 6 FA classes with threshold markers at 11.0V, 13.5V, 14.5V, 15.4V, 16.0V, 17.0V

**TRACE-specific context:** Voltage distributions are generated using \texttt{truncated\_normal()} (\texttt{generate\_dataset\_v9(1).py:206–213}). The EOS classes (ASIC, Track) receive an additional mileage-proportional voltage nudge (W5): \texttt{apply\_eos\_voltage\_nudge()} (line 215) adds \texttt{nudge\_scale $\times$ mileage/200,000} to the base voltage. Validation asserts \texttt{asic\_v.mean() > 15.0}, \texttt{track\_v.mean() > 17.0}, \texttt{ctrl\_v.mean() < 11.0} (lines 715–717).

---

### 4.2.4 Mileage Distribution by Failure Class (~1.0 pages)

**Definition:** Mileage is modeled using log-normal distributions with class-specific parameters, capturing the characteristic usage patterns associated with different failure modes. Connector damage uses a bimodal mixture distribution.

**Formulas:**
- Log-normal distribution:
  $$\text{Mileage} \sim \text{LogNormal}(\mu, \sigma^2), \quad \text{clipped to } [\text{min}, \text{max}]$$
- Class-specific parameters:
  $$\begin{array}{lcccc} \text{Class} & \mu_{\log} & \sigma_{\log} & [\text{min}, \text{max}] \\ \hline \text{Controller} & \ln(18{,}000) & 0.75 & [500, 90{,}000] \\ \text{ASIC} & \ln(45{,}000) & 0.65 & [3{,}000, 180{,}000] \\ \text{NTF} & \ln(35{,}000) & 0.80 & [500, 220{,}000] \\ \text{Sensor moisture} & \ln(50{,}000) & 0.70 & [2{,}000, 200{,}000] \\ \text{Track burnt} & \ln(60{,}000) & 0.65 & [1{,}000, 220{,}000] \\ \text{Connector (wear)} & \ln(75{,}000) & 0.60 & [15{,}000, 230{,}000] \end{array}$$
- Bimodal connector (W6):
  $$\text{Mileage}_{\text{conn}} = \begin{cases} \text{LogNormal}(\ln(2{,}500), 0.50^2) & \text{with } p = 0.15 \\ \text{LogNormal}(\ln(75{,}000), 0.60^2) & \text{with } p = 0.85 \end{cases}$$

**Figure:** `fig:mileage-distributions` — Overlaid histograms showing mileage distributions for all 6 FA classes, with inset showing the bimodal connector damage distribution

**TRACE-specific context:** Implemented in \texttt{gen\_mileage\_km()} (\texttt{generate\_dataset\_v9(1).py:249–271}). The bimodal connector distribution (W6) models two populations: 15\% early-life assembly defects ($<$8,000 km) and 85\% wear-and-tear ($\geq$8,000 km). Validation asserts mileage skewness $>$ 0.6 for all classes except Connector ($>$ 0.3) (lines 754–766). The warranty decision for connector damage is mileage-conditional: early-life uses \texttt{p=[0.92, 0.08]} (PF:CF), wear-and-tear uses \texttt{p=[0.80, 0.20]} (line 463–466).

---

### 4.2.5 DTC Code Architecture (~1.0 pages)

**Definition:** Diagnostic Trouble Codes (DTCs) are assigned to each FA class from native DTC pools with probabilistic companion code injection, cross-FA code injection, and DTC-complaint bias modeling.

**Formulas:**
- Native DTC pool assignment:
  $$\text{DTC}_{\text{primary}} \sim \text{Uniform}(\mathcal{P}_{\text{FA}}), \quad \mathcal{P}_{\text{FA}} \subset \text{all DTCs}$$
- Companion DTC injection (DTC3):
  $$P(\text{append companion } c' \mid c) = p_{\text{companion}}(c, c')$$
  where companion pairs include: $(\text{P0562}, \text{P0563}, 0.55)$, $(\text{P0691}, \text{P0692}, 0.60)$, $(\text{U0100}, \text{U0101}, 0.60)$, etc.
- Cross-FA injection rate (DTC2):
  $$P(\text{inject cross-FA DTC}) = 0.04$$

**Figure:** `fig:dtc-pool-architecture` — Diagram showing native DTC pools per FA class, companion pair connections, and cross-FA injection pathways

**TRACE-specific context:** The system defines 90+ high-value DTCs (\texttt{ml\_predictor.py:104–136}). Native pools are defined at \texttt{generate\_dataset\_v9(1).py:109–117}. Companion pairs are defined at lines 96–107 (13 pairs). Cross-FA injection uses \texttt{DTC\_AMBIGUOUS\_CROSS} (8 codes, line 39–48) with \texttt{CROSS\_FA\_INJECT\_RATE = 0.04} (line 49). The ASIC primary pool was expanded from 10 to 12 codes in v9 (DTC1, line 60–63). DTC-complaint bias is modeled via \texttt{DTC\_COMPLAINT\_BIAS} (lines 119–181, 80+ entries).

---

### 4.2.6 Warranty Decision Probabilities (~0.75 pages)

**Definition:** Warranty decisions are assigned probabilistically based on FA class and contextual factors (voltage thresholds, mileage), with class-specific probability distributions that reflect real-world warranty adjudication patterns.

**Formulas:**
- ASIC warranty decision (voltage-conditional):
  $$P(\text{WD} \mid \text{ASIC}, V) = \begin{cases} [0.78, 0.22, 0.00] & V \leq 14.7 \\ [0.60, 0.40, 0.00] & 14.7 < V < 15.4 \\ [0.38, 0.62, 0.00] & V \geq 15.4 \end{cases}$$
  where the vector is $[P(\text{PF}), P(\text{CF}), P(\text{ATS})]$.
- Track burnt:
  $$P(\text{WD} \mid \text{Track}) = [0.030, 0.960, 0.010]$$
- Controller:
  $$P(\text{WD} \mid \text{Controller}) = [0.960, 0.030, 0.010]$$
- NTF:
  $$P(\text{WD} \mid \text{NTF}) = [0.010, 0.025, 0.965]$$
- Sensor moisture (C4):
  $$P(\text{WD} \mid \text{Moisture}) = [0.010, 0.965, 0.025]$$
- Connector damage (mileage-conditional):
  $$P(\text{WD} \mid \text{Connector}, \text{mileage}) = \begin{cases} [0.92, 0.08, 0.00] & \text{mileage} < 8{,}000 \\ [0.80, 0.20, 0.00] & \text{mileage} \geq 8{,}000 \end{cases}$$

**Figure:** `fig:wd-probabilities` — Stacked bar chart showing warranty decision probabilities for each FA class

**TRACE-specific context:** Probabilities are implemented in each generator function: \texttt{gen\_asic\_cj327()} (lines 344–349), \texttt{gen\_track\_burnt()} (lines 375–378), \texttt{gen\_controller\_failure()} (lines 498–501), \texttt{gen\_ntf()} (lines 438–441), \texttt{gen\_sensor\_moisture()} (lines 411–414), \texttt{gen\_connector\_damage()} (lines 461–466). The C4 change made sensor moisture probabilistic (was 100\% Customer Failure). Validation asserts: \texttt{ntf\_ats\_rate $\geq$ 0.94}, \texttt{track\_cf\_rate $\geq$ 0.93}, \texttt{ctrl\_pf\_rate $\geq$ 0.93} (lines 729–733).

---

### 4.2.7 Label Noise Injection (~1.5 pages)

**Definition:** Label noise is injected post-generation to simulate real-world annotation errors, ambiguous boundary cases, and mileage-zone uncertainty. Six distinct noise mechanisms target specific FA classes with class-appropriate flip rates.

**Formulas:**
- ASIC boundary-zone noise:
  $$P(\text{flip} \mid \text{ASIC}, 14.8 \leq V \leq 15.2) = \eta \times 1.2, \quad \eta = 0.015$$
- Connector random noise:
  $$P(\text{flip} \mid \text{Connector}) = \eta = 0.015$$
- NTF noise:
  $$P(\text{ATS} \to \text{CF} \mid \text{NTF}) = 0.008$$
- Track burnt noise:
  $$P(\text{CF} \to \text{PF} \mid \text{Track}) = 0.007$$
- Controller noise:
  $$P(\text{PF} \to \text{CF} \mid \text{Controller}) = 0.007$$
- Sensor moisture noise:
  $$P(\text{CF} \to \text{PF} \mid \text{Moisture}) = 0.010$$
- Mileage boundary noise (W7):
  $$P(\text{flip} \mid 88{,}000 \leq \text{mileage} \leq 112{,}000) = 0.035$$

**Figure:** `fig:label-noise-mechanisms` — Diagram showing the six noise injection mechanisms with their target classes, flip directions, and rates

**TRACE-specific context:** Implemented in \texttt{inject\_warranty\_label\_noise()} (\texttt{generate\_dataset\_v9(1).py:515–595}). The function flips warranty decision labels (not FA labels) to simulate adjudication uncertainty. The mileage boundary noise (W7) targets the 88,000–112,000 km zone where warranty coverage transitions. Total noise rate is approximately 1.5\% of rows. Validation prints the total flipped count and boundary zone statistics (lines 593–594).

---

## 4.3 Data Preparation Pipeline (~7.5 pages)

### 4.3.1 Data Loading and Cleaning (~1.0 pages)

**Definition:** The data preparation pipeline begins by loading the CSV dataset and applying consistent null-filling and normalization to ensure all features are present and properly typed before any transformation.

**Formulas:**
- Null handling:
  $$\begin{aligned} \text{DTC} &\leftarrow \text{fillna}("") \text{ then replace}(\text{"none"}, "") \\ \text{Customer Complaint} &\leftarrow \text{fillna}(\text{"OBD Light ON"}) \\ \text{Failure Analysis} &\leftarrow \text{fillna}(\text{"NTF}) \\ \text{Warranty Decision} &\leftarrow \text{fillna}(\text{"According to Specification"}) \end{aligned}$$
- Label encoding:
  $$y_{\text{FA}} = \text{LabelEncoder}().\text{fit\_transform}(\text{df}[\text{"Failure Analysis"}])$$
  $$y_{\text{WD}} = \text{LabelEncoder}().\text{fit\_transform}(\text{df}[\text{"Warranty Decision"}])$$

**Figure:** `fig:data-loading-pipeline` — Flow diagram: CSV → null filling → label encoding → DTC feature extraction → train/test split

**TRACE-specific context:** Implemented in \texttt{train\_and\_save()} (\texttt{ml\_predictor.py:278–451}). Data loading at line 280: \texttt{df = pd.read\_csv(DATA\_PATH)}. Null handling at lines 281–284. LabelEncoders are fit on the full dataset (lines 288–289) — this is safe because target encoding does not expose test-set feature statistics. DTC features are extracted at line 291: \texttt{dtc\_feats = pd.DataFrame(list(df["DTC"].apply(extract\_dtc\_features)))}.

---

### 4.3.2 Train-Test Split Strategy (~1.0 pages)

**Definition:** The dataset is split into training (80\%) and test (20\%) sets BEFORE any transformer fitting, ensuring that no test-set statistics (vocabulary, IDF weights, mean, std) can leak into the transformers used for inference.

**Formulas:**
- Split configuration:
  $$\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}} = \text{train\_test\_split}(\mathcal{D}, \text{test\_size}=0.2, \text{random\_state}=42)$$
- Split objects:
  $$(\text{df}_{\text{tr}}, \text{df}_{\text{te}}, \text{dtc}_{\text{tr}}, \text{dtc}_{\text{te}}, y_{\text{fa\_tr}}, y_{\text{fa\_te}}, y_{\text{wd\_tr}}, y_{\text{wd\_te}})$$

**Figure:** `fig:train-test-split` — Diagram showing the split occurring BEFORE transformer fitting, with separate fit/transform paths for train and test

**TRACE-specific context:** The split occurs at \texttt{ml\_predictor.py:297–299}. The comment at lines 293–296 explicitly documents the rationale: "Splitting df before any fit\_transform call ensures that no test-set statistics can leak into the transformers." This addresses FIX 1 from the evaluation script. The split uses \texttt{random\_state=42} for reproducibility.

---

### 4.3.3 Feature Engineering — Derived Columns (~1.5 pages)

**Definition:** Five categories of derived features are engineered from raw columns to capture non-linear relationships, warranty eligibility signals, and cross-feature interactions that improve classifier performance.

**Formulas:**
- Mileage bracket (4 bins):
  $$\text{mileage\_bracket} = \begin{cases} \text{"low"} & 0 \leq \text{km} < 20{,}000 \\ \text{"mid"} & 20{,}000 \leq \text{km} < 60{,}000 \\ \text{"high"} & 60{,}000 \leq \text{km} < 100{,}000 \\ \text{"very\_high"} & \text{km} \geq 100{,}000 \end{cases}$$
- Claim age:
  $$\text{claim\_age} = \text{year}(\text{Date}) - \text{Year}$$
- Voltage bracket (7 bins):
  $$\text{voltage\_bracket} = \begin{cases} \text{"very\_low"} & V \leq 11.0 \\ \text{"low"} & 11.0 < V \leq 13.5 \\ \text{"normal"} & 13.5 < V \leq 14.5 \\ \text{"moderate\_high"} & 14.5 < V \leq 15.4 \\ \text{"high"} & 15.4 < V \leq 16.0 \\ \text{"very\_high"} & 16.0 < V \leq 17.0 \\ \text{"extreme"} & V > 17.0 \end{cases}$$
- DTC count bracket (4 bins):
  $$\text{dtc\_count\_bracket} = \begin{cases} \text{"none"} & c = 0 \\ \text{"single"} & c = 1 \\ \text{"few"} & 2 \leq c \leq 3 \\ \text{"many"} & c > 3 \end{cases}$$
- Interaction features (4 binary):
  $$\begin{aligned} \text{volt\_high\_and\_P} &= \mathbb{I}(V > 15.4 \land \text{has\_P} = 1) \\ \text{volt\_low\_and\_U} &= \mathbb{I}(V < 11.0 \land \text{has\_U} = 1) \\ \text{volt\_normal\_and\_C} &= \mathbb{I}(11.0 \leq V \leq 14.5 \land \text{has\_C} = 1) \\ \text{has\_multiple\_prefixes} &= \mathbb{I}(\text{has\_P} + \text{has\_U} + \text{has\_C} + \text{has\_B} > 1) \end{aligned}$$

**Figure:** `fig:derived-features` — Diagram showing raw columns → derived features with bin thresholds and interaction logic

**TRACE-specific context:** Derived features are computed at \texttt{ml\_predictor.py:303–351}. Mileage bins at lines 305–312. Claim age at lines 318–319. Voltage bracket at lines 323–332. DTC count bracket at lines 335–341. Interaction features at lines 344–351. The voltage threshold at 15.4V specifically separates ASIC CJ327 CF vs PF decisions. The claim age feature combines vehicle year with claim date for a direct warranty-eligibility signal absent from the raw 'Year' column.

---

### 4.3.4 Transformer Fitting Pipeline (~1.5 pages)

**Definition:** Twelve transformers are fitted exclusively on the training slice and then applied to both train and test slices, ensuring no data leakage. The transformers produce sparse matrices that are horizontally stacked into the final feature matrix.

**Formulas:**
- Transformer list and output dimensions:
  $$\begin{array}{lll} \text{Transformer} & \text{Type} & \text{Output} \\ \hline \text{OHE (complaint)} & \text{OneHotEncoder} & \text{sparse } (n \times |\mathcal{C}|) \\ \text{TF-IDF (DTC text)} & \text{TfidfVectorizer} & \text{sparse } (n \times 40) \\ \text{DTC flags} & \text{raw values} & \text{dense } (n \times (5 + |\text{HIGH\_DTCS}|)) \\ \text{OHE (supplier)} & \text{OneHotEncoder} & \text{sparse } (n \times 5) \\ \text{Scaler (mileage)} & \text{StandardScaler} & \text{dense } (n \times 1) \\ \text{Scaler (year)} & \text{StandardScaler} & \text{dense } (n \times 1) \\ \text{OHE (mileage bracket)} & \text{OneHotEncoder} & \text{sparse } (n \times 4) \\ \text{Scaler (claim age)} & \text{StandardScaler} & \text{dense } (n \times 1) \\ \text{Scaler (voltage)} & \text{StandardScaler} & \text{dense } (n \times 1) \\ \text{OHE (voltage bracket)} & \text{OneHotEncoder} & \text{sparse } (n \times 7) \\ \text{OHE (DTC count bracket)} & \text{OneHotEncoder} & \text{sparse } (n \times 4) \\ \text{Interactions} & \text{raw values} & \text{dense } (n \times 4) \end{array}$$
- Feature matrix assembly:
  $$X = \text{hstack}([X_c, X_d, X_n, X_s, X_m, X_y, X_{mb}, X_{ca}, X_v, X_{vb}, X_{dcb}, X_{int}])$$

**Figure:** `fig:transformer-pipeline` — Diagram showing 12 transformers fitted on training data, then applied to train and test separately, with hstack assembly

**TRACE-specific context:** Transformers are initialized at \texttt{ml\_predictor.py:359–368} and fitted at lines 370–381. Test transformation at lines 390–401. Feature matrix assembly at lines 384–387 (train) and 403–406 (test). All transformers use \texttt{sparse\_output=True} for memory efficiency. The TF-IDF vectorizer uses \texttt{max\_features=40} (line 360). All OHE encoders use \texttt{handle\_unknown="ignore"} for robustness to unseen categories at inference.

---

### 4.3.5 DTC Feature Extraction (~1.0 pages)

**Definition:** The \texttt{extract\_dtc\_features()} function parses comma-separated DTC code strings into structured features: prefix flags (P/U/C/B), code count, TF-IDF text, and one-hot flags for 90+ high-value DTCs.

**Formulas:**
- DTC parsing:
  $$\text{codes} = [c.\text{strip()} \mid c \in \text{split}(\text{dtc\_str}, \text{","}), c.\text{strip()} \neq ""]$$
- Prefix flags:
  $$\text{has\_P} = \mathbb{I}(\exists c \in \text{codes}: c.\text{startswith}(\text{"P"}))$$
- High-value DTC one-hot:
  $$\text{dtc\_}\{d\} = \mathbb{I}(d \in \text{codes}), \quad \forall d \in \text{HIGH\_VALUE\_DTCS}$$
- Null handling:
  $$\text{if dtc\_str} \in \{"", \text{"NA"}, \text{"NAN"}, \text{"NONE"}\} \Rightarrow \text{all zeros}$$

**Figure:** `fig:dtc-feature-extraction` — Flow diagram: raw DTC string → split → prefix detection → high-value matching → feature vector

**TRACE-specific context:** Implemented at \texttt{ml\_predictor.py:229–243}. The \texttt{HIGH\_VALUE\_DTCS} list contains 90+ codes (lines 104–136), organized by category: misfire codes (P0300–P0306), ignition (P0351–P0356), OBD processor (P0562–P0563), ECM/PCM (P0601–P0617), power supply (P0620, P0691–P0694), catalyst (P0420, P0430), EVAP (P0455–P0457), vehicle speed (P0500–P0502), CAN bus (U0001–U0184), body (B1031–B3055), chassis (C0031–C0550), and sensor codes (P0038–P0343).

---

### 4.3.6 Complaint Matching (~0.5 pages)

**Definition:** The \texttt{match\_complaint()} function maps free-text technician notes to one of 14 known complaint labels using a keyword-first strategy with fuzzy string matching fallback.

**Formulas:**
- Keyword mapping (first match wins):
  $$\text{complaint} = \text{value}(k) \quad \text{where } k = \text{first key in } \mathcal{K} \text{ such that } k \in \text{lowercase}(\text{notes})$$
- Fuzzy fallback:
  $$\text{complaint} = \begin{cases} \text{get\_close\_matches}(\text{notes}, \text{KNOWN\_COMPLAINTS}, n=1, \text{cutoff}=0.25)[0] & \text{if matches exist} \\ \text{"OBD Light ON"} & \text{otherwise} \end{cases}$$
- Default:
  $$\text{complaint} = \text{"OBD Light ON"} \quad \text{if notes is empty}$$

**Figure:** `fig:complaint-matching` — Flow diagram: notes → keyword scan → first match → complaint label; if no match → fuzzy match → complaint label

**TRACE-specific context:** Implemented at \texttt{ml\_predictor.py:246–274}. The keyword map (\texttt{kmap}, lines 250–269) contains 16 keyword-to-complaint mappings. The \texttt{KNOWN\_COMPLAINTS} list (lines 96–102) contains 14 complaint labels. The function uses Python's \texttt{difflib.get\_close\_matches} with \texttt{cutoff=0.25} for fuzzy matching.

---

## 4.4 Model Architecture (~10.0 pages)

### 4.4.1 XGBoost Classifier Configuration (~1.5 pages)

**Definition:** The TRACE system uses XGBoost gradient-boosted trees as the core ML classifier, configured with 1,000 estimators, maximum depth 10, and learning rate 0.02, optimized through hyperparameter tuning.

**Formulas:**
- XGBoost hyperparameters:
  $$\begin{aligned} \text{n\_estimators} &= 1000 \\ \text{max\_depth} &= 10 \\ \text{learning\_rate} &= 0.02 \\ \text{min\_child\_weight} &= 3 \\ \text{subsample} &= 0.8 \\ \text{colsample\_bytree} &= 0.8 \\ \text{reg\_lambda} &= 0.1 \\ \text{eval\_metric} &= \text{"mlogloss"} \\ \text{random\_state} &= 42 \end{aligned}$$
- Additive model update:
  $$F_{t+1}(x) = F_t(x) + 0.02 \cdot h_{t+1}(x)$$
- Regularized objective:
  $$\mathcal{L}(\theta) = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{1000} \left(0.1 \sum_{j=1}^{T_k} w_j^2\right)$$
  (L2 regularization only; no L1 term in this configuration)

**Figure:** `fig:xgboost-architecture` — Diagram showing sequential tree building with learning rate scaling, subsampling, and column sampling

**TRACE-specific context:** Hyperparameters are defined at \texttt{ml\_predictor.py:409–413} in the \texttt{\_xgb\_params} dictionary. The configuration was optimized through hyperparameter tuning (documented in the evaluation script header comments). Both classifiers (FA and WD) use identical hyperparameters. The \texttt{n\_jobs=-1} flag enables parallel tree construction.

---

### 4.4.2 Cascade Architecture — Out-of-Fold Probabilities (~1.5 pages)

**Definition:** The cascade architecture uses out-of-fold (OOF) probabilities from the FA classifier as additional input features for the WD classifier, preventing data leakage that would occur if the FA classifier scored its own training data.

**Formulas:**
- OOF probability generation (5-fold CV):
  $$\mathbf{p}^{\text{OOF}}_i = f_{\theta_{-k}}^{\text{FA}}(x_i), \quad \text{where } i \in \mathcal{D}_k, \; k = 1,\ldots,5$$
- WD training with OOF:
  $$X_{\text{WD}}^{\text{train}} = \text{hstack}([X_{\text{train}}, \mathbf{p}^{\text{OOF}}])$$
- WD inference with test probabilities:
  $$X_{\text{WD}}^{\text{test}} = \text{hstack}([X_{\text{test}}, f_{\theta}^{\text{FA}}(X_{\text{test}})])$$
- WD prediction:
  $$\hat{y}_{\text{WD}} = \arg\max_k f_{\theta_{\text{WD}}}(X_{\text{WD}}^{\text{test}})_k$$

**Figure:** `fig:cascade-oof` — Diagram showing 5-fold OOF probability generation for training, and direct FA prediction for inference, with both feeding into WD classifier

**TRACE-specific context:** OOF probabilities are generated at \texttt{ml\_predictor.py:417–422} using \texttt{cross\_val\_predict(..., cv=5, method="predict\_proba")}. The FA classifier is then retrained on full training data at line 423–424. Test FA probabilities at line 426. WD feature augmentation at lines 427–428. WD training at lines 431–432. This addresses FIX 3 from the evaluation script: the original code used \texttt{clf\_fa.predict\_proba(X\_tr)} which produced overconfident probabilities.

---

### 4.4.3 Rule Engine — Complete Rule Specification (~2.0 pages)

**Definition:** The rule engine consists of 9 deterministic rules evaluated in priority order using a first-match-wins strategy. Each rule specifies a matching condition, failure analysis, warranty decision, status, confidence, and human-readable reason.

**Formulas:**
- Rule evaluation:
  $$\text{result}(x) = \begin{cases} R_1 & \text{if } \phi_1(x) \\ R_2 & \text{if } \neg\phi_1(x) \land \phi_2(x) \\ \vdots \\ R_9 & \text{if } \bigwedge_{i=1}^{8} \neg\phi_i(x) \land \phi_9(x) \\ \text{None} & \text{if } \bigwedge_{i=1}^{9} \neg\phi_i(x) \end{cases}$$

**Table: Complete Rule Specification**

| # | Rule ID | Match Condition | Failure Analysis | Warranty Decision | Status | Confidence |
|---|---------|----------------|-----------------|-------------------|--------|------------|
| 1 | \texttt{over\_voltage} | $V > 16.0$ | EOS due to over-voltage | Customer Failure | Rejected | 93.0\% |
| 2 | \texttt{low\_voltage} | $V < 11.0$ | Sensor short due to low voltage | Customer Failure | Rejected | 95.0\% |
| 3 | \texttt{moisture} | $\exists k \in \{\text{water, moisture, wet, flood, rain, humid, corrosion, corroded}\} \subseteq \text{notes}$ | Sensor short due to moisture | Customer Failure | Rejected | 91.0\% |
| 4 | \texttt{physical\_damage} | $\exists k \in \{\text{crack, broken, impact, collision, bent, misuse, dropped, physical damage}\} \subseteq \text{notes}$ | Connector damage | Customer Failure | Rejected | 88.5\% |
| 5 | \texttt{ntf} | $\exists k \in \{\text{no fault, ntf, no trouble, no issue, no defect, intermittent, cannot reproduce}\} \subseteq \text{notes}$ | NTF | According to Specification | Approved | 95.0\% |
| 6 | \texttt{u\_code} | $\exists \text{DTC matching } \backslash b\text{U}[0\text{-}9\text{A-Fa-f}]\{4\}\backslash b$ | Controller failure (supplier production) | Production Failure | Approved | 57.0\% |
| 7 | \texttt{p\_code\_engine} | $\exists \text{DTC matching } \backslash b\text{P0}[0\text{-}9]\{3\}\backslash b \land \exists k \in \{\text{jerk, pickup, acceleration, overheat, fuel, idle, rough}\} \subseteq \text{notes}$ | ASIC CJ327 failure due to EOS | Production Failure | Approved | 80.5\% |
| 8 | \texttt{c\_code} | $\exists \text{DTC matching } \backslash b\text{C}[0\text{-}9\text{A-Fa-f}]\{4\}\backslash b$ | Connector damage | Production Failure | Approved | 80.0\% |
| 9 | \texttt{b\_code} | $\exists \text{DTC matching } \backslash b\text{B}[0\text{-}9\text{A-Fa-f}]\{4\}\backslash b$ | Connector damage | Production Failure | Approved | 80.0\% |

**Figure:** `fig:rule-engine-flow` — Flowchart showing sequential rule evaluation with first-match-wins logic

**TRACE-specific context:** Rules are defined at \texttt{ml\_predictor.py:138–226}. The \texttt{run\_rules()} function (lines 465–492) iterates through rules and returns on first match. The u\_code rule has the lowest confidence (57.0\%) because U-codes indicate communication faults that may be intermittent. The p\_code\_engine rule requires both a P0-series DTC AND matching symptom keywords. Rule confidences were recalibrated for v9's noisy patterns (noted at line 28).

---

### 4.4.4 LLM Integration Architecture (~1.5 pages)

**Definition:** The LLM integration provides three services: (1) claim understanding/categorization at Stage 1, (2) feature translation at Stage 3, and (3) output formatting at Stage 6. Each service supports both OpenAI (gpt-4o-mini) and OpenRouter (arcee-ai/trinity-large-preview:free) providers with automatic fallback.

**Formulas:**
- Provider selection:
  $$\text{provider} = \begin{cases} \text{"openai"} & \text{if OPENAI\_API\_KEY is set} \\ \text{"openrouter"} & \text{if OPENROUTER\_API\_KEY is set} \\ \text{None} & \text{otherwise} \end{cases}$$
- Retry with exponential backoff:
  $$\text{delay}_k = 2^k \text{ seconds}, \quad k \in \{0, 1\}, \quad \text{max\_retries} = 2$$
- LLM availability check:
  $$\text{llm\_available} = \text{api\_key\_available} \land \text{len}(\text{notes}) > 5$$

**Figure:** `fig:llm-integration` — Architecture diagram showing LLM client with dual provider support, retry logic, and three service endpoints

**TRACE-specific context:** Implemented in \texttt{llm\_client.py} (476 lines). Provider selection at lines 28–33. OpenAI client at lines 54–60. OpenRouter API call at lines 106–159 with timeout handling and rate limit detection (429). Retry logic at lines 380–405 with \texttt{max\_retries=2} and exponential backoff (\texttt{2**attempt}). The three LLM services use distinct prompts: \texttt{UNDERSTAND\_CLAIM\_PROMPT} (lines 306–343), \texttt{TRANSLATE\_ML\_FEATURES\_PROMPT} (lines 408–434), \texttt{FORMAT\_OUTPUT\_PROMPT} (lines 243–270). All prompts request JSON output with \texttt{temperature=0, seed=42} for determinism.

---

### 4.4.5 LLM Claim Categorization (~1.0 pages)

**Definition:** The Stage 1 LLM service categorizes warranty claims into one of 7 semantic categories using a structured prompt with disambiguation rules, producing category, normalized complaint, severity, failure analysis, reasoning, and confidence.

**Formulas:**
- Category mapping to warranty decision:
  $$\text{WD}_{\text{LLM}} = \begin{cases} \text{"Customer Failure"} & \text{if category} \in \{\text{moisture\_damage, physical\_damage}\} \\ \text{"According to Specification"} & \text{if category} = \text{ntf} \\ \text{"Production Failure"} & \text{if category} \in \{\text{electrical\_issue, engine\_symptom, communication\_fault}\} \\ \text{None} & \text{if category} = \text{other} \end{cases}$$
- Disambiguation rules (first match wins):
  $$\begin{aligned} &\text{1. overheating/jerking/pickup/acceleration/fuel/idle/rough} \to \text{engine\_symptom} \\ &\text{2. CAN bus/LIN bus/communication/network/U-code} \to \text{communication\_fault} \\ &\text{3. moisture/water/wet/flood/rain/humidity/corrosion} \to \text{moisture\_damage} \\ &\text{4. crack/broken/impact/collision/bent/misuse/dropped} \to \text{physical\_damage} \\ &\text{5. no fault/ntf/no trouble/no issue/no defect/intermittent} \to \text{ntf} \\ &\text{6. electrical short/wiring (without engine symptoms)} \to \text{electrical\_issue} \\ &\text{7. otherwise} \to \text{other} \end{aligned}$$

**Figure:** `fig:llm-categorization` — Diagram showing input notes+DTC → LLM prompt → 7-category output with confidence

**TRACE-specific context:** Implemented via \texttt{understand\_claim()} (\texttt{llm\_client.py:346–377}) and \texttt{understand\_claim\_with\_retry()} (lines 380–405). The prompt (\texttt{UNDERSTAND\_CLAIM\_PROMPT}, lines 306–343) includes 7 disambiguation rules applied in order. The normalized complaint must be one of 9 exact strings. The category-to-warranty mapping is defined at \texttt{ml\_predictor.py:634–642}.

---

### 4.4.6 Score Combination Logic (~1.5 pages)

**Definition:** The score combiner merges rule engine confidence, ML confidence, and LLM confidence using weighted linear blending with conditional weights based on inter-source agreement.

**Formulas:**
- Constants:
  $$\begin{aligned} \tau_{\text{firm}} &= 85.0, & \tau_{\text{manual}} &= 65.0, & b_{\text{agree}} &= 5.0 \\ w_{\text{rule}}^{\text{agree}} &= 0.70, & w_{\text{ml}}^{\text{agree}} &= 0.30 \\ w_{\text{rule}}^{\text{disagree}} &= 0.55, & w_{\text{ml}}^{\text{disagree}} &= 0.35 \\ w_{\text{llm}} &= 0.15 \\ w_{\text{rule}}^{\text{agree\_llm}} &= 0.595, & w_{\text{ml}}^{\text{agree\_llm}} &= 0.255 \\ w_{\text{rule}}^{\text{disagree\_llm}} &= 0.4675, & w_{\text{ml}}^{\text{disagree\_llm}} &= 0.2975 \end{aligned}$$
- Scenario A: Rule fires AND agrees with ML (with LLM):
  $$c_{\text{combined}} = 0.595 \cdot c_{\text{rule}} + 0.255 \cdot c_{\text{ml}} + 0.15 \cdot (c_{\text{llm}} \times 100) + 5.0$$
- Scenario A: Rule fires AND agrees with ML (without LLM agreement):
  $$c_{\text{combined}} = 0.70 \cdot c_{\text{rule}} + 0.30 \cdot c_{\text{ml}} + 2.0$$
- Scenario B: Rule fires BUT disagrees with ML (with LLM agreement):
  $$c_{\text{combined}} = 0.4675 \cdot c_{\text{rule}} + 0.2975 \cdot c_{\text{ml}} + 0.15 \cdot (c_{\text{llm}} \times 100)$$
- Scenario B: Rule fires BUT disagrees with ML (no LLM agreement):
  $$c_{\text{combined}} = 0.55 \cdot c_{\text{rule}} + 0.35 \cdot c_{\text{ml}}$$
- Scenario C: No rule fires:
  $$c_{\text{combined}} = 0.85 \cdot c_{\text{ml}} + 0.15 \cdot (c_{\text{llm}} \times 100)$$
  With weak input penalty: if $c_{\text{llm}} < 0.3$, then $c_{\text{combined}} \leftarrow c_{\text{combined}} \times 0.7$
- Final clamping:
  $$c_{\text{final}} = \text{round}(\min(98.0, \max(0.0, c_{\text{combined}})), 1)$$

**Figure:** `fig:score-combination` — Decision tree showing all combination scenarios with weights and bonuses

**TRACE-specific context:** Implemented in \texttt{combine\_scores()} (\texttt{ml\_predictor.py:664–808}). Constants are defined at lines 623–651. The disagreement gap threshold is \texttt{DISAGREEMENT\_GAP\_THRESHOLD = 20.0} (line 626). Status determination uses three tiers (lines 747–773): $\geq 85\%$ firm, $65\text{--}85\%$ cautious, $< 65\%$ manual review. The LLM agreement check is implemented via \texttt{\_llm\_agrees\_with\_decision()} (lines 655–661).

---

## 4.5 Inference Pipeline (~8.0 pages)

### 4.5.1 Stage 1 — LLM Claim Understanding (~1.0 pages)

**Definition:** The first stage calls the LLM to semantically understand the warranty claim, producing a category, normalized complaint, severity assessment, and preliminary failure analysis before any rule or ML logic runs.

**Formulas:**
- Input:
  $$x_1 = (\text{technician\_notes}, \text{fault\_code})$$
- Output:
  $$y_1 = \text{understand\_claim}(x_1) = \{\text{category}, \text{normalized\_complaint}, \text{severity}, \text{failure\_analysis}, \text{reasoning}, \text{confidence}\}$$
- Availability:
  $$\text{Stage 1 active} \iff \text{OPENROUTER\_API\_KEY} \in \text{env} \land \text{len}(\text{notes}) > 5$$

**Figure:** `fig:stage1-llm-understanding` — Flow diagram: notes + DTC → LLM prompt → JSON response → structured output

**TRACE-specific context:** Called at \texttt{ml\_predictor.py:852–859}. The function \texttt{understand\_claim\_with\_retry()} is imported from \texttt{llm\_client} inside a try block to handle import failures gracefully. If the LLM fails, \texttt{llm\_stage1} remains \texttt{None} and the pipeline continues with fallback logic. The result is logged via \texttt{DecisionLogger.log\_decision("Stage 1 LLM", llm\_stage1)} (line 857).

---

### 4.5.2 Stage 2 — Rule Engine Evaluation (~1.0 pages)

**Definition:** The second stage evaluates all 9 rules against the claim inputs in priority order, returning the first matching rule's output or indicating no rule fired.

**Formulas:**
- Input:
  $$x_2 = (\text{fault\_code}, \text{technician\_notes}, \text{voltage})$$
- Output:
  $$y_2 = \text{run\_rules}(x_2) = \{\text{rule\_id}, \text{status}, \text{warranty\_decision}, \text{rule\_confidence}, \text{failure\_analysis}, \text{reason}, \text{rule\_fired}\}$$
- First-match-wins:
  $$y_2 = R_j \quad \text{where } j = \min\{i : \phi_i(x_2) = \text{true}\}$$

**Figure:** `fig:stage2-rule-engine` — Flow diagram: inputs → sequential rule evaluation → first match → output

**TRACE-specific context:** Called at \texttt{ml\_predictor.py:861}. The \texttt{run\_rules()} function (lines 465–492) wraps each rule evaluation in a try/except to handle potential errors in individual rule predicates. If a rule fires, it is logged via \texttt{DecisionLogger.log\_decision("Rule Engine", rule\_result)} (lines 862–863). The rule ID and confidence are logged at line 864–865.

---

### 4.5.3 Stage 3 — Feature Translation (~1.5 pages)

**Definition:** The third stage translates raw claim inputs into structured ML features. When LLM is available, it uses semantic understanding to extract features; otherwise, it falls back to deterministic DTC parsing and complaint matching.

**Formulas:**
- LLM path:
  $$\text{features}_{\text{LLM}} = \text{translate\_to\_ml\_features}(\text{notes}, \text{fc}, \text{category})$$
  Output: $\{\text{customer\_complaint}, \text{dtc\_codes}, \text{dtc\_text}, \text{dtc\_count}, \text{has\_P}, \text{has\_U}, \text{has\_C}, \text{has\_B}\}$
- Fallback path:
  $$\begin{aligned} \text{dtc\_f} &= \text{extract\_dtc\_features}(\text{fc}) \\ \text{features}_{\text{fallback}} &= \{\text{customer\_complaint}: \text{match\_complaint}(\text{notes}), \\ &\quad \text{dtc\_text}: \text{dtc\_f}[\text{"dtc\_text"}], \\ &\quad \text{dtc\_count}: \text{dtc\_f}[\text{"dtc\_count"}], \\ &\quad \text{has\_P/U/C/B}: \text{dtc\_f}[\text{"has\_P/U/C/B"}], \\ &\quad \text{supplier}: \text{"Unknown"}, \text{mileage\_km}: 50000.0, \text{year}: 2024, \\ &\quad \text{claim\_age}: 1, \text{voltage}: \text{voltage or } 14.2\} \end{aligned}$$
- DTC feature merge (LLM path):
  $$\text{features}.\text{update}(\{k: v \mid k \text{ starts with "dtc\_"} \text{ in dtc\_f}\})$$

**Figure:** `fig:stage3-feature-translation` — Two-path diagram: LLM path (semantic extraction) vs. fallback path (deterministic parsing), converging to unified feature dict

**TRACE-specific context:** Called at \texttt{ml\_predictor.py:868–895}. The LLM path is attempted first (lines 869–876); if it fails, the fallback is used (lines 879–895). The fallback sets default values: supplier="Unknown", mileage\_km=50000.0, year=2024, claim\_age=1, voltage=14.2 (or actual voltage if provided). Voltage is ensured to be in features even if the LLM path was taken (lines 898–899).

---

### 4.5.4 Stage 4 — XGBoost Cascade Scoring (~1.5 pages)

**Definition:** The fourth stage runs the two XGBoost classifiers in cascade: the FA classifier predicts root cause, and its probability vector is concatenated with the original features to form the WD classifier input.

**Formulas:**
- Feature construction (inference-time):
  $$X_{\text{row}} = \text{hstack}([X_c, X_d, X_n, X_s, X_m, X_y, X_{mb}, X_{ca}, X_v, X_{vb}, X_{dcb}, X_{int}])$$
  where each component is transformed using the fitted transformers from the model bundle.
- FA prediction:
  $$\mathbf{p}_{\text{FA}} = f_{\theta_{\text{FA}}}(X_{\text{row}}), \quad \hat{y}_{\text{FA}} = \arg\max \mathbf{p}_{\text{FA}}, \quad c_{\text{FA}} = \max \mathbf{p}_{\text{FA}}$$
- WD feature augmentation:
  $$X_{\text{WD}} = \text{hstack}([X_{\text{row}}, \mathbf{p}_{\text{FA}}])$$
- WD prediction:
  $$\mathbf{p}_{\text{WD}} = f_{\theta_{\text{WD}}}(X_{\text{WD}}), \quad \hat{y}_{\text{WD}} = \arg\max \mathbf{p}_{\text{WD}}, \quad c_{\text{WD}} = \max \mathbf{p}_{\text{WD}}$$
- ML confidence (geometric mean):
  $$c_{\text{ML}} = \text{round}(\min(98.0, \max(0.0, \sqrt{c_{\text{FA}} \cdot c_{\text{WD}}} \times 100)), 1)$$

**Figure:** `fig:stage4-xgboost-cascade` — Diagram showing single-row feature construction → FA classifier → probability vector → concatenation → WD classifier → geometric mean confidence

**TRACE-specific context:** Implemented in \texttt{run\_ml()} (\texttt{ml\_predictor.py:495–620}). The model bundle is lazily loaded at lines 512–514. Feature construction uses the same 12 transformer components as training (lines 527–594). The FA probability row is obtained at line 600. WD augmentation at line 604. Geometric mean confidence at line 612: \texttt{min(98.0, max(0.0, (fa\_prob * wd\_prob) ** 0.5 * 100))}. The confidence is clamped to [0, 98].

---

### 4.5.5 Stage 5 — Score Combination (~1.0 pages)

**Definition:** The fifth stage combines rule engine confidence, ML confidence, and optional LLM confidence into a single combined confidence score using the weighted blending logic described in Section 4.4.6.

**Formulas:**
- Input:
  $$(\text{rule\_result}, \text{ml\_result}, \text{llm\_stage1})$$
- Agreement detection:
  $$\text{agreement} = (\text{rule\_wd} == \text{ml\_wd})$$
- LLM agreement:
  $$\text{llm\_agrees\_rule} = \text{WD}_{\text{LLM}}(\text{category}) == \text{rule\_wd}$$
  $$\text{llm\_agrees\_ml} = \text{WD}_{\text{LLM}}(\text{category}) == \text{ml\_wd}$$
- Output:
  $$y_5 = \{\text{status}, \text{warranty\_decision}, \text{combined\_confidence}, \text{agreement}, \text{rule\_fired}, \text{rule\_id}, \text{decision\_engine}\}$$

**Figure:** `fig:stage5-score-combination` — Flow diagram: rule result + ML result + LLM result → agreement check → weighted blend → status determination → output

**TRACE-specific context:** Called at \texttt{ml\_predictor.py:908}. The \texttt{combine\_scores()} function (lines 664–808) handles all combination scenarios. The decision engine tag is determined at lines 780–787. LLM agreement logging at lines 789–795.

---

### 4.5.6 Stage 6 — Output Formatting (~1.0 pages)

**Definition:** The sixth stage formats the combined decision into a human-readable output. When LLM is available, it uses a structured prompt to generate a natural-language reason; otherwise, it falls back to template-based assembly.

**Formulas:**
- LLM path:
  $$\text{output}_{\text{LLM}} = \text{format\_output}(\text{combined}, \text{features})$$
- Fallback path:
  $$\text{reason} = \begin{cases} \text{"Rule '" + rule\_id + "' fired. ML " + ("agrees"/"disagrees") + " with confidence " + conf + "\%."} & \text{if rule fired} \\ \text{"No rule matched. ML predicts " + ml\_wd + " with confidence " + conf + "\%."} & \text{otherwise} \end{cases}$$
- Output schema:
  $$\text{output} = \{\text{status}, \text{failure\_analysis}, \text{warranty\_decision}, \text{confidence}, \text{reason}, \text{matched\_complaint}, \text{decision\_engine}\}$$

**Figure:** `fig:stage6-output-formatting` — Two-path diagram: LLM path (natural language generation) vs. fallback path (template assembly)

**TRACE-specific context:** Called at \texttt{ml\_predictor.py:911–920}. The LLM path is attempted first (lines 912–915); if it fails, the fallback \texttt{assemble\_output\_from\_fields()} is used (lines 811–833). The fallback constructs a reason string from the structured fields. The final output is logged at lines 922–923.

---

### 4.5.7 Complete Prediction Flow (~1.0 pages)

**Definition:** The \texttt{predict()} function orchestrates all six stages into a complete prediction pipeline, handling LLM availability checks, error recovery, and structured logging at each stage.

**Formulas:**
- Full pipeline:
  $$\text{predict}(\text{fc}, \text{notes}, V) = f_6(f_5(f_4(f_3(f_2(f_1(\text{fc}, \text{notes}, V))))))$$
- With fallbacks:
  $$\text{predict}(x) = \begin{cases} f_6^{\text{LLM}}(f_5(f_4(f_3^{\text{LLM}}(f_2(f_1^{\text{LLM}}(x)))))) & \text{full LLM path} \\ f_6^{\text{fallback}}(f_5(f_4(f_3^{\text{fallback}}(f_2(\text{None}))))) & \text{no LLM path} \end{cases}$$

**Figure:** `fig:complete-predict-flow` — End-to-end flow diagram showing all six stages with error handling paths and fallback branches

**TRACE-specific context:** The \texttt{predict()} function spans \texttt{ml\_predictor.py:836–925}. Input normalization at lines 842–843. LLM availability check at lines 848–849. Each LLM stage is wrapped in try/except. The model bundle is lazily loaded at lines 837–839. The function returns a dict matching the \texttt{ClaimResponse} Pydantic model (\texttt{main.py:94–101}).

---

## 4.6 Evaluation Methodology (~4.5 pages)

### 4.6.1 Isolated Classifier Evaluation (~1.0 pages)

**Definition:** Each classifier (FA and WD) is evaluated independently on the held-out test set using standard classification metrics: accuracy, precision (weighted and macro), recall (weighted and macro), and F1 score (weighted and macro).

**Formulas:**
- Accuracy:
  $$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$$
- Weighted precision:
  $$\text{Prec}_{\text{weighted}} = \sum_{c=1}^{K} \frac{N_c}{N} \cdot \frac{TP_c}{TP_c + FP_c}$$
- Weighted F1:
  $$F1_{\text{weighted}} = \sum_{c=1}^{K} \frac{N_c}{N} \cdot \frac{2TP_c}{2TP_c + FP_c + FN_c}$$

**Figure:** `fig:classifier-evaluation` — Diagram showing test set → classifier → predictions → metrics computation

**TRACE-specific context:** Implemented in \texttt{evaluate\_classifier()} (\texttt{evaluate\_model.py:118–135}). The function uses scikit-learn's \texttt{accuracy\_score}, \texttt{precision\_score}, \texttt{recall\_score}, and \texttt{f1\_score} with \texttt{average="weighted"} and \texttt{average="macro"} variants. The FA classifier has 6 classes; the WD classifier has 3 classes. Metrics are printed via \texttt{print\_metrics()} (lines 138–145).

---

### 4.6.2 Cross-Validation on Held-Out Test Set (~1.0 pages)

**Definition:** Cross-validation is run exclusively on the 20,000-row held-out test set (not the training data) using 3 folds, providing a proper variance estimate of test-set generalization without overlap with the training set.

**Formulas:**
- 3-fold CV on test set:
  $$\mathcal{D}_{\text{test}} = \bigcup_{k=1}^{3} \mathcal{D}_{\text{test}}^{(k)}, \quad |\mathcal{D}_{\text{test}}^{(k)}| \approx 6{,}667$$
- CV accuracy:
  $$\text{CV}_{\text{acc}} = \frac{1}{3} \sum_{k=1}^{3} \text{Acc}(\mathcal{D}_{\text{test}}^{(k)})$$
- Confidence interval:
  $$\text{CI} = \text{CV}_{\text{acc}} \pm 2 \cdot \sigma_{\text{CV}}$$

**Figure:** `fig:cv-on-test-set` — Diagram showing the test set partitioned into 3 folds, each used for validation while the other 2 train a fresh model

**TRACE-specific context:** Implemented at \texttt{evaluate\_model.py:434–454}. The FA CV uses \texttt{cross\_val\_score(clf\_fa, X\_te, yfa\_te, cv=3, scoring="accuracy", n\_jobs=-1)} (lines 435–437). The WD CV uses the FA-augmented feature matrix \texttt{X\_wd\_te} (lines 449–451). The original evaluator's bug (sampling 30\% with random\_state=42, causing overlap with training data) is documented as FIX 2.

---

### 4.6.3 Cascade Calibration Check (~0.75 pages)

**Definition:** The cascade calibration check compares the distribution of FA top-class probabilities on training data (as the WD classifier saw during training) versus test data (as it sees at inference), quantifying any distributional shift.

**Formulas:**
- Training FA probability distribution:
  $$\mu_{\text{tr}} = \text{mean}(\max \mathbf{p}_{\text{FA}}^{\text{tr}}), \quad \sigma_{\text{tr}} = \text{std}(\max \mathbf{p}_{\text{FA}}^{\text{tr}})$$
- Test FA probability distribution:
  $$\mu_{\text{te}} = \text{mean}(\max \mathbf{p}_{\text{FA}}^{\text{te}}), \quad \sigma_{\text{te}} = \text{std}(\max \mathbf{p}_{\text{FA}}^{\text{te}})$$
- Mean gap:
  $$\Delta_{\text{mean}} = |\mu_{\text{tr}} - \mu_{\text{te}}|$$
- Alert threshold:
  $$\text{alert} = \begin{cases} \text{true} & \text{if } \Delta_{\text{mean}} > 0.05 \\ \text{false} & \text{otherwise} \end{cases}$$

**Figure:** `fig:cascade-calibration` — Overlaid histograms of FA top-class probabilities for training vs. test data with mean markers and gap annotation

**TRACE-specific context:** Implemented in \texttt{check\_cascade\_calibration()} (\texttt{evaluate\_model.py:162–190}). The function computes \texttt{top\_conf\_tr} and \texttt{top\_conf\_te} at lines 171–173. The mean gap is computed at line 183. If the gap exceeds 0.05, a warning is printed recommending the use of \texttt{cross\_val\_predict} for OOF probabilities (lines 185–188).

---

### 4.6.4 End-to-End Pipeline Evaluation (~1.0 pages)

**Definition:** The end-to-end evaluation runs the complete \texttt{predict()} pipeline (Rule Engine → ML → Score Combination) on a sample of held-out test rows, comparing the system's output to ground truth labels.

**Formulas:**
- Pipeline accuracy:
  $$\text{Acc}_{\text{pipeline}} = \frac{1}{N_{\text{sample}}} \sum_{i=1}^{N_{\text{sample}}} \mathbb{I}(\text{predict}(x_i).\text{warranty\_decision} = y_i^{\text{WD}})$$
- Per-engine accuracy:
  $$\text{Acc}_{\text{engine } e} = \frac{1}{N_e} \sum_{i: \text{engine}_i = e} \mathbb{I}(\text{predict}(x_i).\text{warranty\_decision} = y_i^{\text{WD}})$$
- Engine breakdown:
  $$N_e = |\{i : \text{predict}(x_i).\text{decision\_engine} = e\}|$$

**Figure:** `fig:pipeline-evaluation` — Diagram showing test rows → predict() pipeline → output comparison → per-engine accuracy breakdown

**TRACE-specific context:** Implemented in \texttt{evaluate\_pipeline()} (\texttt{evaluate\_model.py:197–275}). The default sample size is 3 (line 211) to avoid long runtimes when LLM is enabled (~55s per prediction). The function tracks decision engine usage and computes per-engine accuracy (lines 262–273). The full classification report is printed for both FA and WD (lines 250–258).

---

### 4.6.5 Feature Importance Analysis (~0.75 pages)

**Definition:** Feature importance is computed from the XGBoost classifiers using Gini importance (total impurity reduction), identifying which input features contribute most to prediction accuracy.

**Formulas:**
- Gini importance for feature $j$ in tree $t$:
  $$\text{Imp}_j^{(t)} = \sum_{s \in \text{splits on } j} N_s \cdot \Delta G_s$$
- Ensemble importance:
  $$\text{Imp}_j = \frac{1}{T} \sum_{t=1}^{T} \text{Imp}_j^{(t)}$$
- WD cascade feature names:
  $$\text{FeatureNames}_{\text{WD}} = \text{FeatureNames}_{\text{base}} \cup \{\text{fa\_prob}_c \mid c \in \text{FA classes}\}$$

**Figure:** `fig:feature-importance` — Horizontal bar charts showing top 20 features for FA and WD classifiers

**TRACE-specific context:** Implemented at \texttt{evaluate\_model.py:399–417}. The FA feature names are constructed from all transformer outputs (lines 105–113). The WD feature names include the FA cascade probability features (line 411): \texttt{fa\_prob\_\{class\}} for each of the 6 FA classes. The top 20 features are printed for both classifiers.

---

## 4.7 System Architecture & Deployment (~4.0 pages)

### 4.7.1 FastAPI Backend Architecture (~1.0 pages)

**Definition:** The TRACE backend is a FastAPI application serving the prediction pipeline through RESTful endpoints, with Pydantic models for request/response validation, CORS middleware for cross-origin access, and structured logging.

**Formulas:**
- Request schema:
  $$\text{ClaimRequest} = \{\text{fault\_code}: \text{str}, \text{technician\_notes}: \text{str}, \text{voltage}: \text{float}\}$$
- Response schema:
  $$\text{ClaimResponse} = \{\text{status}: \text{str}, \text{failure\_analysis}: \text{str}, \text{warranty\_decision}: \text{str}, \text{confidence}: \text{float}, \text{reason}: \text{str}, \text{matched\_complaint}: \text{str}, \text{decision\_engine}: \text{str}\}$$
- Endpoint:
  $$\text{POST } /analyze: \text{ClaimRequest} \to \text{ClaimResponse}$$
- Health check:
  $$\text{GET } /: \{\text{"message": "TRACE Backend Running ✅", "version": "2.0"}\}$$

**Figure:** `fig:fastapi-architecture` — Diagram showing FastAPI app with middleware, endpoints, schemas, and ML predictor integration

**TRACE-specific context:** Implemented in \texttt{main.py} (136 lines). The app is created at line 31: \texttt{app = FastAPI(title="TRACE Backend API", version="2.0")}. CORS middleware at lines 34–40 allows all origins. The \texttt{/analyze} endpoint is defined at lines 112–136. An OCR endpoint \texttt{/scan-image-easyocr} is defined at lines 53–68 using EasyOCR for image-based DTC extraction. Environment variables are loaded via \texttt{python-dotenv} at line 29.

---

### 4.7.2 Frontend Architecture (~0.75 pages)

**Definition:** The TRACE frontend is a single-page HTML application that provides a user interface for submitting warranty claims and viewing analysis results, communicating with the backend API via fetch requests.

**Formulas:**
- API call:
  $$\text{fetch}(\text{"http://localhost:8000/analyze"}, \{\text{method: "POST"}, \text{headers: \{"Content-Type": "application/json"\}}, \text{body: JSON.stringify(request)}\})$$
- Response handling:
  $$\text{response.json()} \to \{\text{status}, \text{failure\_analysis}, \text{warranty\_decision}, \text{confidence}, \text{reason}, \text{matched\_complaint}, \text{decision\_engine}\}$$

**Figure:** `fig:frontend-architecture` — Diagram showing HTML/CSS/JS single-page app with form inputs, API call, and result display

**TRACE-specific context:** Implemented in \texttt{frontend/index.html} (654 lines). The frontend communicates with the backend at port 8000. The UI includes form inputs for fault code, technician notes, and voltage, with a results display showing the decision, confidence, and reasoning.

---

### 4.7.3 Docker Compose Deployment (~0.75 pages)

**Definition:** The TRACE system is deployed using Docker Compose with two services: the FastAPI backend and an nginx frontend, connected via a shared bridge network with health checks and dependency ordering.

**Formulas:**
- Service configuration:
  $$\begin{array}{lll} \text{Service} & \text{Port} & \text{Health Check} \\ \hline \text{backend} & 8000:8000 & \text{HTTP GET } / \text{ every 30s} \\ \text{frontend} & 3000:3000 & \text{depends on backend healthy} \end{array}$$
- Network:
  $$\text{trace\_net}: \text{bridge driver}$$
- Restart policy:
  $$\text{restart} = \text{"unless-stopped"}$$

**Figure:** `fig:docker-compose-architecture` — Diagram showing two containers (backend, frontend) on shared network with port mappings and health check flow

**TRACE-specific context:** Defined in \texttt{docker-compose.yml} (41 lines). The backend builds from \texttt{./backend} context (line 6). Environment variables are loaded from \texttt{./backend/.env} (line 8). The health check (lines 17–22) uses a Python one-liner to hit the health endpoint. The frontend waits for the backend to be healthy before starting (line 33). The shared network \texttt{trace\_net} is defined at lines 39–41.

---

### 4.7.4 Logging and Observability (~0.75 pages)

**Definition:** The TRACE system uses a centralized logging configuration with hierarchical logger namespacing, structured decision logging, and configurable log levels via environment variables.

**Formulas:**
- Log format:
  $$\text{format} = \text{"\%(asctime)s [\%(levelname)s] \%(name)s \%(filename)s:\%(funcName)s:\%(lineno)d - \%(message)s"}$$
- Date format:
  $$\text{datefmt} = \text{"\%Y-\%m-\%dT\%H:\%M:\%S"}$$
- Log level:
  $$\text{LOG\_LEVEL} = \text{os.getenv("LOG\_LEVEL", "INFO")}$$
- Logger hierarchy:
  $$\text{trace} \to \text{trace.ml\_predictor}, \text{trace.llm\_client}, \text{trace.api}$$

**Figure:** `fig:logging-architecture` — Diagram showing logger hierarchy with DecisionLogger wrapper and structured log output format

**TRACE-specific context:** Implemented in \texttt{logging\_config.py} (61 lines). The \texttt{setup\_logging()} function (lines 22–30) configures the root logger. The \texttt{get\_logger()} function (lines 33–35) returns named loggers. The \texttt{DecisionLogger} class (lines 38–60) provides structured logging methods: \texttt{log\_stage()}, \texttt{log\_decision()}, \texttt{log\_input()}, \texttt{log\_output()}. Loggers are used throughout: \texttt{trace.ml\_predictor} (line 87 of ml\_predictor.py), \texttt{trace.llm\_client} (line 25 of llm\_client.py), \texttt{trace.api} (line 27 of main.py).

---

### 4.7.5 Model Serialization and Auto-Training (~0.75 pages)

**Definition:** Trained models and fitted transformers are serialized to a pickle file for persistence. The system auto-trains on startup if the model file does not exist, ensuring the application is always operational.

**Formulas:**
- Model bundle:
  $$\text{bundle} = \{\text{clf\_fa}, \text{clf\_wd}, \text{le\_fa}, \text{le\_wd}, \text{ohe}, \text{tfidf\_d}, \text{ohe\_supplier}, \text{mileage\_scaler}, \text{year\_scaler}, \text{ohe\_mileage}, \text{claim\_age\_scaler}, \text{voltage\_scaler}, \text{ohe\_voltage\_bracket}, \text{ohe\_dtc\_count\_bracket}\}$$
- Serialization:
  $$\text{pickle.dump}(\text{bundle}, \text{open}(\text{MODEL\_PATH}, \text{"wb"}))$$
- Auto-training:
  $$\text{load\_models()} = \begin{cases} \text{pickle.load}(\text{MODEL\_PATH}) & \text{if file exists} \\ \text{train\_and\_save()} & \text{otherwise} \end{cases}$$

**Figure:** `fig:model-serialization` — Diagram showing training → bundle creation → pickle serialization → file storage → deserialization → inference

**TRACE-specific context:** Model path is defined at \texttt{ml\_predictor.py:93}: \texttt{MODEL\_PATH = os.path.join(BASE\_DIR, "trace\_models.pkl")}. The bundle contains 14 components (lines 439–447). Serialization at lines 448–449. The \texttt{load\_models()} function (lines 454–458) checks for the file and auto-trains if missing. The bundle is loaded lazily via the global \texttt{\_bundle} variable (line 461), initialized on first \texttt{predict()} call (lines 837–839).

---

## Figures Required

| # | Label | Description | Section |
|---|-------|-------------|---------|
| 1 | `fig:hybrid-decision-engine-overview` | High-level six-stage pipeline architecture diagram | 4.1.1 |
| 2 | `fig:six-stage-pipeline` | Detailed stage-by-stage flow with inputs, processing, outputs, and fallbacks | 4.1.2 |
| 3 | `fig:dataset-overview` | Dataset schema table with 11 columns, example values, and data types | 4.2.1 |
| 4 | `fig:fa-class-distribution` | Stacked bar chart of FA class proportions across years 2019–2025 with drift | 4.2.2 |
| 5 | `fig:voltage-distributions` | Overlaid KDE plots of voltage for all 6 FA classes with threshold markers | 4.2.3 |
| 6 | `fig:mileage-distributions` | Overlaid histograms of mileage for all 6 FA classes with bimodal connector inset | 4.2.4 |
| 7 | `fig:dtc-pool-architecture` | Native DTC pools per FA class with companion pair connections and cross-FA injection | 4.2.5 |
| 8 | `fig:wd-probabilities` | Stacked bar chart of warranty decision probabilities per FA class | 4.2.6 |
| 9 | `fig:label-noise-mechanisms` | Six noise injection mechanisms with target classes, flip directions, and rates | 4.2.7 |
| 10 | `fig:data-loading-pipeline` | CSV → null filling → label encoding → DTC extraction → split flow | 4.3.1 |
| 11 | `fig:train-test-split` | Split-before-fit diagram showing separate fit/transform paths | 4.3.2 |
| 12 | `fig:derived-features` | Raw columns → derived features with bin thresholds and interaction logic | 4.3.3 |
| 13 | `fig:transformer-pipeline` | 12 transformers fitted on train, applied to train and test, hstack assembly | 4.3.4 |
| 14 | `fig:dtc-feature-extraction` | DTC string parsing → prefix detection → high-value matching → feature vector | 4.3.5 |
| 15 | `fig:complaint-matching` | Keyword-first → fuzzy fallback → complaint label flow | 4.3.6 |
| 16 | `fig:xgboost-architecture` | Sequential tree building with learning rate, subsampling, column sampling | 4.4.1 |
| 17 | `fig:cascade-oof` | 5-fold OOF probability generation for training, direct FA prediction for inference | 4.4.2 |
| 18 | `fig:rule-engine-flow` | Sequential rule evaluation with first-match-wins logic (9 rules) | 4.4.3 |
| 19 | `fig:llm-integration` | Dual provider support, retry logic, three service endpoints | 4.4.4 |
| 20 | `fig:llm-categorization` | Notes + DTC → LLM prompt → 7-category output with confidence | 4.4.5 |
| 21 | `fig:score-combination` | Decision tree for all combination scenarios with weights and bonuses | 4.4.6 |
| 22 | `fig:stage1-llm-understanding` | Stage 1 flow: notes + DTC → LLM → structured output | 4.5.1 |
| 23 | `fig:stage2-rule-engine` | Stage 2 flow: inputs → sequential rule evaluation → first match | 4.5.2 |
| 24 | `fig:stage3-feature-translation` | Two-path diagram: LLM semantic extraction vs. deterministic parsing | 4.5.3 |
| 25 | `fig:stage4-xgboost-cascade` | Feature construction → FA classifier → probability vector → WD classifier | 4.5.4 |
| 26 | `fig:stage5-score-combination` | Rule + ML + LLM → agreement check → weighted blend → status | 4.5.5 |
| 27 | `fig:stage6-output-formatting` | LLM natural language vs. template assembly fallback | 4.5.6 |
| 28 | `fig:complete-predict-flow` | End-to-end six-stage pipeline with error handling and fallbacks | 4.5.7 |
| 29 | `fig:classifier-evaluation` | Test set → classifier → predictions → metrics computation | 4.6.1 |
| 30 | `fig:cv-on-test-set` | Test set partitioned into 3 folds for cross-validation | 4.6.2 |
| 31 | `fig:cascade-calibration` | Overlaid histograms of FA top-class probabilities: train vs. test | 4.6.3 |
| 32 | `fig:pipeline-evaluation` | Test rows → predict() pipeline → per-engine accuracy breakdown | 4.6.4 |
| 33 | `fig:feature-importance` | Horizontal bar charts: top 20 features for FA and WD classifiers | 4.6.5 |
| 34 | `fig:fastapi-architecture` | FastAPI app with middleware, endpoints, schemas, ML integration | 4.7.1 |
| 35 | `fig:frontend-architecture` | Single-page HTML app with form, API call, result display | 4.7.2 |
| 36 | `fig:docker-compose-architecture` | Two containers on shared network with port mappings and health checks | 4.7.3 |
| 37 | `fig:logging-architecture` | Logger hierarchy with DecisionLogger wrapper and structured output | 4.7.4 |
| 38 | `fig:model-serialization` | Training → bundle → pickle → file → deserialization → inference | 4.7.5 |

---

## Tables Required

| # | Title | Columns | Section |
|---|-------|---------|---------|
| 1 | Section Hierarchy and Page Estimates | Section, Subsections, Est. Pages | Header |
| 2 | Writing Style Rules | Rule, Specification | Document Guidelines |
| 3 | Dataset Schema | Column, Type, Example, Distribution | 4.2.1 |
| 4 | FA Class Base Proportions and Drift | Class, Base %, Drift/yr, 2025 % | 4.2.2 |
| 5 | Voltage Distribution Parameters | Class, μ, σ, [a, b], Validation | 4.2.3 |
| 6 | Mileage Distribution Parameters | Class, μ_log, σ_log, [min, max], Skewness Floor | 4.2.4 |
| 7 | Native DTC Pools by FA Class | FA Class, DTC Pool, Count, Examples | 4.2.5 |
| 8 | Companion DTC Pairs | Primary, Companion, Probability | 4.2.5 |
| 9 | Warranty Decision Probabilities | FA Class, P(PF), P(CF), P(ATS), Conditions | 4.2.6 |
| 10 | Label Noise Mechanisms | Target Class, Flip Direction, Rate, Condition | 4.2.7 |
| 11 | Derived Feature Specifications | Feature, Type, Bins/Thresholds, Rationale | 4.3.3 |
| 12 | Transformer Pipeline | Transformer, Type, Output Shape, Fit On | 4.3.4 |
| 13 | High-Value DTC Categories | Category, DTC Codes, Count | 4.3.5 |
| 14 | Complaint Keyword Map | Keyword, Complaint Label | 4.3.6 |
| 15 | XGBoost Hyperparameters | Parameter, Value, Rationale | 4.4.1 |
| 16 | Complete Rule Specification (9 rules) | #, Rule ID, Condition, FA, WD, Status, Confidence | 4.4.3 |
| 17 | LLM Category to Warranty Mapping | Category, Warranty Decision | 4.4.5 |
| 18 | Score Combination Constants | Constant, Value, Description | 4.4.6 |
| 19 | Score Combination Formulas | Scenario, Formula, Conditions | 4.4.6 |
| 20 | Confidence Thresholds | Threshold, Value, Outcome | 4.4.6 |
| 21 | LLM Prompt Services | Service, Function, Prompt Variable, Output Keys | 4.4.4 |
| 22 | Model Bundle Components | Component, Type, Purpose | 4.7.5 |
| 23 | API Endpoints | Method, Path, Request, Response | 4.7.1 |
| 24 | Docker Service Configuration | Service, Build Context, Port, Health Check | 4.7.3 |
| 25 | Logger Hierarchy | Logger Name, Module, Level | 4.7.4 |
| 26 | Evaluation Metrics Summary | Metric, FA Value, WD Value, Description | 4.6.1 |

---

## LaTeX-Specific Notes

### Equation Numbering

All equations must be numbered using the `equation` or `align` environments (NOT starred versions). Multi-line equations should use `align` with `&` alignment points.

### Code References

Code references use `\texttt{}` with escaped underscores:
- `\texttt{ml\_predictor.py:836}` for the `predict()` function
- `\texttt{generate\_dataset\_v9(1).py:206}` for the `truncated_normal()` function
- File paths: `\texttt{backend/ml\_predictor.py}`

### Mathematical Notation

- Use $\mathbb{I}(\cdot)$ for indicator functions
- Use $\mathcal{D}$ for datasets, $\mathcal{X}$ for input space
- Use $\mathbf{p}$ for probability vectors (bold)
- Use $\theta$ for model parameters
- Use $\tau$ for thresholds
- Use $w$ for weights, $c$ for confidence scores

### Table Formatting

Use `booktabs` for all tables:
```latex
\begin{table}[H]
\centering
\caption{Complete Rule Specification}
\label{tab:rules}
\begin{tabular}{lp{2.5cm}p{3.5cm}p{2.5cm}p{2cm}c}
\toprule
\textbf{\#} & \textbf{Rule ID} & \textbf{Condition} & \textbf{Warranty Decision} & \textbf{Status} & \textbf{Confidence} \\
\midrule
1 & \texttt{over\_voltage} & $V > 16.0$ & Customer Failure & Rejected & 93.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure Placement

Use `[H]` placement for all figures to ensure they appear exactly where specified. Use `0.85\textwidth` for standard figures and `0.95\textwidth` for wide tables.
