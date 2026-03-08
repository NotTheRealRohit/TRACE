---
description: Run model evaluation on TRACE ML classifiers (Failure Analysis and Warranty Decision), outputs to thoughts/shared/evaluation_output.txt — NOTE: Disable LLM to avoid 55+ second delays per prediction
---

# Evaluate Model

You are tasked with running the model evaluation script to assess the performance of the TRACE ML classifiers.

---

## IMPORTANT: LLM Timeout Context

**Before running evaluation, you MUST disable LLM calls** — the OpenRouter API is very slow:

| Stage | Function | Typical Response Time |
|-------|----------|----------------------|
| Stage 1 | `understand_claim` (Semantic Understanding) | ~15 seconds |
| Stage 3 | `translate_to_ml_features` (Feature Translation) | ~18 seconds |
| Stage 6 | `format_output` (Output Formatting) | ~23 seconds |
| **Total** | **All 3 LLM calls per prediction** | **~55 seconds** |

The `evaluate_pipeline()` function runs `predict()` on **2000 samples**. With LLM enabled, this would take **2000 × 55s = ~30+ hours** — causing severe timeout errors.

---

## Execution Steps

### Step 0: Disable LLM (CRITICAL)

Before running evaluation, ensure LLM is disabled. The `predict()` function in `ml_predictor.py` checks:

```python
llm_available = api_key_available and len(notes) > 5
```

**Option A — Unset the API key** (recommended for evaluation):

```bash
cd backend
unset OPENROUTER_API_KEY
```

**Option B — Use a dummy key that won't work**:

```bash
export OPENROUTER_API_KEY=""
```

This forces the fallback code paths (rule-based + ML only, no LLM), reducing each prediction to **<1 second**.

### Step 1: Check if Model Exists

```bash
ls -la backend/trace_models.pkl
```

If the file does not exist, you must train the model first:

```bash
cd backend && python3 -c "from ml_predictor import train_and_save; train_and_save()"
```

This will:
- Load the training data from `synthetic_warranty_claims_v2.csv`
- Train two RandomForest classifiers:
  - **Failure Analysis** classifier (6 classes)
  - **Warranty Decision** classifier (3 classes, with FA cascade)
- Save the model bundle to `trace_models.pkl`

### Step 2: Run Evaluation

**CRITICAL**: Run with LLM disabled to avoid timeouts:

```bash
cd backend
unset OPENROUTER_API_KEY  # or export OPENROUTER_API_KEY=""
python3 evaluate_model.py
```

**Expected runtime**: ~2-5 minutes with LLM disabled (ML-only predictions).

### Step 3: Monitor Progress (View Pipeline Logs)

The evaluation outputs detailed logs. To watch progress in real-time:

```bash
# Run in background and capture logs
cd backend
unset OPENROUTER_API_KEY
python3 evaluate_model.py 2>&1 | tee evaluation_log.txt
```

**Expected runtime**: ~2-5 minutes with LLM disabled (3 pipeline samples + ML eval).

The logs show:
- Data leakage warnings
- ML classifier metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Feature importance
- Cross-validation results
- **End-to-end pipeline evaluation** (FIX 4) — shows decision engine breakdown:
  - `LLM+Rule+ML` — if LLM was used (slow!)
  - `Rule+ML` — rule fired, ML agreed
  - `ML` — no rule matched

### Step 4: Save Output

The script prints results to stdout. Capture and save to `<project-root>/thoughts/shared/evaluation_output.txt`:

```bash
cd backend && python3 evaluate_model.py 2>&1 > <project-root>/thoughts/shared/evaluation_output.txt
```

Or append to existing file:

```bash
cd backend && python3 evaluate_model.py >> <project-root>/thoughts/shared/evaluation_output.txt
```

---

## Expected Output

The evaluation produces:

| Metric | Description |
|--------|-------------|
| **Failure Analysis** | 6-class classification (ASIC, Connector, NTF, Sensor, Track, Controller) |
| **Warranty Decision** | 3-class classification (According to Spec, Customer Failure, Production Failure) |
| **Metrics** | Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1 (weighted/macro) |
| **Confusion Matrix** | Per-class TP/FP/FN |
| **Feature Importance** | Top 20 features for each classifier |
| **Cross-Validation** | 3-fold CV on held-out test set |

**NOTE**: The end-to-end pipeline evaluation (FIX 4) is limited to **3 samples** by default to avoid long runtimes. If LLM is disabled, you can increase this in the code.

---

## Key Files and Paths

| Path | Description |
|------|-------------|
| `backend/evaluate_model.py` | Evaluation script |
| `backend/ml_predictor.py` | Model training (`train_and_save()`) |
| `backend/trace_models.pkl` | Model bundle (clf_fa, clf_wd, encoders, scalers) |
| `backend/synthetic_warranty_claims_v2.csv` | Training data (12K rows) |
| `thoughts/shared/evaluation_output.txt` | Output file |

---

## Model Architecture

The trained models use:

- **Features**: Customer Complaint (OneHot), DTC text (TF-IDF), DTC flags, Voltage, Supplier (OneHot), Mileage, Year
- **Failure Analysis**: RandomForest (200 trees), predicts 6 failure types
- **Warranty Decision**: RandomForest (200 trees, class_weight=balanced), uses FA probabilities as cascade features

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: trace_models.pkl` | Run `python3 -c "from ml_predictor import train_and_save; train_and_save()"` |
| `ValueError: X has N features, expected M` | Model trained with different feature set — retrain by deleting pkl file |
| Import errors | Ensure running from `backend/` directory with correct PYTHONPATH |
| **Evaluation hangs / times out** | **UNSET OPENROUTER_API_KEY** before running — LLM adds ~55s per prediction |
| **Pipeline logs show LLM being called** | Check that `OPENROUTER_API_KEY` is not set in environment |

## Timeout and API Response Context

The `predict()` function has 6 stages, but only calls OpenRouter when:
1. `OPENROUTER_API_KEY` is set in environment
2. Technician notes have > 5 characters

**With LLM enabled** (DO NOT use for evaluation):
- Each prediction takes ~55 seconds
- 2000 samples = 30+ hours → **guaranteed timeout**

**With LLM disabled** (recommended):
- Predictions use rule-based + ML only
- Each prediction takes <1 second
- 2000 samples = ~3-5 minutes

**Pipeline logs to watch**:
- `[STAGE 1] LLM Understanding` — appears if LLM is enabled (slow!)
- `Rule fired: X with confidence Y` — rule-based decision
- `Decision: ML Model` — ML prediction
- `Decision: Combined` — final result

If you see `[STAGE 1]` logs, abort and re-run with `unset OPENROUTER_API_KEY`.

---

## Running from Root

If running from project root:

```bash
python3 -c "
import sys
sys.path.insert(0, 'backend')
from ml_predictor import train_and_save, load_models
train_and_save()
"

python3 -c "
import sys
sys.path.insert(0, 'backend')
from evaluate_model import main
main()
" > thoughts/shared/evaluation_output.txt
```

---

## Verification

After running, verify output was created:

```bash
ls -la thoughts/shared/evaluation_output.txt
head -30 thoughts/shared/evaluation_output.txt
```

Expected key metrics:
- Failure Analysis accuracy: ~98%
- Warranty Decision accuracy: ~90%
