---
description: Run model evaluation on TRACE ML classifiers (Failure Analysis and Warranty Decision), outputs to thoughts/shared/evaluation_output.txt
---

# Evaluate Model

You are tasked with running the model evaluation script to assess the performance of the TRACE ML classifiers.

---

## Prerequisites

Before running, ensure:

1. **Working directory**: `backend/`
2. **Required files**:
   - `backend/ml_predictor.py` — contains model training logic and paths
   - `backend/synthetic_warranty_claims_v2.csv` — training dataset
   - `backend/trace_models.pkl` — trained model bundle (created if missing)

---

## Execution Steps

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

```bash
cd backend && python3 evaluate_model.py
```

### Step 3: Save Output

The script prints results to stdout. Capture and save to `thoughts/shared/evaluation_output.txt`:

```bash
cd backend && python3 evaluate_model.py 2>&1 > thoughts/shared/evaluation_output.txt
```

Or append to existing file:

```bash
cd backend && python3 evaluate_model.py >> thoughts/shared/evaluation_output.txt
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
| **Cross-Validation** | 3-fold CV on 30% sample |

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
