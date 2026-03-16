# Voltage Field Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the voltage field from the FastAPI backend and HTML frontend since ml_predictor.py no longer uses this parameter.

**Architecture:** Remove voltage from API schema, remove voltage from frontend form, update logging - straightforward deletion task with validation tests.

**Tech Stack:** FastAPI (Python), Vanilla HTML/JavaScript

---

## Background

The research document `thoughts/shared/research/2026-03-16-voltage-removal-research.md` identified all voltage references in:
- `backend/main.py` - 3 locations (schema, logging, API call)
- `frontend/index.html` - 5 locations (form field, placeholder, JS extraction, validation, request body)

---

## Phase 1: Backend Changes (main.py)

### Task 1: Update ClaimRequest Schema

**Files:**
- Modify: `backend/main.py:38-50`

**Step 1: Write the failing test**

Create test to verify API accepts claims without voltage:
```bash
cd backend
python3 -c "
import requests
response = requests.post('http://localhost:8000/analyze', json={
    'fault_code': 'P0562',
    'technician_notes': 'Engine overheating'
})
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')
"
```
Expected: 422 Unprocessable Entity (voltage required)

**Step 2: Run test to confirm it fails**
```
Status: 422
```

**Step 3: Remove voltage from ClaimRequest schema**

Edit `backend/main.py:41` - remove the voltage field line:
```python
class ClaimRequest(BaseModel):
    fault_code:         str
    technician_notes:   str
    # voltage: float  # REMOVED
```

**Step 4: Run test to verify it passes**
```
Status: 200
```

---

### Task 2: Update ml_predict() call

**Files:**
- Modify: `backend/main.py:73-78`

**Step 1: Write the failing test**

```bash
python3 -c "
from main import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post('/analyze', json={
    'fault_code': 'P0562',
    'technician_notes': 'Engine overheating'
})
print(f'Status: {response.status_code}')
"
```
Expected: TypeError (unexpected keyword argument 'voltage')

**Step 2: Run test to confirm it fails**

**Step 3: Remove voltage parameter from ml_predict() call**

Edit `backend/main.py:73-78`:
```python
try:
    result = ml_predict(
        fault_code        = claim.fault_code,
        technician_notes  = claim.technician_notes,
    )
```

**Step 4: Run test to verify it passes**

---

### Task 3: Update logging

**Files:**
- Modify: `backend/main.py:68-69`

**Step 1: Write the failing test**

Verify logging works without voltage:
```bash
python3 -c "
import logging
logging.basicConfig(level=logging.INFO)
from main import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post('/analyze', json={
    'fault_code': 'P0562',
    'technician_notes': 'Engine overheating'
})
" 2>&1 | grep "REQUEST /analyze"
```
Expected: Should show voltage in log (old behavior)

**Step 2: Run test to confirm it fails**

**Step 3: Remove voltage from log statement**

Edit `backend/main.py:68-69`:
```python
logger.info("REQUEST /analyze | fault_code=%s",
            claim.fault_code)
```

**Step 4: Run test to verify it passes**

---

## Phase 2: Frontend Changes (index.html)

### Task 4: Remove voltage input field

**Files:**
- Modify: `frontend/index.html:446-453`

**Step 1: Write the failing test**

Verify frontend requires voltage (current behavior):
- Open frontend in browser
- Try to submit without voltage
Expected: Blocked by validation

**Step 2: Remove voltage HTML field**

Edit `frontend/index.html:446-453` - remove the entire voltage field div:
```html
<!-- REMOVE THIS BLOCK:
<div class="field">
  <label>Voltage Reading</label>
  <div class="voltage-row">
    <input type="number" id="voltage" placeholder="e.g. 14.2" step="0.1" min="0" max="30"/>
    <span class="voltage-unit">V</span>
  </div>
  <div class="hint">Normal range: 11–16 V · Over-voltage (&gt;16 V) triggers EOS rejection</div>
</div>
-->
```

**Step 3: Verify frontend works without voltage field**

---

### Task 5: Remove voltage from JavaScript

**Files:**
- Modify: `frontend/index.html:537-548`

**Step 1: Write the failing test**

Verify current code has voltage in request body:
```javascript
// Current: body: JSON.stringify({ fault_code, technician_notes, voltage })
```

**Step 2: Remove voltage variable and validation**

Edit lines around 537-548:
```javascript
async function analyzeClaim() {
  const fault_code       = document.getElementById("fault_code").value.trim();
  const technician_notes = document.getElementById("technician_notes").value.trim();

  if (!technician_notes && !fault_code) {
    flashInput(); return;
  }
  // REMOVED: voltage validation block
  
  // ... rest of function
  body: JSON.stringify({ fault_code, technician_notes }),  // REMOVED voltage
```

**Step 3: Verify submission works**

---

### Task 6: Update placeholder text

**Files:**
- Modify: `frontend/index.html:472`

**Step 1: Remove voltage from placeholder**

Edit `frontend/index.html:472`:
```html
<div id="result-placeholder">
  &gt; Awaiting claim data...<br>
  &gt; Load ECU fault codes<br>
  &gt; Enter technician observations<br>
  &gt; Run analysis
</div>
```

---

## Phase 3: Verification

### Task 7: End-to-end verification

**Step 1: Run backend tests**

```bash
cd backend
python3 -m pytest backend/tests/ -v
```

**Step 2: Start server and test API**

```bash
uvicorn main:app --reload --port 8000
# In another terminal:
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"fault_code": "P0562", "technician_notes": "Engine overheating"}'
```

Expected: Valid response without voltage

**Step 3: Verify frontend loads**

Open `frontend/index.html` in browser - voltage field should be gone, form should submit successfully.

---

## Summary of Changes

| File | Line(s) | Change |
|------|---------|--------|
| `backend/main.py` | 41 | Remove `voltage: float` from ClaimRequest |
| `backend/main.py` | 68-69 | Remove voltage from log statement |
| `backend/main.py` | 75 | Remove voltage from ml_predict() call |
| `frontend/index.html` | 446-453 | Remove voltage input HTML |
| `frontend/index.html` | 472 | Remove voltage from placeholder |
| `frontend/index.html` | 537 | Remove voltage variable |
| `frontend/index.html` | 543-548 | Remove voltage validation |
| `frontend/index.html` | 556 | Remove voltage from JSON body |

---

## Testing Commands

```bash
# Backend tests
cd backend && python3 -m pytest backend/tests/ -v

# API smoke test (no voltage)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"fault_code": "P0562", "technician_notes": "Engine overheating"}'

# Frontend - open in browser and verify voltage field gone
```

---

**Plan complete.**
