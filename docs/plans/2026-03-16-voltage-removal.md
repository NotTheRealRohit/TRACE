# Voltage Field Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove voltage input field from both backend API and frontend, since ml_predictor.py no longer uses this parameter.

**Architecture:** Remove voltage from API schema (backend), remove voltage input field and validation (frontend), update all related CSS and JavaScript. The ml_predict() function already accepts only fault_code and technician_notes.

**Tech Stack:** FastAPI (backend), Vanilla HTML/JS (frontend), pytest (testing)

---

## Pre-Flight Check

Before starting, verify current state:

```bash
# Confirm ml_predict no longer accepts voltage
grep -n "def predict" backend/ml_predictor.py
# Expected: def predict(fault_code: str, technician_notes: str) -> dict:

# Check current test suite works
cd backend && python3 -m pytest tests/ -v --tb=short
```

---

## Task 1: Write Failing API Test (No Voltage)

**Files:**
- Modify: `backend/tests/test_e2e.py`

**Step 1: Add failing test for API without voltage**

Add this test to `backend/tests/test_e2e.py` (after existing tests):

```python
    def test_analyze_endpoint_accepts_claim_without_voltage(self):
        """API /analyze endpoint should accept claims without voltage field"""
        import sys
        import os
        
        # Set API key for test
        with open('.env') as f:
            content = f.read()
        import re
        match = re.search(r'OPENROUTER_API_KEY=(\S+)', content)
        api_key = match.group(1).strip('"')
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        # Reload modules to pick up fresh environment
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        from ml_predictor import predict
        
        # This should work without voltage parameter
        result = predict("P0562", "Engine overheating")
        
        required_keys = ["status", "failure_analysis", "warranty_decision", 
                        "confidence", "reason", "matched_complaint", "decision_engine"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
```

**Step 2: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_analyze_endpoint_accepts_claim_without_voltage -v
```

**Step 3: Commit**

```bash
git add backend/tests/test_e2e.py
git commit -m "test: add test for predict() without voltage parameter"
```

---

## Task 2: Remove Voltage from Backend API Schema

**Files:**
- Modify: `backend/main.py:38-50`

**Step 1: Write failing test - API rejects voltage**

Add this test to `backend/tests/test_e2e.py`:

```python
    def test_api_claim_request_schema_no_voltage(self):
        """ClaimRequest should NOT have voltage field"""
        from main import ClaimRequest
        
        # Verify voltage is NOT in the schema
        fields = ClaimRequest.model_fields
        assert 'voltage' not in fields, "voltage field should be removed from ClaimRequest"
        
        # Verify required fields still exist
        assert 'fault_code' in fields
        assert 'technician_notes' in fields
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_api_claim_request_schema_no_voltage -v
# Expected: FAIL - voltage still in ClaimRequest
```

**Step 3: Remove voltage from ClaimRequest schema**

Edit `backend/main.py` lines 38-41:

```python
class ClaimRequest(BaseModel):
    fault_code:         str
    technician_notes:    str
    # voltage removed - no longer used by ml_predictor
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_api_claim_request_schema_no_voltage -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add backend/main.py
git commit -m "refactor: remove voltage from ClaimRequest schema"
```

---

## Task 3: Remove Voltage from API Logging and ml_predict Call

**Files:**
- Modify: `backend/main.py:61-84`

**Step 1: Write failing test - API logs without voltage**

Add this test to `backend/tests/test_e2e.py`:

```python
    def test_api_endpoint_logs_without_voltage(self):
        """API /analyze endpoint should not log or pass voltage"""
        import inspect
        from main import analyze_claim
        
        # Get the source code of analyze_claim
        source = inspect.getsource(analyze_claim)
        
        # Verify voltage is not in the logging or ml_predict call
        assert 'voltage' not in source, "voltage should not appear in analyze_claim"
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_api_endpoint_logs_without_voltage -v
# Expected: FAIL - voltage still in logging and ml_predict call
```

**Step 3: Update analyze_claim function**

Edit `backend/main.py` lines 61-84:

```python
@app.post("/analyze", response_model=ClaimResponse)
def analyze_claim(claim: ClaimRequest):
    """
    Accepts warranty claim inputs from the TRACE frontend,
    routes them through the ML predictor (hybrid rule + RandomForest),
    and returns a structured warranty decision.
    """
    logger.info("REQUEST /analyze | fault_code=%s",
                claim.fault_code)
    
    try:
        result = ml_predict(
            fault_code        = claim.fault_code,
            technician_notes  = claim.technician_notes,
        )
        logger.info("RESPONSE /analyze | status=%s confidence=%.1f engine=%s",
                    result["status"], result["confidence"], 
                    result.get("decision_engine", "unknown"))
        return ClaimResponse(**result)
    except Exception as e:
        logger.error("ERROR /analyze | %s: %s", type(e).__name__, str(e),
                     exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_api_endpoint_logs_without_voltage -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add backend/main.py
git commit -m "refactor: remove voltage from API logging and ml_predict call"
```

---

## Task 4: Remove Voltage Input Field from Frontend HTML

**Files:**
- Modify: `frontend/index.html:446-453`

**Step 1: Write failing test - frontend has no voltage field**

Add this test to `backend/tests/test_e2e.py`:

```python
    def test_frontend_has_no_voltage_input(self):
        """Frontend should not have voltage input field"""
        import os
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'index.html')
        with open(frontend_path, 'r') as f:
            content = f.read()
        
        # Verify voltage input is removed
        assert 'id="voltage"' not in content, "voltage input should be removed"
        assert 'Voltage Reading' not in content, "Voltage label should be removed"
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_has_no_voltage_input -v
# Expected: FAIL - voltage field still exists
```

**Step 3: Remove voltage input HTML**

Edit `frontend/index.html` lines 446-453:

```html
    <!-- Voltage field removed - no longer needed for prediction -->

    <div class="field">
      <label>Technician Notes</label>
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_has_no_voltage_input -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "refactor: remove voltage input field from frontend"
```

---

## Task 5: Remove Voltage Placeholder Text

**Files:**
- Modify: `frontend/index.html:470-476`

**Step 1: Write failing test**

```python
    def test_frontend_placeholder_no_voltage(self):
        """Frontend placeholder should not mention voltage"""
        import os
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'index.html')
        with open(frontend_path, 'r') as f:
            content = f.read()
        
        assert 'Input voltage reading' not in content, "voltage placeholder should be removed"
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_placeholder_no_voltage -v
# Expected: FAIL
```

**Step 3: Remove voltage placeholder**

Edit `frontend/index.html` lines 470-476:

```html
    <div id="result-placeholder">
      &gt; Awaiting claim data...<br>
      &gt; Load ECU fault codes<br>
      &gt; Enter technician observations<br>
      &gt; Run analysis
    </div>
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_placeholder_no_voltage -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "refactor: remove voltage placeholder text"
```

---

## Task 6: Remove Voltage from Frontend JavaScript

**Files:**
- Modify: `frontend/index.html:535-572`

**Step 1: Write failing test**

```python
    def test_frontend_js_no_voltage(self):
        """Frontend JavaScript should not reference voltage"""
        import os
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'index.html')
        with open(frontend_path, 'r') as f:
            content = f.read()
        
        # Check JavaScript section (between <script> and </script>)
        import re
        script_match = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
        if script_match:
            js_content = script_match.group(1)
            assert 'voltage' not in js_content.lower(), "voltage should not be in JavaScript"
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_js_no_voltage -v
# Expected: FAIL
```

**Step 3: Update JavaScript to remove voltage**

Edit `frontend/index.html` lines 535-556:

```javascript
  async function analyzeClaim() {
    const fault_code       = document.getElementById("fault_code").value.trim();
    const technician_notes = document.getElementById("technician_notes").value.trim();

    if (!technician_notes && !fault_code) {
      flashInput(); return;
    }

    setBusy(true);

    try {
      const resp = await fetch(API_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ fault_code, technician_notes }),
      });
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_js_no_voltage -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "refactor: remove voltage from frontend JavaScript"
```

---

## Task 7: Remove Unused CSS for Voltage Row

**Files:**
- Modify: `frontend/index.html:198-206`

**Step 1: Write failing test**

```python
    def test_frontend_css_no_voltage(self):
        """Frontend CSS should not have voltage-related styles"""
        import os
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'frontend', 'index.html')
        with open(frontend_path, 'r') as f:
            content = f.read()
        
        # Check style section
        style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
        if style_match:
            css_content = style_match.group(1)
            assert '.voltage-row' not in css_content, "voltage-row CSS should be removed"
```

**Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_css_no_voltage -v
# Expected: FAIL
```

**Step 3: Remove voltage CSS**

Edit `frontend/index.html` lines 198-206:

```css
    /* Voltage row styles removed */

    /* ── Analyze button ── */
```

**Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_e2e.py::TestE2EIntegration::test_frontend_css_no_voltage -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "refactor: remove voltage CSS from frontend"
```

---

## Task 8: Full Integration Test

**Files:**
- Test: End-to-end flow

**Step 1: Run all voltage-related tests**

```bash
cd backend && python3 -m pytest tests/test_e2e.py -v -k "voltage"
# Expected: All PASS
```

**Step 2: Run full test suite**

```bash
cd backend && python3 -m pytest tests/ -v --tb=short
# Expected: All tests PASS
```

**Step 3: Manual verification - API**

```bash
# Test API without voltage
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"fault_code": "P0562", "technician_notes": "Engine overheating"}'
# Expected: Valid JSON response with warranty decision
```

**Step 4: Manual verification - Frontend**

Open `frontend/index.html` in browser:
- Verify voltage input field is gone
- Verify form submits with just fault_code and technician_notes
- Verify result displays correctly

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: complete voltage removal from API and frontend"
```

---

## Task 9: Lint and Typecheck

**Files:**
- Lint: All modified files

**Step 1: Run linting**

```bash
cd backend && python3 -m ruff check main.py ml_predictor.py
# Expected: No errors
```

**Step 2: Run formatting**

```bash
cd backend && python3 -m black main.py --diff
# Review changes
```

**Step 3: Final test run**

```bash
cd backend && python3 -m pytest tests/ -v --tb=short
# Expected: All tests PASS
```

---

## Success Criteria

### Automated Verification:
- [ ] All TDD tests written before implementation (RED confirmed)
- [ ] `pytest tests/test_e2e.py -k "voltage"` all pass
- [ ] `pytest tests/` full suite passes
- [ ] No lint errors: `ruff check backend/`

### Manual Verification:
- [ ] API accepts claim without voltage: `curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"fault_code": "P0562", "technician_notes": "Engine overheating"}'`
- [ ] Frontend displays without voltage input field
- [ ] Frontend form submits and displays result correctly

---

## Rollback Plan

If issues occur:

```bash
# Revert all changes
git revert HEAD
# Or revert specific file
git checkout HEAD~1 -- backend/main.py frontend/index.html
```

---

## References

- Research: `thoughts/shared/research/2026-03-16-voltage-removal-research.md`
- Original ml_predictor.py: `backend/ml_predictor.py:654`
- API endpoint: `backend/main.py:61`
