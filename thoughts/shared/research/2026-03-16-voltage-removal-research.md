---
date: 2026-03-16T00:00:00+00:00
researcher: opencode
git_commit: c1f2494b85f3d18cdfde6c254dce6570694cbd07
branch: feature/remove-voltage-dependency
repository: capProj-2
topic: "Voltage field removal from TRACE API and frontend"
tags: [research, backend, frontend, api, voltage-removal]
status: complete
last_updated: 2026-03-16
last_updated_by: opencode
---

# Research: Voltage Field Removal from TRACE API and Frontend

**Date**: 2026-03-16T00:00:00+00:00
**Researcher**: opencode
**Git Commit**: c1f2494b85f3d18cdfde6c254dce6570694cbd07
**Branch**: feature/remove-voltage-dependency
**Repository**: capProj-2

## Research Question

The user updated `backend/ml_predictor.py` to remove voltage as an input. This research identifies all locations in `backend/main.py` and `frontend/index.html` where voltage is used, to enable complete removal of the voltage dependency.

## Summary

The voltage field is used in two files beyond `ml_predictor.py`:
- **backend/main.py**: Requires voltage in API request, logs it, and passes to ML predictor
- **frontend/index.html**: Has a voltage input field, validates it is present, and sends it in API requests

All voltage references can be safely removed from both files since `ml_predictor.py` no longer uses this parameter.

## Detailed Findings

### backend/main.py

| Line | Type | Description |
|------|------|-------------|
| 41 | Schema | `voltage: float` - Required field in `ClaimRequest` Pydantic model |
| 68-69 | Logging | `logger.info("REQUEST /analyze \| fault_code=%s voltage=%s", ...)` - Logs voltage with request |
| 75 | API Call | Passes `voltage=claim.voltage` to `ml_predict()` function |

**Key observation**: The `ClaimRequest` model currently requires voltage as a mandatory float field. The ml_predict call passes this value but the function signature has been updated to not accept it.

### frontend/index.html

| Line | Type | Description |
|------|------|-------------|
| 446-453 | HTML | Voltage input field with label, placeholder "e.g. 14.2", unit "V", and hint about normal range |
| 472 | Placeholder | Text "> Input voltage reading" in result panel placeholder |
| 537 | JavaScript | `const voltage = parseFloat(document.getElementById("voltage").value);` - Reads input value |
| 543-548 | Validation | Blocks submission if `isNaN(voltage)` - requires voltage to be present and valid |
| 556 | API Request | `body: JSON.stringify({ fault_code, technician_notes, voltage })` - Sends voltage in request |

**Key observation**: The frontend currently requires voltage input (validation at lines 543-548) and will fail to submit claims without it. The voltage field must be removed entirely along with all validation logic.

## Code References

### backend/main.py:38-50
```python
class ClaimRequest(BaseModel):
    fault_code:         str
    technician_notes:   str
    voltage:            float  # REMOVE THIS LINE
```

### backend/main.py:68-76
```python
logger.info("REQUEST /analyze | fault_code=%s voltage=%s",
            claim.fault_code, claim.voltage)  # REMOVE voltage from log

try:
    result = ml_predict(
        fault_code        = claim.fault_code,
        technician_notes  = claim.technician_notes,
        voltage           = claim.voltage,  # REMOVE THIS LINE
    )
```

### frontend/index.html:446-453
```html
<div class="field">
  <label>Voltage Reading</label>
  <div class="voltage-row">
    <input type="number" id="voltage" placeholder="e.g. 14.2" step="0.1" min="0" max="30"/>
    <span class="voltage-unit">V</span>
  </div>
  <div class="hint">Normal range: 11–16 V · Over-voltage (&gt;16 V) triggers EOS rejection</div>
</div>
<!-- REMOVE THIS ENTIRE BLOCK -->
```

### frontend/index.html:535-548
```javascript
async function analyzeClaim() {
  const fault_code       = document.getElementById("fault_code").value.trim();
  const voltage          = parseFloat(document.getElementById("voltage").value);  // REMOVE
  const technician_notes = document.getElementById("technician_notes").value.trim();

  if (!technician_notes && !fault_code) {
    flashInput(); return;
  }
  if (isNaN(voltage)) {  // REMOVE THIS VALIDATION BLOCK
    document.getElementById("voltage").focus();
    document.getElementById("voltage").style.borderColor = "var(--amber)";
    setTimeout(() => document.getElementById("voltage").style.borderColor = "", 1500);
    return;
  }
  // ... rest of function - remove voltage from JSON body
```

### frontend/index.html:556
```javascript
body: JSON.stringify({ fault_code, technician_notes, voltage }),  // REMOVE voltage
// Should become: body: JSON.stringify({ fault_code, technician_notes }),
```

### frontend/index.html:472
```html
<div id="result-placeholder">
  &gt; Awaiting claim data...<br>
  &gt; Load ECU fault codes<br>
  &gt; Input voltage reading<br>  <!-- REMOVE THIS LINE -->
  &gt; Enter technician observations<br>
  &gt; Run analysis
</div>
```

## Architecture Insights

1. **API Contract Change**: Removing voltage requires updates to both the API schema (backend/main.py) and the frontend consumer (frontend/index.html)

2. **Validation Removal**: The frontend currently enforces voltage as mandatory - this validation must be removed to allow claims without voltage

3. **Logging Impact**: Backend logging references voltage and should be updated to remove this field from request logs

4. **ML Predictor Independence**: The ml_predict() function has already been updated to not accept voltage, making this cleanup straightforward

## Related Research

No prior research documents found for this topic.

## Open Questions

1. Should the voltage field be保留ed in the database/dataset for historical purposes even if not used for prediction?
2. Are there any other consumers of the /analyze endpoint that might be affected by this API change?
3. Should a version bump be considered for this breaking API change?
