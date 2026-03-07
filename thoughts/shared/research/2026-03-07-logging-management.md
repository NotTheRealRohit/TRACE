---
date: 2026-03-07T14:30:00+00:00
researcher: opencode
git_commit: c0fae4a26ba751d5730b4db13429197f29008d77
branch: feature/logging
repository: capProj-2
topic: "How logging is managed, which files have enabled logging, what LOG levels are allowed"
tags: [research, logging, backend, configuration]
status: complete
last_updated: 2026-03-07
last_updated_by: opencode
---

# Research: Logging Management in TRACE Project

**Date**: 2026-03-07T14:30:00+00:00
**Researcher**: opencode
**Git Commit**: c0fae4a26ba751d5730b4db13429197f29008d77
**Branch**: feature/logging
**Repository**: capProj-2

## Research Question
How is logging managed currently, which files have enabled logging, what are the LOG levels allowed.

## Summary
Logging in the TRACE project is minimally implemented. Only `llm_client.py` uses Python's standard `logging` module. The rest of the backend (`main.py`, `ml_predictor.py`) uses `print()` statements for output. The LOG_LEVEL environment variable supports all standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL), defaulting to INFO in development and DEBUG in Docker.

## Detailed Findings

### Logging Configuration

The logging configuration is centralized in `backend/llm_client.py` (lines 18-27):

```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("trace.llm_client")
```

Key details:
- **Environment Variable**: `LOG_LEVEL` (defaults to "INFO")
- **Logger Name**: `"trace.llm_client"`
- **Format**: `%(asctime)s [%(levelname)s] %(name)s - %(message)s`
- **Date Format**: `%Y-%m-%dT%H:%M:%S`

### Files with Python Logging Enabled

| File | Lines | Logging Used |
|------|-------|--------------|
| `backend/llm_client.py` | 20-27, 34, 83-534 | `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()` |
| `backend/backup/llm_client.py` | 20-27, 34, 83-207 | Same as above (backup copy) |

### Files Using print() Instead of Logging

| File | Purpose | Output Method |
|------|---------|---------------|
| `backend/main.py` | FastAPI endpoints | No logging found |
| `backend/ml_predictor.py` | ML prediction engine | `print()` statements |
| `backend/ml_predictor_DecisionTree.py` | Alternative ML model | `print()` statements |

### LOG Levels Allowed

All standard Python logging levels are supported:

| Level | Description | Default |
|-------|-------------|---------|
| `DEBUG` | Detailed diagnostic information | Docker: Yes |
| `INFO` | General runtime events | Development: Yes (default) |
| `WARNING` | Unexpected events that don't prevent operation | - |
| `ERROR` | Serious problems that prevented function execution | - |
| `CRITICAL` | Very serious errors that may cause the program to stop | - |

### Environment Configuration

- **Development**: Uses default `INFO` level (or set manually via `export LOG_LEVEL=INFO`)
- **Docker**: Set to `DEBUG` in `docker-compose.yml` (line 10): `LOG_LEVEL=DEBUG`

## Code References

- `backend/llm_client.py:20` - LOG_LEVEL environment variable read
- `backend/llm_client.py:22-26` - logging.basicConfig() configuration
- `backend/llm_client.py:27` - Logger instance creation
- `backend/llm_client.py:34` - Example: `logger.error()` usage
- `backend/llm_client.py:83` - Example: `logger.info()` usage
- `backend/llm_client.py:89` - Example: `logger.debug()` usage
- `docker-compose.yml:10` - Docker LOG_LEVEL=DEBUG setting
- `AGENTS.md:74` - Documentation of LOG_LEVEL environment variable

## Architecture Insights

1. **Inconsistent Logging**: The project uses two different logging approaches:
   - Python `logging` module in `llm_client.py` (proper)
   - `print()` statements in `main.py` and `ml_predictor.py` (not recommended for production)

2. **No Centralized Logging**: There's no unified logging configuration across all backend files.

3. **No File Handlers**: Logging currently goes to stdout only (via `basicConfig`). No file rotation or persistence.

4. **Docker Debug Mode**: The docker-compose.yml sets DEBUG level, suggesting more verbose logging in containerized environments.

## Recommendations

1. Consider adding Python logging to `main.py` and `ml_predictor.py` for consistency
2. Add file handlers for log persistence
3. Consider using a logging configuration file (logging.conf) for centralized management

## Open Questions

- Should logging be added to `main.py` for API request/response tracking?
- Is there a need for structured logging (JSON format) for container orchestration?
- Should log rotation be implemented for production deployments?
