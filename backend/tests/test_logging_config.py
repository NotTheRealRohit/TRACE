"""
Tests for logging_config module.
"""

import logging
import pytest
from io import StringIO


class TestLoggingConfig:
    """Test suite for logging configuration module."""

    def test_format_includes_filename(self):
        """Log output should contain filename."""
        from logging_config import TRACE_FORMAT
        assert "%(filename)s" in TRACE_FORMAT

    def test_format_includes_method(self):
        """Log output should contain function name."""
        from logging_config import TRACE_FORMAT
        assert "%(funcName)s" in TRACE_FORMAT

    def test_format_includes_line(self):
        """Log output should contain line number."""
        from logging_config import TRACE_FORMAT
        assert "%(lineno)d" in TRACE_FORMAT

    def test_get_logger_returns_logger(self):
        """get_logger() should return a valid logger instance."""
        from logging_config import get_logger
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_decision_logger_log_decision_includes_values(self):
        """Decision logs should include input values in the message."""
        from logging_config import get_logger, DecisionLogger
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        test_logger = logging.getLogger("test_decision")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        decision_logger = DecisionLogger(test_logger)
        
        result = {"status": "Approved", "confidence": 85.0}
        decision_logger.log_decision("test_decision", result, extra="context")
        
        log_output = log_capture.getvalue()
        assert "test_decision" in log_output
        assert "Approved" in log_output
        assert "85.0" in log_output

    def test_decision_logger_log_stage_includes_params(self):
        """Stage logs should include parameters."""
        from logging_config import get_logger, DecisionLogger
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        test_logger = logging.getLogger("test_stage")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        decision_logger = DecisionLogger(test_logger)
        decision_logger.log_stage(1, "LLM Understanding", fault_code="P0562", voltage=14.2)
        
        log_output = log_capture.getvalue()
        assert "[STAGE 1]" in log_output
        assert "LLM Understanding" in log_output
        assert "fault_code=P0562" in log_output
        assert "voltage=14.2" in log_output
