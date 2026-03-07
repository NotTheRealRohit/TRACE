"""
Tests for main.py logging integration.
"""

import logging
from unittest.mock import patch, MagicMock
from io import StringIO
from fastapi.testclient import TestClient


class TestMainLogging:
    """Test suite for main.py logging enhancements."""

    def test_analyze_endpoint_logs_request(self):
        """Request should be logged in analyze_claim endpoint."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.api")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch('backend.ml_predictor.predict') as mock_predict:
                mock_predict.return_value = {
                    "status": "Approved",
                    "failure_analysis": "Test",
                    "warranty_decision": "Production Failure",
                    "confidence": 85.0,
                    "reason": "Test reason",
                    "matched_complaint": "Engine overheating",
                    "decision_engine": "Rule+ML"
                }
                
                from backend.main import app
                client = TestClient(app)
                response = client.post("/analyze", json={
                    "fault_code": "P0562",
                    "technician_notes": "Engine overheating",
                    "voltage": 14.2
                })
                
                log_output = log_capture.getvalue()
                assert "REQUEST" in log_output or "/analyze" in log_output
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_analyze_endpoint_logs_response(self):
        """Response should be logged in analyze_claim endpoint."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.api")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch('backend.ml_predictor.predict') as mock_predict:
                mock_predict.return_value = {
                    "status": "Approved",
                    "failure_analysis": "Test",
                    "warranty_decision": "Production Failure",
                    "confidence": 85.0,
                    "reason": "Test reason",
                    "matched_complaint": "Engine overheating",
                    "decision_engine": "Rule+ML"
                }
                
                from backend.main import app
                client = TestClient(app)
                response = client.post("/analyze", json={
                    "fault_code": "P0562",
                    "technician_notes": "Engine overheating",
                    "voltage": 14.2
                })
                
                log_output = log_capture.getvalue()
                assert "RESPONSE" in log_output or "Approved" in log_output
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_analyze_endpoint_logs_error(self):
        """Error logging code should exist in analyze_claim endpoint."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        with open(os.path.join(os.path.dirname(__file__), '..', 'main.py'), 'r') as f:
            content = f.read()
        
        assert 'logger.error("ERROR' in content
        assert 'exc_info=True' in content
