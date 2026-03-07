"""
Tests for llm_client logging integration.
"""

import logging
from unittest.mock import patch, MagicMock
from io import StringIO


class TestLLMClientLogging:
    """Test suite for llm_client logging enhancements."""

    def test_categorize_notes_logs_stage(self):
        """Stage 1 logging should be present in categorize_notes."""
        import llm_client
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.llm_client")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(llm_client, 'get_api_key', return_value="test-key"):
                with patch('requests.post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": '{"category": "electrical_issue", "confidence": 0.85, "failure_analysis": "test", "reasoning": "test"}'
                            }
                        }]
                    }
                    mock_post.return_value = mock_response
                    
                    llm_client.categorize_notes("Engine overheating", "P0562", 14.2)
                    
                    log_output = log_capture.getvalue()
                    assert "Categorizing" in log_output or "categorize" in log_output.lower()
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_translate_logs_stage(self):
        """Stage 3 logging should be present in translate_to_ml_features."""
        import llm_client
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.llm_client")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(llm_client, 'get_api_key', return_value="test-key"):
                with patch('requests.post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": '{"customer_complaint": "Engine overheating", "dtc_codes": ["P0562"], "dtc_text": "P0562", "dtc_count": 1, "voltage": 14.2, "has_P": 1, "has_U": 0, "has_C": 0, "has_B": 0}'
                            }
                        }]
                    }
                    mock_post.return_value = mock_response
                    
                    llm_client.translate_to_ml_features("Engine overheating", "P0562", 14.2, "electrical_issue")
                    
                    log_output = log_capture.getvalue()
                    assert "STAGE 3" in log_output or "Feature Translation" in log_output
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_format_logs_decision(self):
        """Decision output should be logged in format_output."""
        import llm_client
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.llm_client")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(llm_client, 'get_api_key', return_value="test-key"):
                with patch('requests.post') as mock_post:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": '{"status": "Approved", "failure_analysis": "test", "warranty_decision": "Production Failure", "confidence": 85.0, "reason": "test", "matched_complaint": "Engine overheating", "decision_engine": "Rule+ML"}'
                            }
                        }]
                    }
                    mock_post.return_value = mock_response
                    
                    combined = {"decision_engine": "Rule+ML", "status": "Approved"}
                    features = {"customer_complaint": "Engine overheating"}
                    llm_client.format_output(combined, features)
                    
                    log_output = log_capture.getvalue()
                    assert "Formatting" in log_output or "output" in log_output.lower()
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)
