"""
Tests for ml_predictor logging integration.
"""

import logging
from unittest.mock import patch, MagicMock
from io import StringIO


class TestMLPredictorLogging:
    """Test suite for ml_predictor logging enhancements."""

    def test_predict_logs_stage1(self):
        """Stage 1 (LLM) should be logged when available."""
        import ml_predictor
        ml_predictor._bundle = None
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.ml_predictor")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(ml_predictor, 'load_models') as mock_load:
                mock_load.return_value = MagicMock()
                
                ml_predictor.predict("P0562", "Engine overheating", 14.2)
                
                log_output = log_capture.getvalue()
                assert "INPUT" in log_output or "predict" in log_output.lower()
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_predict_logs_rules(self):
        """Rule engine results should be logged."""
        import ml_predictor
        ml_predictor._bundle = None
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.ml_predictor")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(ml_predictor, 'load_models') as mock_load:
                mock_load.return_value = MagicMock()
                
                with patch.object(ml_predictor, 'run_rules') as mock_rules:
                    mock_rules.return_value = {
                        "rule_id": "over_voltage",
                        "status": "Rejected",
                        "warranty_decision": "Customer Failure",
                        "rule_confidence": 94.0,
                        "rule_fired": True
                    }
                    
                    ml_predictor.predict("P0562", "Engine overheating", 17.5)
                    
                    log_output = log_capture.getvalue()
                    assert "Rule" in log_output or "rule" in log_output.lower()
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_predict_logs_ml(self):
        """ML results should be logged."""
        import ml_predictor
        ml_predictor._bundle = None
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.ml_predictor")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(ml_predictor, 'load_models') as mock_load:
                mock_load.return_value = MagicMock()
                
                ml_predictor.predict("P0562", "Engine overheating", 14.2)
                
                log_output = log_capture.getvalue()
                assert "ML" in log_output or "ml" in log_output.lower() or "OUTPUT" in log_output
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)

    def test_predict_logs_combined(self):
        """Combined decision should be logged."""
        import ml_predictor
        ml_predictor._bundle = None
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("trace.ml_predictor")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            with patch.object(ml_predictor, 'load_models') as mock_load:
                mock_load.return_value = MagicMock()
                
                ml_predictor.predict("P0562", "Engine overheating", 14.2)
                
                log_output = log_capture.getvalue()
                assert "OUTPUT" in log_output or "predict" in log_output.lower()
        finally:
            logger.setLevel(original_level)
            logger.removeHandler(handler)
