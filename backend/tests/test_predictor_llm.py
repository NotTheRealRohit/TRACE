"""
Tests for LLM Integration in Predictor

These tests validate the integration of LLM client into ml_predictor.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import sys


class TestPredictorLLMIntegration:
    """Test suite for LLM integration in ml_predictor"""

    @patch('llm_client.requests.post')
    @patch('llm_client.get_api_key')
    def test_llm_called_before_rules(self, mock_key, mock_post):
        """LLM gets called when notes provided"""
        mock_key.return_value = "test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "moisture_damage",
                        "confidence": 0.85,
                        "failure_analysis": "Sensor short due to moisture",
                        "reasoning": "Water found in connector"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("P0562", "Water found in connector", 12.5)
        
        assert result["decision_engine"] == "LLM"

    @patch('llm_client.categorize_notes')
    def test_fallback_to_rules_on_llm_error(self, mock_categorize):
        """Falls back to keyword rules when LLM fails"""
        mock_categorize.side_effect = RuntimeError("API error")
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("P0562", "Engine overheating", 14.2)
        
        assert result["decision_engine"] == "Rule-based"

    @patch('llm_client.requests.post')
    @patch('llm_client.get_api_key')
    def test_decision_engine_label_llm(self, mock_key, mock_post):
        """Returns 'LLM' in decision_engine field"""
        mock_key.return_value = "test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "ntf",
                        "confidence": 0.90,
                        "failure_analysis": "NTF",
                        "reasoning": "No fault found"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("P0562", "No fault found", 12.5)
        
        assert result["decision_engine"] == "LLM"

    @patch('llm_client.categorize_notes')
    def test_empty_notes_skips_llm(self, mock_categorize):
        """Skips LLM call when notes empty"""
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("P0562", "", 12.5)
        
        mock_categorize.assert_not_called()
        assert result["decision_engine"] in ["Rule-based", "ML model"]

    @patch('llm_client.categorize_notes')
    def test_short_notes_skips_llm(self, mock_categorize):
        """Skips LLM call when notes too short (< 5 chars)"""
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("P0562", "test", 12.5)
        
        mock_categorize.assert_not_called()
        assert result["decision_engine"] in ["Rule-based", "ML model"]
