"""
End-to-End Tests for LLM Integration

These tests validate the full pipeline with LLM integration.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import sys


class TestE2EIntegration:
    """End-to-end integration tests for LLM pipeline"""

    @patch('llm_client.requests.post')
    @patch('llm_client.get_api_key')
    def test_full_pipeline_llm(self, mock_key, mock_post):
        """Full predict() with LLM returns valid response"""
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
        assert result["status"] == "Rejected"
        assert result["confidence"] == 85.0

    @patch('llm_client.categorize_notes')
    def test_fallback_chain(self, mock_categorize):
        """LLM → Rules → ML fallback works"""
        mock_categorize.side_effect = RuntimeError("API error")
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        
        result = predict("P0562", "Engine overheating", 14.2)
        
        assert result["decision_engine"] in ["Rule-based", "ML model"]

    def test_response_schema(self):
        """Response matches ClaimResponse schema"""
        import re
        with open('.env') as f:
            content = f.read()
        match = re.search(r'OPENROUTER_API_KEY=(\S+)', content)
        api_key = match.group(1).strip('"')
        
        import os
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        from ml_predictor import predict
        result = predict("P0562", "Engine overheating, low idle", 14.2)
        
        required_keys = ["status", "failure_analysis", "warranty_decision", 
                        "confidence", "reason", "matched_complaint", "decision_engine"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 100

    @patch('llm_client.requests.post')
    @patch('llm_client.get_api_key')
    def test_llm_categorizes_ntf(self, mock_key, mock_post):
        """LLM correctly categorizes NTF (No Trouble Found)"""
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
        result = predict("P0562", "No fault found, intermittent", 12.5)
        
        assert result["decision_engine"] == "LLM"
        assert result["status"] == "Approved"
        assert result["warranty_decision"] == "According to Specification"

    @patch('llm_client.requests.post')
    @patch('llm_client.get_api_key')
    def test_llm_categorizes_communication_fault(self, mock_key, mock_post):
        """LLM correctly categorizes communication faults"""
        mock_key.return_value = "test-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "communication_fault",
                        "confidence": 0.88,
                        "failure_analysis": "CAN bus communication error",
                        "reasoning": "U-code indicates communication fault"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        if 'ml_predictor' in sys.modules:
            del sys.modules['ml_predictor']
        
        from ml_predictor import predict
        result = predict("U0100", "CAN bus communication error", 12.5)
        
        assert result["decision_engine"] == "LLM"
        assert result["status"] == "Approved"
