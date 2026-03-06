"""
Tests for LLM Client Resilience

These tests validate error handling and resilience of the LLM client.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import sys


class TestLLMClientResilience:
    """Test suite for LLM client error handling and resilience"""

    def test_timeout_handled(self):
        """30s timeout doesn't hang forever"""
        import requests
        
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        with patch('llm_client.requests.post') as mock_post:
            mock_post.side_effect = requests.Timeout("Request timed out")
            
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                
                with pytest.raises(RuntimeError, match="timed out"):
                    categorize_notes("Test notes", "P0562", 12.5)

    def test_invalid_json_handled(self):
        """Malformed LLM JSON doesn't crash"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "not valid json at all"
                }
            }]
        }
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                
                with pytest.raises(RuntimeError, match="Failed to parse LLM response as JSON"):
                    categorize_notes("Test notes", "P0562", 12.5)

    def test_rate_limit_handled(self):
        """Rate limit returns 429 gracefully"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                
                with pytest.raises(RuntimeError, match="Rate limited"):
                    categorize_notes("Test notes", "P0562", 12.5)

    def test_retry_on_rate_limit(self):
        """Retry logic triggers on rate limit"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.text = "Rate limit exceeded"
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "engine_symptom",
                        "confidence": 0.85,
                        "failure_analysis": "Engine issue",
                        "reasoning": "Test"
                    })
                }
            }]
        }
        
        with patch('llm_client.requests.post', side_effect=[mock_response_429, mock_response_success]):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes_with_retry
                
                result = categorize_notes_with_retry("Test notes", "P0562", 12.5)
                
                assert result is not None
                assert result["category"] == "engine_symptom"

    def test_retry_exhausted_raises(self):
        """After max retries, raises exception"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes_with_retry
                
                with pytest.raises(RuntimeError):
                    categorize_notes_with_retry("Test notes", "P0562", 12.5, max_retries=2)
