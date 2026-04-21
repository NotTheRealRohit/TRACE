"""
Tests for LLM Client Module

These tests validate the OpenRouter LLM client functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import sys
import importlib


class TestLLMClient:
    """Test suite for llm_client module"""

    def test_returns_category_for_valid_notes(self):
        """LLM returns parsed category for normal input"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "moisture_damage",
                        "confidence": 0.85,
                        "failure_analysis": "Sensor short due to moisture",
                        "reasoning": "Technician noted water intrusion"
                    })
                }
            }]
        }
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                result = categorize_notes("Water found in connector", "P0562", 12.5)
                
        assert result["category"] in ["moisture_damage", "connector_damage"]
        assert result["confidence"] >= 0.8
        assert "moisture" in result["failure_analysis"].lower() or "connector" in result["failure_analysis"].lower()

    def test_handles_empty_notes(self):
        """Graceful handling of empty/missing notes"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "other",
                        "confidence": 0.5,
                        "failure_analysis": "Unknown",
                        "reasoning": "No notes provided"
                    })
                }
            }]
        }
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                result = categorize_notes("", "P0562", 12.5)
                
        assert result["category"] == "ntf"

    def test_handles_api_error(self):
        """API error still returns a result (fallback behavior)"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                result = categorize_notes("Test notes", "P0562", 12.5)
                assert result is not None

    def test_parses_json_response(self):
        """Correctly parses JSON from LLM response"""
        if 'llm_client' in sys.modules:
            del sys.modules['llm_client']
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "category": "physical_damage",
                        "confidence": 0.92,
                        "failure_analysis": "Connector damage",
                        "reasoning": "Physical impact noted"
                    })
                }
            }]
        }
        
        with patch('llm_client.requests.post', return_value=mock_response):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                from llm_client import categorize_notes
                result = categorize_notes("Connector cracked from impact", "C1234", None)
                
        assert result["category"] == "physical_damage"
        assert result["confidence"] >= 0.8
        assert isinstance(result["failure_analysis"], str)
        assert isinstance(result["reasoning"], str)

    @pytest.mark.skip(reason="Depends on env var priority")
    def test_api_key_loaded_from_env(self):
        """API key is loaded from environment variable"""
        import re
        with open('.env') as f:
            content = f.read()
        match = re.search(r'OPENROUTER_API_KEY=(\S+)', content)
        if match:
            api_key = match.group(1).strip('"')
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': api_key}):
                if 'llm_client' in sys.modules:
                    del sys.modules['llm_client']
                from llm_client import get_api_key
                assert get_api_key() == api_key
        else:
            pytest.skip("No OPENROUTER_API_KEY in .env file")

    def test_api_key_missing_raises_error(self):
        """Raises ValueError when API key not set"""
        env_without_key = {'PATH': '/usr/bin'}
        with patch.dict('os.environ', env_without_key, clear=True):
            if 'llm_client' in sys.modules:
                del sys.modules['llm_client']
            from llm_client import get_api_key
            with pytest.raises(ValueError, match="No LLM provider configured"):
                get_api_key()
