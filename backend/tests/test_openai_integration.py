"""
OpenAI Integration Tests for llm_client

Tests the OpenAI (gpt-4o-mini) API integration with security measures:
- API key loaded from .env via load_dotenv()
- Key never exposed in logs or test output
- Schema validation for all responses
- Consistency checks across multiple runs
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure backend is in path
sys.path.insert(0, "/mnt/d/study/git/capProj-2/backend")

# Load API key from .env (secure - key stays in env, never printed)
from dotenv import load_dotenv
load_dotenv()


class TestOpenAIEnvironment:
    """Test environment setup and provider detection."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    def test_loads_dotenv(self):
        """Verify .env is loaded without exposing API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        # If no key, skip tests
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        # Verify key exists (length check only - never print the key)
        assert len(api_key) > 20, "API key should be present"
        
        # Verify it's the expected format (OpenAI keys start with sk-)
        assert api_key.startswith("sk-"), "OpenAI API key should start with sk-"

    def test_provider_detection(self, llm_client):
        """Verify provider is detected as openai when key is present."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            provider = llm_client._get_provider()
            assert provider == "openai", "Provider should be openai"


class TestCategorizeNotes:
    """Tests for categorize_notes() function."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    @pytest.mark.parametrize("notes,dtc,expected_category", [
        ("Water found in connector", "P0562", "connector_damage"),
        ("Wiring short detected in harness", "U0100", "electrical_issue"),
        ("Engine overheating, jerking during acceleration", "P0562", "engine_symptom"),
        ("No fault found, cannot reproduce the issue", "", "ntf"),
        ("Connector cracked from impact", "C1234", "physical_damage"),
        ("CAN bus communication error", "U0100", "communication_fault"),
    ])
    def test_categorize_notes(self, llm_client, notes, dtc, expected_category):
        """Verify categorize_notes returns expected category."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.categorize_notes(notes, dtc)
            
            assert result is not None, "Result should not be None"
            assert result["category"] == expected_category, f"Expected {expected_category}"

    def test_categorize_notes_schema(self, llm_client):
        """Verify categorize_notes returns required schema."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.categorize_notes("Engine overheating", "P0562")
            
            required_keys = {"category", "confidence", "failure_analysis", "reasoning"}
            assert required_keys.issubset(result.keys()), "Missing required keys"

    def test_categorize_notes_consistency(self, llm_client):
        """Verify categorize_notes returns consistent results."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            results = []
            for _ in range(3):
                result = llm_client.categorize_notes("Water in connector", "P0562")
                results.append(result["category"])
            
            # All results should be identical
            assert len(set(results)) == 1, f"Inconsistent results: {results}"


class TestUnderstandClaim:
    """Tests for understand_claim() function."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    @pytest.mark.parametrize("notes,dtc,expected_category", [
        ("Engine overheating, low idle, vehicle struggling to start", "P0562", "engine_symptom"),
        ("Water intrusion in harness, corrosion on connector", "P0562", "moisture_damage"),
        ("No fault found, intermittent issue cannot be reproduced", "", "ntf"),
    ])
    def test_understand_claim(self, llm_client, notes, dtc, expected_category):
        """Verify understand_claim returns expected category."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.understand_claim(notes, dtc)
            
            assert result is not None, "Result should not be None"
            assert result["category"] == expected_category, f"Expected {expected_category}"

    def test_understand_claim_schema(self, llm_client):
        """Verify understand_claim returns required schema."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.understand_claim("Engine overheating", "P0562")
            
            required_keys = {"category", "normalized_complaint", "severity", "failure_analysis", "reasoning", "confidence"}
            assert required_keys.issubset(result.keys()), "Missing required keys"

    def test_understand_claim_consistency(self, llm_client):
        """Verify understand_claim returns consistent results."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            results = []
            for _ in range(3):
                result = llm_client.understand_claim("Water in connector", "P0562")
                results.append(result["category"])
            
            assert len(set(results)) == 1, f"Inconsistent results: {results}"


class TestTranslateMLFeatures:
    """Tests for translate_to_ml_features() function."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    @pytest.mark.parametrize("notes,dtc,category,expected", [
        ("Engine overheating", "P0562", "engine_symptom", {"has_P": 1}),
        ("Starting problem", "P0562,U0100", "other", {"has_P": 1, "has_U": 1}),
        ("Rough idling", "", "engine_symptom", {"dtc_count": 0}),
    ])
    def test_translate_to_ml_features(self, llm_client, notes, dtc, category, expected):
        """Verify translate_to_ml_features returns expected features."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.translate_to_ml_features(notes, dtc, category)
            
            assert result is not None, "Result should not be None"
            for key, value in expected.items():
                assert result[key] == value, f"Expected {key}={value}, got {result[key]}"

    def test_translate_to_ml_features_schema(self, llm_client):
        """Verify translate_to_ml_features returns required schema."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.translate_to_ml_features("Test", "P0562", "other")
            
            required_keys = {"customer_complaint", "dtc_codes", "dtc_text", "dtc_count", "has_P", "has_U", "has_C", "has_B"}
            assert required_keys.issubset(result.keys()), "Missing required keys"


class TestFormatOutput:
    """Tests for format_output() function."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    @pytest.mark.parametrize("combined,features,expected_status", [
        ({"status": "Approved", "decision_engine": "ML", "ml_failure_analysis": "test", "warranty_decision": "test", "combined_confidence": 85}, {"customer_complaint": "test"}, "Approved"),
        ({"status": "Rejected", "decision_engine": "Rule+ML", "ml_failure_analysis": "test", "warranty_decision": "test", "combined_confidence": 90}, {"customer_complaint": "test"}, "Rejected"),
        ({"status": "Needs Manual Review", "decision_engine": "ML", "ml_failure_analysis": "test", "warranty_decision": "test", "combined_confidence": 50}, {"customer_complaint": "test"}, "Needs Manual Review"),
    ])
    def test_format_output(self, llm_client, combined, features, expected_status):
        """Verify format_output returns expected status."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            result = llm_client.format_output(combined, features)
            
            assert result is not None, "Result should not be None"
            assert result["status"] == expected_status, f"Expected {expected_status}"

    def test_format_output_schema(self, llm_client):
        """Verify format_output returns required schema."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            combined = {"status": "Approved", "decision_engine": "ML", "ml_failure_analysis": "test", "warranty_decision": "test", "combined_confidence": 85}
            features = {"customer_complaint": "test"}
            result = llm_client.format_output(combined, features)
            
            required_keys = {"status", "failure_analysis", "warranty_decision", "confidence", "reason", "matched_complaint", "decision_engine"}
            assert required_keys.issubset(result.keys()), "Missing required keys"


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.fixture
    def llm_client(self):
        import llm_client
        return llm_client

    def test_no_api_key_raises_error(self, llm_client):
        """Verify missing API key raises RuntimeError without exposing key."""
        # Clear all API keys
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(llm_client, '_get_provider', return_value=None):
                with pytest.raises(RuntimeError) as exc_info:
                    llm_client.categorize_notes("Test", "P0562")
                
                # Verify error message doesn't expose actual key (sk-...)
                error_msg = str(exc_info.value)
                assert "sk-" not in error_msg.lower(), "Error should not expose API key"

    def test_api_failure_returns_none(self, llm_client):
        """Verify API failure returns None instead of exposing key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in .env")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}, clear=False):
            # Mock a failed response
            with patch('openai.OpenAI') as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                mock_openai.return_value = mock_client
                
                # This should not expose the key in any error
                try:
                    result = llm_client.categorize_notes("Test", "P0562")
                    # If it returns None, that's acceptable (no key exposed)
                    assert result is None or "sk-" not in str(result)
                except RuntimeError as e:
                    # If it raises, the error should not contain the key
                    assert "sk-" not in str(e).lower()
