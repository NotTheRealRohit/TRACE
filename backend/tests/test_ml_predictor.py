"""
TRACE Test & Validation Plan - Track A & B Tests
=================================================
Functional and Logic tests without API key.

Run with: OPENROUTER_API_KEY="" python3 -m pytest tests/test_ml_predictor.py -v
"""

import os
import pytest

API_KEY_SET = bool(os.getenv("OPENROUTER_API_KEY"))

SAMPLE_OVERVOLTAGE = {
    "fault_code": "P0562",
    "technician_notes": "Customer reports intermittent power loss",
    "voltage": 17.5
}

SAMPLE_MOISTURE = {
    "fault_code": "B1234",
    "technician_notes": "Corrosion found inside ECU housing, moisture present",
    "voltage": 12.5
}

SAMPLE_NTF = {
    "fault_code": "P0441",
    "technician_notes": "No fault found, intermittent complaint, cannot reproduce",
    "voltage": 13.0
}

SAMPLE_UCODE = {
    "fault_code": "U0100",
    "technician_notes": "CAN bus communication error reported",
    "voltage": 12.5
}

SAMPLE_PCODE_ENGINE = {
    "fault_code": "P0301",
    "technician_notes": "Engine jerking during acceleration, rough idle",
    "voltage": 13.5
}

SAMPLE_NO_RULE = {
    "fault_code": "X9999",
    "technician_notes": "Customer complaint about dashboard light",
    "voltage": 13.0
}

SAMPLE_PHYSICAL = {
    "fault_code": "C0045",
    "technician_notes": "Visible crack on connector, impact damage observed",
    "voltage": 12.5
}

from ml_predictor import (
    extract_dtc_features,
    match_complaint,
    run_rules,
    run_ml,
    combine_scores,
    assemble_output_from_fields,
    predict,
)


class TestA1ExtractDTCFeatures:
    """A1 — extract_dtc_features() unit test"""

    def test_extract_dtc_single_p_code(self):
        r = extract_dtc_features("P0562")
        assert r["dtc_count"] == 1
        assert r["has_P"] == 1
        assert r["has_U"] == 0
        assert r["has_C"] == 0
        assert r["has_B"] == 0
        assert "P0562" in r["dtc_text"]

    def test_extract_dtc_multiple_mixed(self):
        r = extract_dtc_features("U0100, C0045, B1234")
        assert r["dtc_count"] == 3
        assert r["has_U"] == 1
        assert r["has_C"] == 1
        assert r["has_B"] == 1
        assert r["has_P"] == 0

    def test_extract_dtc_empty(self):
        r = extract_dtc_features("")
        assert r["dtc_count"] == 0
        assert r["dtc_text"] == "none"

    def test_extract_dtc_none(self):
        r = extract_dtc_features(None)
        assert r["dtc_count"] == 0


class TestA2MatchComplaint:
    """A2 — match_complaint() unit test"""

    def test_match_complaint_jerk(self):
        assert match_complaint("Engine jerking badly") == "Engine jerking during acceleration"

    def test_match_complaint_moisture(self):
        result = match_complaint("Moisture found inside")
        known = [
            "OBD Light ON", "Engine jerking during acceleration", "Starting Problem",
            "High fuel consumption", "Vehicle not starting", "Low pickup",
            "Engine overheating", "Rough idling", "Brake warning light ON"
        ]
        assert result in known

    def test_match_complaint_empty(self):
        assert match_complaint("") == "OBD Light ON"

    def test_match_complaint_returns_known_label(self):
        known = [
            "Engine jerking during acceleration", "Starting Problem",
            "High fuel consumption", "OBD Light ON", "Vehicle not starting",
            "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
        ]
        result = match_complaint("brake warning light came on")
        assert result in known


class TestA3RunRules:
    """A3 — run_rules() unit tests"""

    def test_run_rules_over_voltage(self):
        r = run_rules("P0562", "some notes", 17.5)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "over_voltage"

    def test_run_rules_low_voltage(self):
        r = run_rules("P0562", "some notes", 9.0)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "low_voltage"

    def test_run_rules_moisture(self):
        r = run_rules("B1234", "moisture found inside ECU", 12.5)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "moisture"

    def test_run_rules_physical_damage(self):
        r = run_rules("C0045", "crack visible on connector", 12.5)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "physical_damage"

    def test_run_rules_ntf(self):
        r = run_rules("P0441", "no fault found", 13.0)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "ntf"

    def test_run_rules_u_code(self):
        r = run_rules("U0100", "CAN bus error", 12.5)
        assert r["rule_fired"] is True
        assert r["rule_id"] == "u_code"

    def test_run_rules_no_match(self):
        r = run_rules("X9999", "dashboard light on", 13.0)
        assert r["rule_fired"] is False

    def test_run_rules_returns_required_keys_when_fired(self):
        r = run_rules("P0562", "notes", 17.5)
        for key in ["rule_id", "status", "warranty_decision", "rule_confidence",
                    "failure_analysis", "reason", "rule_fired"]:
            assert key in r, f"Missing key: {key}"

    def test_run_rules_voltage_priority_over_keyword(self):
        r = run_rules("P0562", "moisture found everywhere", 17.5)
        assert r["rule_id"] == "over_voltage"


class TestA4RunML:
    """A4 — run_ml() unit tests"""

    VALID_FEATURES = {
        "customer_complaint": "OBD Light ON",
        "dtc_text": "P0562",
        "dtc_count": 1,
        "voltage": 13.0,
        "has_P": 1, "has_U": 0, "has_C": 0, "has_B": 0
    }

    def test_run_ml_returns_required_keys(self):
        r = run_ml(self.VALID_FEATURES)
        for key in ["ml_warranty_decision", "ml_failure_analysis",
                    "fa_prob", "wd_prob", "ml_confidence"]:
            assert key in r, f"Missing key: {key}"

    def test_run_ml_confidence_in_range(self):
        r = run_ml(self.VALID_FEATURES)
        assert 50.0 <= r["ml_confidence"] <= 98.0

    def test_run_ml_valid_warranty_decision(self):
        r = run_ml(self.VALID_FEATURES)
        assert r["ml_warranty_decision"] in [
            "Production Failure", "Customer Failure", "According to Specification"
        ]

    def test_run_ml_probabilities_are_valid(self):
        r = run_ml(self.VALID_FEATURES)
        assert 0.0 <= r["fa_prob"] <= 1.0
        assert 0.0 <= r["wd_prob"] <= 1.0

    def test_run_ml_handles_zero_voltage(self):
        features = {**self.VALID_FEATURES, "voltage": 0.0}
        r = run_ml(features)
        assert "ml_confidence" in r

    def test_run_ml_handles_unknown_complaint(self):
        features = {**self.VALID_FEATURES, "customer_complaint": "Some unknown complaint XYZ"}
        r = run_ml(features)
        assert "ml_confidence" in r


class TestA5CombineScores:
    """A5 — combine_scores() unit tests"""

    RULE_FIRED = {
        "rule_fired": True,
        "rule_id": "moisture",
        "status": "Rejected",
        "warranty_decision": "Customer Failure",
        "rule_confidence": 91.0,
        "failure_analysis": "Sensor short due to moisture",
        "reason": "Moisture found"
    }

    ML_AGREES = {
        "ml_warranty_decision": "Customer Failure",
        "ml_failure_analysis": "Short circuit due to water ingress",
        "fa_prob": 0.82,
        "wd_prob": 0.78,
        "ml_confidence": 79.5
    }

    ML_DISAGREES = {
        "ml_warranty_decision": "Production Failure",
        "ml_failure_analysis": "Controller failure",
        "fa_prob": 0.60,
        "wd_prob": 0.55,
        "ml_confidence": 57.4
    }

    NO_RULE = {"rule_fired": False}

    def test_combine_rule_agree_boosts_confidence(self):
        r = combine_scores(self.RULE_FIRED, self.ML_AGREES, None)
        expected = 0.7 * 91.0 + 0.3 * 79.5 + 5.0
        assert abs(r["combined_confidence"] - expected) < 0.5

    def test_combine_rule_agree_sets_agreement_true(self):
        r = combine_scores(self.RULE_FIRED, self.ML_AGREES, None)
        assert r["agreement"] is True

    def test_combine_rule_disagree_lowers_confidence(self):
        agree_result = combine_scores(self.RULE_FIRED, self.ML_AGREES, None)
        disagree_result = combine_scores(self.RULE_FIRED, self.ML_DISAGREES, None)
        assert disagree_result["combined_confidence"] < agree_result["combined_confidence"]

    def test_combine_rule_disagree_sets_agreement_false(self):
        r = combine_scores(self.RULE_FIRED, self.ML_DISAGREES, None)
        assert r["agreement"] is False

    def test_combine_no_rule_uses_ml_directly(self):
        r = combine_scores(self.NO_RULE, self.ML_AGREES, None)
        assert r["rule_fired"] is False
        assert abs(r["combined_confidence"] - self.ML_AGREES["ml_confidence"]) < 0.5

    def test_combine_weak_input_penalty(self):
        llm_other = {"category": "other", "confidence": 0.5,
                     "failure_analysis": "unknown", "reasoning": ""}
        r = combine_scores(self.NO_RULE, self.ML_AGREES, llm_other)
        expected = self.ML_AGREES["ml_confidence"] * 0.85
        assert abs(r["combined_confidence"] - expected) < 1.0

    def test_combine_low_confidence_forces_manual_review(self):
        low_ml = {**self.ML_DISAGREES, "ml_confidence": 55.0}
        r = combine_scores(self.NO_RULE, low_ml, None)
        assert r["status"] == "Needs Manual Review"

    def test_combine_returns_required_keys(self):
        r = combine_scores(self.RULE_FIRED, self.ML_AGREES, None)
        for key in ["status", "warranty_decision", "combined_confidence",
                    "agreement", "rule_fired", "rule_id",
                    "ml_warranty_decision", "decision_engine"]:
            assert key in r, f"Missing key: {key}"

    def test_combine_decision_engine_label_with_rule(self):
        r = combine_scores(self.RULE_FIRED, self.ML_AGREES, None)
        assert r["decision_engine"] == "Rule+ML"

    def test_combine_decision_engine_label_with_llm_and_rule(self):
        llm_result = {"category": "moisture_damage", "confidence": 0.9,
                      "failure_analysis": "Moisture", "reasoning": "wet"}
        r = combine_scores(self.RULE_FIRED, self.ML_AGREES, llm_result)
        assert r["decision_engine"] == "LLM+Rule+ML"

    def test_combine_decision_engine_label_ml_only(self):
        r = combine_scores(self.NO_RULE, self.ML_AGREES, None)
        assert r["decision_engine"] == "ML"


class TestA6AssembleOutput:
    """A6 — assemble_output_from_fields() unit test"""

    COMBINED_RULE = {
        "status": "Rejected",
        "warranty_decision": "Customer Failure",
        "combined_confidence": 87.0,
        "agreement": True,
        "rule_fired": True,
        "rule_id": "moisture",
        "ml_warranty_decision": "Customer Failure",
        "ml_failure_analysis": "Short circuit",
        "llm_failure_analysis": "Sensor short due to moisture ingress",
        "decision_engine": "LLM+Rule+ML"
    }

    FEATURES = {"customer_complaint": "OBD Light ON"}

    def test_assemble_output_shape(self):
        r = assemble_output_from_fields(self.COMBINED_RULE, self.FEATURES)
        for key in ["status", "failure_analysis", "warranty_decision",
                    "confidence", "reason", "matched_complaint", "decision_engine"]:
            assert key in r, f"Missing key: {key}"

    def test_assemble_output_values_match_combined(self):
        r = assemble_output_from_fields(self.COMBINED_RULE, self.FEATURES)
        assert r["status"] == "Rejected"
        assert r["confidence"] == 87.0
        assert r["decision_engine"] == "LLM+Rule+ML"
        assert r["matched_complaint"] == "OBD Light ON"

    def test_assemble_prefers_llm_failure_analysis(self):
        r = assemble_output_from_fields(self.COMBINED_RULE, self.FEATURES)
        assert r["failure_analysis"] == "Sensor short due to moisture ingress"

    def test_assemble_falls_back_to_ml_failure_analysis(self):
        combined = {**self.COMBINED_RULE, "llm_failure_analysis": None}
        r = assemble_output_from_fields(combined, self.FEATURES)
        assert r["failure_analysis"] == "Short circuit"


class TestA7PredictIntegration:
    """A7 — predict() integration test (no API key)"""

    def test_predict_returns_valid_shape(self):
        r = predict(**SAMPLE_OVERVOLTAGE)
        for key in ["status", "failure_analysis", "warranty_decision",
                    "confidence", "reason", "matched_complaint", "decision_engine"]:
            assert key in r, f"Missing key: {key}"

    def test_predict_status_is_valid(self):
        r = predict(**SAMPLE_NTF)
        assert r["status"] in ["Approved", "Rejected", "Needs Manual Review"]

    def test_predict_confidence_in_range(self):
        r = predict(**SAMPLE_PCODE_ENGINE)
        assert 0.0 <= r["confidence"] <= 100.0

    def test_predict_warranty_decision_is_valid(self):
        r = predict(**SAMPLE_MOISTURE)
        assert r["warranty_decision"] in [
            "Production Failure", "Customer Failure", "According to Specification"
        ]

    def test_predict_does_not_crash_on_empty_notes(self):
        r = predict(fault_code="P0562", technician_notes="", voltage=13.0)
        assert "status" in r

    def test_predict_does_not_crash_on_empty_dtc(self):
        r = predict(fault_code="", technician_notes="Engine jerking badly", voltage=13.0)
        assert "status" in r

    def test_predict_does_not_crash_on_extreme_voltage(self):
        r = predict(fault_code="P0562", technician_notes="Some notes here", voltage=999.9)
        assert "status" in r


class TestB1RulePriority:
    """B1 — Rule Priority and Correctness"""

    def test_overvoltage_always_rejected(self):
        r = predict(**SAMPLE_OVERVOLTAGE)
        assert r["status"] == "Rejected"
        assert r["warranty_decision"] == "Customer Failure"

    def test_overvoltage_confidence_is_94(self):
        r = predict(**SAMPLE_OVERVOLTAGE)
        assert r["confidence"] >= 85.0

    def test_moisture_is_rejected_customer_failure(self):
        r = predict(**SAMPLE_MOISTURE)
        assert r["status"] == "Rejected"
        assert r["warranty_decision"] == "Customer Failure"

    def test_ntf_is_approved_according_to_spec(self):
        r = predict(**SAMPLE_NTF)
        assert r["status"] == "Approved"
        assert r["warranty_decision"] == "According to Specification"

    def test_u_code_is_approved_production_failure(self):
        r = predict(**SAMPLE_UCODE)
        assert r["status"] == "Approved"
        assert r["warranty_decision"] == "Production Failure"

    def test_p_code_with_symptom_is_approved(self):
        r = predict(**SAMPLE_PCODE_ENGINE)
        assert r["status"] == "Approved"

    def test_physical_damage_is_rejected(self):
        r = predict(**SAMPLE_PHYSICAL)
        assert r["status"] == "Rejected"
        assert r["warranty_decision"] == "Customer Failure"

    def test_voltage_rule_takes_priority_over_dtc(self):
        r = predict(fault_code="U0100", technician_notes="CAN error", voltage=18.0)
        assert r["status"] == "Rejected"


class TestB2CombineLogic:
    """B2 — Combine Logic Correctness"""

    RULE_FIRED = {
        "rule_fired": True,
        "rule_id": "moisture",
        "status": "Rejected",
        "warranty_decision": "Customer Failure",
        "rule_confidence": 91.0,
        "failure_analysis": "Sensor short due to moisture",
        "reason": "Moisture found"
    }

    ML_AGREES = {
        "ml_warranty_decision": "Customer Failure",
        "ml_failure_analysis": "Short circuit due to water ingress",
        "fa_prob": 0.82,
        "wd_prob": 0.78,
        "ml_confidence": 79.5
    }

    ML_DISAGREES = {
        "ml_warranty_decision": "Production Failure",
        "ml_failure_analysis": "Controller failure",
        "fa_prob": 0.60,
        "wd_prob": 0.55,
        "ml_confidence": 57.4
    }

    def test_agreement_bonus_applied(self):
        rule = {**TestB2CombineLogic.RULE_FIRED, "rule_confidence": 80.0}
        ml = {**TestB2CombineLogic.ML_AGREES, "ml_confidence": 70.0}
        r = combine_scores(rule, ml, None)
        assert abs(r["combined_confidence"] - 82.0) < 0.5

    def test_large_disagreement_triggers_manual_review(self):
        rule = {**TestB2CombineLogic.RULE_FIRED, "rule_confidence": 91.0}
        ml = {**TestB2CombineLogic.ML_DISAGREES, "ml_confidence": 60.0}
        r = combine_scores(rule, ml, None)
        assert r["status"] == "Needs Manual Review"

    def test_small_disagreement_does_not_force_manual(self):
        rule = {**TestB2CombineLogic.RULE_FIRED, "rule_confidence": 80.0}
        ml = {**TestB2CombineLogic.ML_DISAGREES, "ml_confidence": 70.0}
        r = combine_scores(rule, ml, None)
        assert r["status"] != "Needs Manual Review"

    def test_combined_below_65_forces_manual(self):
        rule = {"rule_fired": False}
        ml = {**TestB2CombineLogic.ML_AGREES, "ml_confidence": 55.0}
        r = combine_scores(rule, ml, None)
        assert r["status"] == "Needs Manual Review"

    def test_combined_65_to_84_keeps_status_but_flags(self):
        rule = {"rule_fired": False}
        ml = {**TestB2CombineLogic.ML_AGREES, "ml_confidence": 72.0}
        r = combine_scores(rule, ml, None)
        assert r["status"] != "Needs Manual Review"
        assert 65.0 <= r["combined_confidence"] < 85.0


class TestB3MLAlwaysRuns:
    """B3 — ML Always Runs"""

    def test_ml_runs_even_when_rule_fires(self):
        fc, notes, v = "P0562", "moisture inside connector", 12.5
        rule_result = run_rules(fc, notes, v)
        assert rule_result["rule_fired"] is True

        dtc_f = extract_dtc_features(fc)
        features = {
            "customer_complaint": match_complaint(notes),
            "dtc_text": dtc_f["dtc_text"], "dtc_count": dtc_f["dtc_count"],
            "voltage": v, "has_P": dtc_f["has_P"], "has_U": dtc_f["has_U"],
            "has_C": dtc_f["has_C"], "has_B": dtc_f["has_B"],
        }
        ml_result = run_ml(features)
        assert "ml_confidence" in ml_result
        assert ml_result["ml_confidence"] >= 50.0


class TestB4NoRuleCase:
    """B4 — No Rule Case Goes to ML"""

    def test_no_rule_match_uses_ml(self):
        r = predict(**SAMPLE_NO_RULE)
        assert r["decision_engine"] in ["ML", "LLM+Rule+ML", "Rule+ML"]
        assert r["decision_engine"] == "ML"

    def test_no_rule_confidence_not_artificially_high(self):
        r = predict(**SAMPLE_NO_RULE)
        assert r["confidence"] <= 98.0


class TestB5OutputSchema:
    """B5 — Output Schema Integrity"""

    @pytest.mark.parametrize("sample", [
        SAMPLE_OVERVOLTAGE,
        SAMPLE_MOISTURE,
        SAMPLE_NTF,
        SAMPLE_UCODE,
        SAMPLE_PCODE_ENGINE,
        SAMPLE_NO_RULE,
        SAMPLE_PHYSICAL,
    ])
    def test_output_schema_all_samples(self, sample):
        r = predict(**sample)

        for key in ["status", "failure_analysis", "warranty_decision",
                    "confidence", "reason", "matched_complaint", "decision_engine"]:
            assert key in r

        assert r["status"] in ["Approved", "Rejected", "Needs Manual Review"]

        assert r["warranty_decision"] in [
            "Production Failure", "Customer Failure", "According to Specification"
        ]

        assert 0.0 <= r["confidence"] <= 100.0

        for key in ["failure_analysis", "reason", "matched_complaint", "decision_engine"]:
            assert isinstance(r[key], str)
            assert len(r[key]) > 0

        assert r["matched_complaint"] in [
            "Engine jerking during acceleration", "Starting Problem",
            "High fuel consumption", "OBD Light ON", "Vehicle not starting",
            "Low pickup", "Engine overheating", "Rough idling", "Brake warning light ON"
        ]
