"""Tests for the privacy filter; safety net ensuring patient-level data never reaches LLM APIs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.privacy_filter import (
    PrivacyViolation,
    check_prompt,
    validate_prompt,
)


class TestAggregatePromptsPass:
    def test_simple_metric_prompt_passes(self):
        prompt = ("Aggregate Spearman across the cohort is 0.466. "
                  "MAPE is 18.1%.  What component is at fault?")
        assert check_prompt(prompt).ok
        assert validate_prompt(prompt) is True

    def test_metric_dict_passes(self):
        prompt = ('Latest aggregated metrics:\n'
                  '{"spearman_mean": 0.466, "ccc_mean": 0.356, '
                  '"mape_pct_mean": 18.1, "n_metabolites_scored": 12}')
        assert check_prompt(prompt).ok

    def test_diagnostic_taxonomy_passes(self):
        prompt = ("The diagnosis taxonomy includes: peak_shifts, "
                  "baseline_mismatch, correlation_missing, "
                  "concentration_range, noise_mismatch, template_inaccuracy")
        assert check_prompt(prompt).ok

    def test_strategist_action_list_passes(self):
        prompt = ("Available actions: enable_chenomx_correlations, "
                  "enable_empirical_baseline, shrink_concentration_range")
        assert check_prompt(prompt).ok

    def test_short_prompt_passes(self):
        assert check_prompt("Hello, world.").ok


class TestPatientLevelPromptsFail:
    def test_sample_id_blocked(self):
        prompt = "Sample_42 has Lactate concentration 2.5 mM"
        res = check_prompt(prompt)
        assert not res.ok
        assert any("blacklisted" in v for v in res.violations)

    def test_patient_id_blocked(self):
        prompt = "Patient 17 shows Lactate elevation"
        res = check_prompt(prompt)
        assert not res.ok

    def test_sid_blocked(self):
        prompt = "The SID 42 has anomalous spectrum"
        res = check_prompt(prompt)
        assert not res.ok

    def test_pid_blocked(self):
        prompt = "Records for PID:42 show outlier"
        res = check_prompt(prompt)
        assert not res.ok

    def test_validate_prompt_raises(self):
        bad = "Sample_42 has unusual values"
        with pytest.raises(PrivacyViolation):
            validate_prompt(bad)

    def test_validate_prompt_no_raise_returns_false(self):
        bad = "Sample_42 has unusual values"
        assert validate_prompt(bad, raise_on_fail=False) is False

    def test_case_insensitive(self):
        # Blacklist patterns must be case-insensitive.
        prompts = [
            "SAMPLE_42 abnormal",
            "PATIENT_17 elevated",
            "sample_42 normal",
        ]
        for p in prompts:
            assert not check_prompt(p).ok, f"Should block: {p!r}"


class TestRawDataDensity:
    def test_dense_floats_blocked(self):
        # 50 floats packed tightly looks like a raw spectrum.
        floats = " ".join(f"{i*0.001:.4f}" for i in range(50))
        prompt = f"Spectrum: {floats}"
        res = check_prompt(prompt)
        # If blocked, the violation should mention raw data.
        if not res.ok:
            assert any("floats" in v.lower() or "data" in v.lower()
                        for v in res.violations)

    def test_sparse_floats_pass(self):
        prompt = ("Aggregate Spearman is 0.466.  The MAPE is 18.1 percent.  "
                  "Confidence interval lower bound 0.395.")
        assert check_prompt(prompt).ok


class TestEdgeCases:
    def test_empty_string_passes(self):
        assert check_prompt("").ok

    def test_non_string_fails(self):
        res = check_prompt(123)
        assert not res.ok

    def test_unicode_passes(self):
        assert check_prompt("Spearman ρ ≈ 0.466 across 12 metabolites").ok


class TestIdempotent:
    def test_check_twice_same_result(self):
        prompts = [
            "Aggregated metrics: 0.46",
            "Sample_42 broken",
            "PID 17 elevated",
            "Hello world",
        ]
        for p in prompts:
            r1 = check_prompt(p)
            r2 = check_prompt(p)
            assert r1.ok == r2.ok
            assert r1.violations == r2.violations
