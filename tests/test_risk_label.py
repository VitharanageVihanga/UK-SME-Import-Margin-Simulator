"""
Pytest suite for scripts/risk_label.py
Covers: boundary classification, invalid inputs.
"""
import pytest
from scripts.risk_label import risk_label


class TestRiskLabel:
    """G2 — HIGH priority: financial risk classification."""

    def test_high_risk_negative_margin(self):
        assert risk_label(-5) == "HIGH"

    def test_high_risk_zero_margin(self):
        assert risk_label(0) == "HIGH"

    def test_high_risk_below_5(self):
        assert risk_label(4.99) == "HIGH"

    def test_boundary_5_is_moderate(self):
        assert risk_label(5) == "MODERATE"

    def test_moderate_at_9(self):
        assert risk_label(9.99) == "MODERATE"

    def test_boundary_10_is_low(self):
        assert risk_label(10) == "LOW"

    def test_low_at_20(self):
        assert risk_label(20) == "LOW"

    def test_low_at_100(self):
        assert risk_label(100) == "LOW"

    def test_none_is_high(self):
        assert risk_label(None) == "HIGH"

    def test_string_input_returns_high(self):
        assert risk_label("not_a_number") == "HIGH"

    def test_string_numeric(self):
        """String '15.0' should be cast to float and return LOW."""
        assert risk_label("15.0") == "LOW"
