"""
Pytest suite for scripts/confidence_band.py and scripts/risk_adjuster.py
"""
import pytest
from scripts.confidence_band import confidence_multiplier, compute_confidence_band
from scripts.risk_adjuster import adjust_risk


# ═══════════════════════════════════
#  CONFIDENCE BAND (G3)
# ═══════════════════════════════════

class TestConfidenceMultiplier:
    def test_high_coverage(self):
        assert confidence_multiplier("High coverage") == 0.05

    def test_partial_coverage(self):
        assert confidence_multiplier("Partial coverage") == 0.15

    def test_low_coverage(self):
        assert confidence_multiplier("Low coverage") == 0.25

    def test_no_coverage(self):
        assert confidence_multiplier("No coverage") == 0.40

    def test_unknown_string(self):
        assert confidence_multiplier("Unknown") == 0.30

    def test_none_input(self):
        assert confidence_multiplier(None) == 0.30


class TestComputeConfidenceBand:
    def test_positive_profit_high_coverage(self):
        lo, hi = compute_confidence_band(1000, "High coverage")
        assert lo == pytest.approx(950, abs=1)
        assert hi == pytest.approx(1050, abs=1)

    def test_zero_profit(self):
        lo, hi = compute_confidence_band(0, "High coverage")
        assert lo == 0.0 and hi == 0.0

    def test_negative_profit(self):
        lo, hi = compute_confidence_band(-1000, "No coverage")
        # profit*(1-0.4) = -600, profit*(1+0.4) = -1400
        assert lo == pytest.approx(-600, abs=1)
        assert hi == pytest.approx(-1400, abs=1)

    def test_invalid_profit(self):
        lo, hi = compute_confidence_band("bad", "High coverage")
        assert lo == 0.0 and hi == 0.0


# ═══════════════════════════════════
#  RISK ADJUSTER (G4)
# ═══════════════════════════════════

class TestAdjustRisk:
    def test_high_stays_high(self):
        assert adjust_risk("HIGH", "High coverage") == "HIGH"

    def test_moderate_no_coverage_escalates(self):
        assert adjust_risk("MODERATE", "No coverage") == "HIGH"

    def test_moderate_high_coverage_stays(self):
        assert adjust_risk("MODERATE", "High coverage") == "MODERATE"

    def test_low_no_coverage_escalates(self):
        assert adjust_risk("LOW", "No coverage") == "MODERATE"

    def test_low_high_coverage_stays(self):
        assert adjust_risk("LOW", "High coverage") == "LOW"

    def test_non_string_margin_risk(self):
        assert adjust_risk(None, "High coverage") == "HIGH"

    def test_non_string_coverage(self):
        result = adjust_risk("LOW", None)
        assert isinstance(result, str)
