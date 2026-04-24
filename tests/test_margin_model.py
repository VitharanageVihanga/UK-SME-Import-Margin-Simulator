# tests/test_margin_model.py
"""
Pytest suite for scripts/margin_model.py
Covers: normal cases, edge cases, boundary values, error handling.
Target: ≥80 % line coverage of margin_model.py
"""
import pytest
from scripts.margin_model import compute_margin


# ────────────────────────────
#  BASIC FUNCTIONALITY
# ────────────────────────────

class TestComputeMarginBasic:
    """Verify the happy-path landed-cost calculation."""

    def test_zero_extras(self):
        """No FX shock, shipping, insurance, or tariff → landed == import value."""
        r = compute_margin(import_value_gbp=100_000, revenue_gbp=150_000)
        assert r["goods_cost"] == 100_000
        assert r["shipping_cost"] == 0
        assert r["insurance_cost"] == 0
        assert r["tariff_cost"] == 0
        assert r["landed_cost"] == 100_000
        assert r["profit"] == 50_000
        assert r["margin_pct"] == pytest.approx(33.33, abs=0.01)

    def test_all_extras(self):
        """Apply FX shock + shipping + insurance + tariff."""
        r = compute_margin(
            import_value_gbp=100_000,
            revenue_gbp=150_000,
            fx_shock_pct=0.05,       # +5 %
            shipping_pct=0.10,       # 10 %
            insurance_pct=0.02,      # 2 %
            tariff_pct=0.03,         # 3 %
        )
        # goods_cost = 100 000 × 1.05 = 105 000
        assert r["goods_cost"] == 105_000
        # shipping = 105 000 × 0.10 = 10 500
        assert r["shipping_cost"] == 10_500
        # insurance = 105 000 × 0.02 = 2 100
        assert r["insurance_cost"] == 2_100
        # tariff = 105 000 × 0.03 = 3 150
        assert r["tariff_cost"] == 3_150
        # landed = 105 000 + 10 500 + 2 100 + 3 150 = 120 750
        assert r["landed_cost"] == 120_750
        assert r["profit"] == 150_000 - 120_750
        assert r["margin_pct"] == pytest.approx(
            (150_000 - 120_750) / 150_000 * 100, abs=0.01
        )

    def test_negative_fx_shock(self):
        """GBP strengthens → goods cost decreases."""
        r = compute_margin(
            import_value_gbp=100_000,
            revenue_gbp=150_000,
            fx_shock_pct=-0.10,
        )
        assert r["goods_cost"] == 90_000
        assert r["landed_cost"] == 90_000
        assert r["profit"] == 60_000

    def test_return_types(self):
        """All returned values should be float (or None for margin_pct)."""
        r = compute_margin(import_value_gbp=1000, revenue_gbp=2000)
        for key in ("goods_cost", "shipping_cost", "insurance_cost",
                     "tariff_cost", "landed_cost", "profit"):
            assert isinstance(r[key], float)
        assert isinstance(r["margin_pct"], float)


# ────────────────────────────
#  EDGE / BOUNDARY CASES
# ────────────────────────────

class TestComputeMarginEdgeCases:
    """Boundary conditions and economic safety caps."""

    def test_zero_revenue_gives_none_margin(self):
        """revenue=0 → margin_pct is None."""
        r = compute_margin(import_value_gbp=100_000, revenue_gbp=0)
        assert r["margin_pct"] is None

    def test_cost_cap_at_200pct(self):
        """Landed cost must not exceed 200 % of import value."""
        r = compute_margin(
            import_value_gbp=100_000,
            revenue_gbp=500_000,
            fx_shock_pct=0.50,    # extreme
            shipping_pct=0.50,
            insurance_pct=0.10,
            tariff_pct=0.10,
        )
        assert r["landed_cost"] <= 100_000 * 2.0

    def test_profit_floor(self):
        """Profit is floored at -import_value."""
        r = compute_margin(
            import_value_gbp=100_000,
            revenue_gbp=10_000,   # revenue way below cost
        )
        assert r["profit"] >= -100_000

    def test_margin_floor_minus100(self):
        """Margin percentage should not go below -100 %."""
        r = compute_margin(
            import_value_gbp=100_000,
            revenue_gbp=10_000,
            fx_shock_pct=0.20,
            shipping_pct=0.30,
        )
        assert r["margin_pct"] >= -100

    def test_breakeven(self):
        """Revenue exactly equals landed cost → 0 % margin."""
        r = compute_margin(import_value_gbp=100_000, revenue_gbp=100_000)
        assert r["margin_pct"] == pytest.approx(0.0, abs=0.01)

    def test_small_values(self):
        """Tiny import value doesn't crash."""
        r = compute_margin(import_value_gbp=0.01, revenue_gbp=0.02)
        assert r["landed_cost"] > 0
        assert r["profit"] > 0

    def test_large_values(self):
        """Very large numbers don't overflow."""
        r = compute_margin(
            import_value_gbp=1_000_000_000,
            revenue_gbp=1_500_000_000,
        )
        assert r["profit"] == 500_000_000

    def test_zero_import_value(self):
        """Import value of zero is handled."""
        r = compute_margin(import_value_gbp=0, revenue_gbp=100_000)
        assert r["goods_cost"] == 0
        assert r["landed_cost"] == 0
        assert r["profit"] == 100_000
        assert r["margin_pct"] == 100.0

    def test_invalid_string_input(self):
        """Non-numeric input triggers error path (covers lines 42-43)."""
        r = compute_margin(import_value_gbp="not_a_number", revenue_gbp=100_000)
        assert "error" in r
        assert "Invalid input" in r["error"]

    def test_none_input(self):
        """None input triggers error path."""
        r = compute_margin(import_value_gbp=None, revenue_gbp=100_000)
        assert "error" in r
