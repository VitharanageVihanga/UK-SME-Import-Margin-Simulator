"""
Pytest suite for scripts/scenario_runner.py
Covers: sensitivity grid generation, edge cases, input validation.
"""
import pytest
import pandas as pd
import numpy as np
from scripts.scenario_runner import run_sensitivity_scenarios


class TestRunSensitivityScenarios:
    """G1 — HIGH priority: the sensitivity heatmap engine."""

    def test_basic_grid_shape(self):
        """Default 11 steps → 11×11 = 121 rows."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=150_000, steps=11
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11 * 11
        assert set(df.columns) >= {"fx_shock_pct", "shipping_pct", "profit", "margin_pct"}

    def test_margins_decrease_with_higher_fx_shock(self):
        """Higher FX shock should reduce margin (all else equal)."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=150_000,
            fx_range=(0.0, 0.2), shipping_range=(0.0, 0.0), steps=5
        )
        margins = df.sort_values("fx_shock_pct")["margin_pct"].values
        # monotonically non-increasing
        assert all(margins[i] >= margins[i + 1] - 0.01 for i in range(len(margins) - 1))

    def test_zero_revenue(self):
        """revenue=0 → margin_pct should be None for every row."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=0, steps=3
        )
        assert df["margin_pct"].isna().all()

    def test_zero_import_value(self):
        """import_value=0 → landed cost = 0, profit = revenue."""
        df = run_sensitivity_scenarios(
            import_value_gbp=0, revenue_gbp=100_000, steps=3
        )
        assert (df["profit"] == 100_000).all()

    def test_custom_tariff_and_insurance(self):
        """Tariff and insurance are passed through correctly."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=200_000,
            fx_range=(0.0, 0.0), shipping_range=(0.0, 0.0),
            steps=2, tariff_pct=0.10, insurance_pct=0.05,
        )
        # With 0 fx shock and 0 shipping, landed = 100k * (1 + 0.10 + 0.05) = 115k
        assert df["profit"].iloc[0] == pytest.approx(85_000, abs=1)

    def test_invalid_input_returns_empty(self):
        """Non-numeric input → graceful empty DataFrame."""
        df = run_sensitivity_scenarios(
            import_value_gbp="bad", revenue_gbp=100_000
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_negative_margin_scenario(self):
        """Revenue below cost → negative margin in some cells."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=50_000,
            fx_range=(0.0, 0.1), shipping_range=(0.0, 0.1), steps=3,
        )
        assert (df["margin_pct"] < 0).any()

    def test_minimum_steps(self):
        """Steps < 2 is clamped to 2."""
        df = run_sensitivity_scenarios(
            import_value_gbp=100_000, revenue_gbp=150_000, steps=1
        )
        assert len(df) == 4  # 2×2
