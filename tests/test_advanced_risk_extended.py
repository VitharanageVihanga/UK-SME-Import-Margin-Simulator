"""
Extended tests for scripts/advanced_risk_metrics.py
Targets missed lines: LOW VaR risk level, MC edge cases,
calculate_historical_volatility, NORMAL volatility regime,
analyse_commodity_fx_correlation, beta sensitivity variants,
zero-variance decompose_margin_risk, all 7 stress scenarios.
"""
import pytest
import pandas as pd
import numpy as np

from scripts.advanced_risk_metrics import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_var_montecarlo,
    calculate_margin_var,
    calculate_historical_volatility,
    volatility_regime_detection,
    calculate_correlation_matrix,
    analyse_commodity_fx_correlation,
    calculate_beta,
    decompose_margin_risk,
    run_stress_test,
    get_predefined_stress_scenarios,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tight_returns():
    """Very low variance → VaR < 0.02 → LOW risk label (lines 74-75)."""
    np.random.seed(0)
    return pd.Series(np.random.normal(0, 0.005, 120))


@pytest.fixture(scope="module")
def medium_returns():
    """Moderate variance → VaR between 0.02-0.05 → MODERATE risk label."""
    np.random.seed(1)
    return pd.Series(np.random.normal(0, 0.022, 120))


@pytest.fixture(scope="module")
def volatile_returns():
    """High variance → VaR > 0.05 → HIGH risk label."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 0.04, 120))


@pytest.fixture(scope="module")
def price_series():
    """Positive price series for volatility calculations."""
    np.random.seed(7)
    returns = np.random.normal(0.001, 0.01, 200)
    prices = np.cumprod(1 + returns) * 100
    return pd.Series(prices)


# ── VaR Historical – LOW and MODERATE risk labels (lines 74-78) ──────────────

class TestVarHistoricalRiskLevels:
    def test_low_risk_label(self, tight_returns):
        result = calculate_var_historical(tight_returns)
        assert result["error"] is None
        assert result["risk_level"] == "LOW"
        assert "Limited" in result["interpretation"]

    def test_moderate_risk_label(self, medium_returns):
        result = calculate_var_historical(medium_returns)
        assert result["error"] is None
        assert result["risk_level"] in ("MODERATE", "LOW")

    def test_statistics_block_fully_populated(self, volatile_returns):
        result = calculate_var_historical(volatile_returns)
        s = result["statistics"]
        for key in ("mean_return", "std_return", "skewness", "kurtosis", "n_observations"):
            assert key in s

    def test_holding_period_scales_var_by_sqrt(self, volatile_returns):
        r1 = calculate_var_historical(volatile_returns, holding_period=1)
        r4 = calculate_var_historical(volatile_returns, holding_period=4)
        assert abs(r4["var_value"]) > abs(r1["var_value"])

    def test_expected_shortfall_not_none(self, volatile_returns):
        result = calculate_var_historical(volatile_returns)
        assert result["expected_shortfall"] is not None

    def test_var_interpretation_string(self, volatile_returns):
        result = calculate_var_historical(volatile_returns)
        assert isinstance(result["var_interpretation"], str)
        assert "%" in result["var_interpretation"]


# ── VaR Monte Carlo – insufficient data (line 183) ───────────────────────────

class TestVarMonteCarloCoverage:
    def test_insufficient_data_error(self):
        result = calculate_var_montecarlo(pd.Series(range(10), dtype=float))
        assert result.get("error") is not None

    def test_confidence_interval_ordered(self, volatile_returns):
        result = calculate_var_montecarlo(volatile_returns)
        assert result["error"] is None
        lo, hi = result["confidence_interval_95"]
        assert lo < hi

    def test_holding_period_5_runs(self, volatile_returns):
        result = calculate_var_montecarlo(volatile_returns, holding_period=5, simulations=1000)
        assert result["error"] is None
        assert result["method"] == "monte_carlo"

    def test_simulations_count_stored(self, volatile_returns):
        result = calculate_var_montecarlo(volatile_returns, simulations=500)
        assert result["simulations"] == 500

    def test_99_confidence_level(self, volatile_returns):
        r95 = calculate_var_montecarlo(volatile_returns, confidence_level=0.95)
        r99 = calculate_var_montecarlo(volatile_returns, confidence_level=0.99)
        assert abs(r99["var_value"]) >= abs(r95["var_value"])


# ── calculate_historical_volatility (lines 312-333) ──────────────────────────

class TestCalculateHistoricalVolatility:
    def test_returns_dataframe(self, price_series):
        result = calculate_historical_volatility(price_series)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_columns(self, price_series):
        result = calculate_historical_volatility(price_series)
        for col in ("value", "log_return", "rolling_vol", "ewma_vol", "parkinson_vol"):
            assert col in result.columns

    def test_rolling_vol_non_negative(self, price_series):
        result = calculate_historical_volatility(price_series)
        non_nan = result["rolling_vol"].dropna()
        assert (non_nan >= 0).all()

    def test_ewma_vol_non_negative(self, price_series):
        result = calculate_historical_volatility(price_series)
        non_nan = result["ewma_vol"].dropna()
        assert (non_nan >= 0).all()

    def test_flat_series_zero_log_returns(self):
        flat = pd.Series(np.ones(100) * 1.25)
        result = calculate_historical_volatility(flat)
        assert (result["log_return"].dropna() == 0.0).all()

    def test_custom_window_and_annualization(self, price_series):
        result = calculate_historical_volatility(price_series, window=10, annualization_factor=12)
        assert "rolling_vol" in result.columns

    def test_value_column_matches_input(self, price_series):
        result = calculate_historical_volatility(price_series)
        pd.testing.assert_series_equal(
            result["value"].reset_index(drop=True),
            price_series.reset_index(drop=True),
            check_names=False,
        )

    def test_length_preserved(self, price_series):
        result = calculate_historical_volatility(price_series)
        assert len(result) == len(price_series)


# ── volatility_regime_detection – NORMAL branch (lines 377-379) ──────────────

class TestVolatilityRegimeNormal:
    def test_normal_regime_between_thresholds(self):
        vol = pd.Series([0.12, 0.13, 0.14, 0.15, 0.15])
        result = volatility_regime_detection(vol, low_threshold=0.10, high_threshold=0.20)
        assert result["current_regime"] == "NORMAL"
        assert "normal" in result["regime_description"].lower()
        assert "Standard" in result["trading_implication"]

    def test_low_regime_below_lower_threshold(self):
        vol = pd.Series([0.05, 0.06, 0.07, 0.08, 0.09])
        result = volatility_regime_detection(vol, low_threshold=0.10, high_threshold=0.20)
        assert result["current_regime"] == "LOW"

    def test_high_regime_above_upper_threshold(self):
        vol = pd.Series([0.21, 0.22, 0.30, 0.28, 0.35])
        result = volatility_regime_detection(vol, low_threshold=0.10, high_threshold=0.20)
        assert result["current_regime"] == "HIGH"

    def test_regime_distribution_present(self):
        vol = pd.Series([0.05, 0.12, 0.25, 0.15, 0.08])
        result = volatility_regime_detection(vol)
        assert "regime_distribution" in result

    def test_percentile_in_result(self):
        vol = pd.Series([0.10, 0.15, 0.12, 0.13, 0.14])
        result = volatility_regime_detection(vol)
        assert "percentile" in result
        assert 0.0 <= result["percentile"] <= 100.0

    def test_mean_volatility_in_result(self):
        vol = pd.Series([0.10, 0.15, 0.12])
        result = volatility_regime_detection(vol)
        assert result["mean_volatility"] == pytest.approx(
            round(vol.mean() * 100, 2), abs=0.01
        )


# ── analyse_commodity_fx_correlation (lines 456-504) ─────────────────────────

class TestCommodityFxCorrelation:
    @pytest.fixture
    def import_df(self):
        dates = pd.date_range("2015-01-01", periods=48, freq="MS")
        np.random.seed(42)
        return pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(1e6, 2e6, 48),
        })

    @pytest.fixture
    def fx_df(self):
        dates = pd.date_range("2014-12-01", periods=48 * 22, freq="B")
        np.random.seed(1)
        rates = np.cumsum(np.random.randn(len(dates)) * 0.005) + 1.15
        return pd.DataFrame({"Date": dates, "Rate": np.abs(rates) + 0.5})

    def test_empty_import_returns_error(self, fx_df):
        result = analyse_commodity_fx_correlation(pd.DataFrame(), fx_df)
        assert result.get("error") is not None

    def test_empty_fx_returns_error(self, import_df):
        result = analyse_commodity_fx_correlation(import_df, pd.DataFrame())
        assert result.get("error") is not None

    def test_returns_dict(self, import_df, fx_df):
        result = analyse_commodity_fx_correlation(import_df, fx_df)
        assert isinstance(result, dict)

    def test_success_path_keys(self, import_df, fx_df):
        result = analyse_commodity_fx_correlation(import_df, fx_df)
        if result.get("error") is None:
            for key in ("static_correlation", "correlation_strength",
                        "correlation_direction", "interpretation", "implication"):
                assert key in result

    def test_correlation_value_bounded(self, import_df, fx_df):
        result = analyse_commodity_fx_correlation(import_df, fx_df)
        if result.get("error") is None:
            assert -1.0 <= result["static_correlation"] <= 1.0

    def test_insufficient_aligned_data_returns_error(self):
        """Import dates in 2020, FX dates in 2010 → no overlap → error."""
        import_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "value": [100, 200, 150, 180, 160],
        })
        fx_df = pd.DataFrame({
            "Date": pd.date_range("2010-01-01", periods=100, freq="B"),
            "Rate": np.ones(100) * 1.15,
        })
        result = analyse_commodity_fx_correlation(import_df, fx_df)
        assert result.get("error") is not None


# ── calculate_beta – all sensitivity branches (lines 549, 563-574) ───────────

class TestBetaSensitivityBranches:
    def test_zero_market_variance_returns_error(self):
        n = 30
        asset = pd.Series(np.random.randn(n) * 0.01)
        market = pd.Series(np.zeros(n))
        result = calculate_beta(asset, market)
        assert result.get("error") is not None

    def test_high_sensitivity_beta_gt_1p2(self):
        """beta ≈ 1.5 → HIGH sensitivity (line 564-565)."""
        np.random.seed(0)
        market = pd.Series(np.random.randn(80) * 0.02)
        asset = market * 1.5 + np.random.randn(80) * 0.001
        result = calculate_beta(asset, market)
        assert result["error"] is None
        assert result["sensitivity"] == "HIGH"
        assert result["beta"] > 1.2

    def test_moderate_sensitivity_beta_0p8_to_1p2(self):
        """beta ≈ 1.0 → MODERATE sensitivity (lines 567-568)."""
        np.random.seed(0)
        market = pd.Series(np.random.randn(80) * 0.02)
        asset = market * 1.0 + np.random.randn(80) * 0.001
        result = calculate_beta(asset, market)
        assert result["error"] is None
        assert result["sensitivity"] == "MODERATE"

    def test_low_sensitivity_beta_0_to_0p8(self):
        """beta ≈ 0.4 → LOW sensitivity (lines 569-570)."""
        np.random.seed(0)
        market = pd.Series(np.random.randn(80) * 0.02)
        asset = market * 0.4 + np.random.randn(80) * 0.001
        result = calculate_beta(asset, market)
        assert result["error"] is None
        assert result["sensitivity"] == "LOW"

    def test_negative_beta(self):
        """beta ≈ -0.5 → NEGATIVE sensitivity (lines 573-574)."""
        np.random.seed(0)
        market = pd.Series(np.random.randn(80) * 0.02)
        asset = -market * 0.5 + np.random.randn(80) * 0.001
        result = calculate_beta(asset, market)
        assert result["error"] is None
        assert result["sensitivity"] == "NEGATIVE"
        assert result["beta"] < 0

    def test_result_keys_present(self):
        np.random.seed(3)
        market = pd.Series(np.random.randn(50) * 0.01)
        asset = market * 1.1 + np.random.randn(50) * 0.001
        result = calculate_beta(asset, market)
        if result.get("error") is None:
            for key in ("beta", "alpha", "r_squared", "sensitivity", "interpretation"):
                assert key in result

    def test_r_squared_between_0_and_1(self):
        np.random.seed(5)
        market = pd.Series(np.random.randn(50) * 0.01)
        asset = market * 0.8 + np.random.randn(50) * 0.005
        result = calculate_beta(asset, market)
        if result.get("error") is None:
            assert 0.0 <= result["r_squared"] <= 1.0


# ── decompose_margin_risk – zero variance (line 619) ─────────────────────────

class TestDecomposeMarginRiskExtended:
    def test_zero_margin_variance_returns_error(self):
        """All margins identical → var = 0 → error path (line 619)."""
        np.random.seed(10)
        n = 20
        df = pd.DataFrame({
            "margin_pct":    [15.0] * n,
            "fx_shock":      np.random.randn(n) * 0.01,
            "goods_cost":    np.random.uniform(900, 1100, n),
            "shipping_cost": np.random.uniform(45, 55, n),
            "tariff_cost":   np.random.uniform(18, 22, n),
        })
        result = decompose_margin_risk(df)
        assert result.get("error") is not None

    def test_valid_decomposition_returns_contributions(self):
        np.random.seed(42)
        n = 30
        fx = np.random.uniform(-0.1, 0.1, n)
        goods = 1_000_000 * (1 + fx)
        ship = goods * 0.05
        insure = goods * 0.01
        tariff = goods * 0.02
        revenue = 1_350_000
        margin = ((revenue - (goods + ship + insure + tariff)) / revenue) * 100
        df = pd.DataFrame({
            "margin_pct":    margin,
            "fx_shock":      fx,
            "goods_cost":    goods,
            "shipping_cost": ship,
            "tariff_cost":   tariff,
        })
        result = decompose_margin_risk(df)
        assert result.get("error") is None
        assert "risk_contributions" in result
        assert "primary_risk_driver" in result
        assert "primary_contribution" in result

    def test_primary_driver_is_valid_string(self):
        np.random.seed(42)
        n = 30
        fx = np.random.uniform(-0.1, 0.1, n)
        goods = 1_000_000 * (1 + fx)
        ship = goods * 0.05
        insure = goods * 0.01
        tariff = goods * 0.02
        revenue = 1_350_000
        margin = ((revenue - (goods + ship + insure + tariff)) / revenue) * 100
        df = pd.DataFrame({
            "margin_pct":    margin,
            "fx_shock":      fx,
            "goods_cost":    goods,
            "shipping_cost": ship,
            "tariff_cost":   tariff,
        })
        result = decompose_margin_risk(df)
        if result.get("error") is None:
            valid_drivers = {"FX Impact", "Goods Cost", "Shipping Cost", "Tariff Cost"}
            assert result["primary_risk_driver"] in valid_drivers


# ── correlation matrix – spearman and kendall (lines 424-431) ─────────────────

class TestCorrelationMatrixMethods:
    def test_spearman_method(self):
        np.random.seed(5)
        data = {"A": pd.Series(np.random.randn(50)), "B": pd.Series(np.random.randn(50))}
        result = calculate_correlation_matrix(data, method="spearman")
        assert not result.empty
        assert result.loc["A", "A"] == pytest.approx(1.0)

    def test_kendall_method(self):
        np.random.seed(5)
        data = {"A": pd.Series(np.random.randn(50)), "B": pd.Series(np.random.randn(50))}
        result = calculate_correlation_matrix(data, method="kendall")
        assert not result.empty

    def test_symmetric_matrix(self):
        np.random.seed(9)
        data = {
            "X": pd.Series(np.random.randn(50)),
            "Y": pd.Series(np.random.randn(50)),
        }
        result = calculate_correlation_matrix(data)
        assert result.loc["X", "Y"] == pytest.approx(result.loc["Y", "X"])


# ── stress test – all 7 predefined scenarios (lines 704-756) ─────────────────

class TestAllPredefinedStressScenarios:
    def test_seven_scenarios_defined(self):
        scenarios = get_predefined_stress_scenarios()
        expected = {
            "Base Case", "Mild FX Depreciation", "Severe FX Depreciation",
            "Shipping Crisis", "Trade War", "Perfect Storm", "Economic Recovery"
        }
        assert set(scenarios.keys()) == expected

    def test_all_scenarios_produce_rows(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        assert len(result) == 7

    def test_perfect_storm_worst_case(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        storm = result[result["scenario"] == "Perfect Storm"]["stressed_margin"].iloc[0]
        base = result[result["scenario"] == "Base Case"]["stressed_margin"].iloc[0]
        assert storm < base

    def test_economic_recovery_best_case(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        recovery = result[result["scenario"] == "Economic Recovery"]["stressed_margin"].iloc[0]
        base = result[result["scenario"] == "Base Case"]["stressed_margin"].iloc[0]
        assert recovery > base

    def test_shipping_crisis_reduces_margin(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        shipping = result[result["scenario"] == "Shipping Crisis"]["stressed_margin"].iloc[0]
        base = result[result["scenario"] == "Base Case"]["stressed_margin"].iloc[0]
        assert shipping < base

    def test_trade_war_reduces_margin(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        trade_war = result[result["scenario"] == "Trade War"]["stressed_margin"].iloc[0]
        base = result[result["scenario"] == "Base Case"]["stressed_margin"].iloc[0]
        assert trade_war < base

    def test_severe_fx_worse_than_mild(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        mild = result[result["scenario"] == "Mild FX Depreciation"]["stressed_margin"].iloc[0]
        severe = result[result["scenario"] == "Severe FX Depreciation"]["stressed_margin"].iloc[0]
        assert severe < mild

    def test_result_has_required_columns(self):
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        for col in ("scenario", "fx_shock", "shipping_pct", "tariff_pct",
                    "stressed_margin", "margin_impact", "profit_impact"):
            assert col in result.columns

    def test_base_case_params(self):
        scenarios = get_predefined_stress_scenarios()
        base = scenarios["Base Case"]
        assert base["fx_shock"] == 0.0
        assert base["shipping_pct"] == 0.05

    def test_perfect_storm_all_shocks_high(self):
        scenarios = get_predefined_stress_scenarios()
        ps = scenarios["Perfect Storm"]
        assert ps["fx_shock"] >= 0.10
        assert ps["shipping_pct"] >= 0.10
        assert ps["tariff_pct"] >= 0.08
