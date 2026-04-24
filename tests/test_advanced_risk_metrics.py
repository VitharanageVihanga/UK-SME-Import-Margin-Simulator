"""
Pytest suite for scripts/advanced_risk_metrics.py
Covers: VaR (historical, parametric, MC), volatility regime, correlation,
        beta, risk decomposition, stress testing.
"""
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture
def returns_series():
    """Synthetic daily returns (~252 days)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 300))


@pytest.fixture
def short_returns():
    return pd.Series([0.01, -0.02, 0.005])


@pytest.fixture
def margin_sim_df():
    """Simulated margin data with all required columns."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    fx = np.random.normal(0, 0.03, n)
    goods = np.random.normal(100_000, 5_000, n)
    return pd.DataFrame({
        "date": dates,
        "margin_pct": np.random.normal(15, 4, n),
        "fx_shock": fx,
        "goods_cost": goods,
        "shipping_cost": goods * 0.05,
        "tariff_cost": goods * 0.02,
        "landed_cost": goods * 1.08,
    })


# ═══════════════════════════════════
#  VaR — HISTORICAL
# ═══════════════════════════════════

class TestVarHistorical:
    def test_basic(self, returns_series):
        from scripts.advanced_risk_metrics import calculate_var_historical
        r = calculate_var_historical(returns_series, confidence_level=0.95)
        assert r["error"] is None
        assert r["var_value"] < 0  # loss is negative
        assert r["risk_level"] in ("LOW", "MODERATE", "HIGH")

    def test_insufficient_data(self, short_returns):
        from scripts.advanced_risk_metrics import calculate_var_historical
        r = calculate_var_historical(short_returns)
        assert r["error"] is not None


# ═══════════════════════════════════
#  VaR — PARAMETRIC
# ═══════════════════════════════════

class TestVarParametric:
    def test_basic(self, returns_series):
        from scripts.advanced_risk_metrics import calculate_var_parametric
        r = calculate_var_parametric(returns_series)
        assert r["error"] is None
        assert "var_value" in r

    def test_insufficient(self, short_returns):
        from scripts.advanced_risk_metrics import calculate_var_parametric
        r = calculate_var_parametric(short_returns)
        assert r["error"] is not None


# ═══════════════════════════════════
#  VaR — MONTE CARLO
# ═══════════════════════════════════

class TestVarMonteCarlo:
    def test_basic(self, returns_series):
        from scripts.advanced_risk_metrics import calculate_var_montecarlo
        r = calculate_var_montecarlo(returns_series, simulations=1000)
        assert r["error"] is None
        assert r["var_value"] < 0


# ═══════════════════════════════════
#  MARGIN VaR
# ═══════════════════════════════════

class TestMarginVar:
    def test_valid(self, margin_sim_df):
        from scripts.advanced_risk_metrics import calculate_margin_var
        r = calculate_margin_var(margin_sim_df)
        assert r["error"] is None
        assert "current_margin" in r

    def test_empty_df(self):
        from scripts.advanced_risk_metrics import calculate_margin_var
        r = calculate_margin_var(pd.DataFrame())
        assert r["error"] is not None

    def test_missing_column(self):
        from scripts.advanced_risk_metrics import calculate_margin_var
        r = calculate_margin_var(pd.DataFrame({"other": [1, 2, 3]}))
        assert r["error"] is not None


# ═══════════════════════════════════
#  VOLATILITY REGIME
# ═══════════════════════════════════

class TestVolatilityRegime:
    def test_low_vol(self):
        from scripts.advanced_risk_metrics import volatility_regime_detection
        low = pd.Series([0.05] * 30)
        r = volatility_regime_detection(low)
        assert r["current_regime"] == "LOW"

    def test_high_vol(self):
        from scripts.advanced_risk_metrics import volatility_regime_detection
        high = pd.Series([0.30] * 30)
        r = volatility_regime_detection(high)
        assert r["current_regime"] == "HIGH"

    def test_empty(self):
        from scripts.advanced_risk_metrics import volatility_regime_detection
        r = volatility_regime_detection(pd.Series(dtype=float))
        assert r.get("error") is not None


# ═══════════════════════════════════
#  CORRELATION
# ═══════════════════════════════════

class TestCorrelation:
    def test_perfect_positive(self):
        from scripts.advanced_risk_metrics import calculate_correlation_matrix
        s = pd.Series(range(50), dtype=float)
        corr = calculate_correlation_matrix({"a": s, "b": s * 2})
        assert corr.loc["a", "b"] == pytest.approx(1.0, abs=0.001)

    def test_insufficient_data(self):
        from scripts.advanced_risk_metrics import calculate_correlation_matrix
        s = pd.Series([1.0, 2.0])
        corr = calculate_correlation_matrix({"a": s, "b": s})
        assert corr.empty


# ═══════════════════════════════════
#  BETA
# ═══════════════════════════════════

class TestBeta:
    def test_identical_returns(self):
        from scripts.advanced_risk_metrics import calculate_beta
        np.random.seed(0)
        r = pd.Series(np.random.randn(50) * 0.01)
        result = calculate_beta(r, r)
        assert result["beta"] == pytest.approx(1.0, abs=0.01)

    def test_insufficient_data(self):
        from scripts.advanced_risk_metrics import calculate_beta
        r = calculate_beta(pd.Series([0.01]), pd.Series([0.02]))
        assert r["error"] is not None


# ═══════════════════════════════════
#  RISK DECOMPOSITION
# ═══════════════════════════════════

class TestRiskDecomposition:
    def test_valid(self, margin_sim_df):
        from scripts.advanced_risk_metrics import decompose_margin_risk
        r = decompose_margin_risk(margin_sim_df)
        assert r["error"] is None
        assert "primary_risk_driver" in r

    def test_empty(self):
        from scripts.advanced_risk_metrics import decompose_margin_risk
        r = decompose_margin_risk(pd.DataFrame())
        assert r["error"] is not None

    def test_missing_columns(self):
        from scripts.advanced_risk_metrics import decompose_margin_risk
        r = decompose_margin_risk(pd.DataFrame({"margin_pct": [1, 2]}))
        assert r["error"] is not None


# ═══════════════════════════════════
#  STRESS TESTING
# ═══════════════════════════════════

class TestStressTesting:
    def test_predefined_scenarios(self):
        from scripts.advanced_risk_metrics import run_stress_test, get_predefined_stress_scenarios
        scenarios = get_predefined_stress_scenarios()
        result = run_stress_test(20.0, scenarios)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(scenarios)
        assert "stressed_margin" in result.columns

    def test_base_case_margin_positive(self):
        from scripts.advanced_risk_metrics import run_stress_test
        result = run_stress_test(20.0, {
            "Base": {"fx_shock": 0, "shipping_pct": 0, "insurance_pct": 0, "tariff_pct": 0}
        })
        assert result["stressed_margin"].iloc[0] > 0

    def test_severe_shock_reduces_margin(self):
        from scripts.advanced_risk_metrics import run_stress_test
        result = run_stress_test(20.0, {
            "Severe": {"fx_shock": 0.20, "shipping_pct": 0.15, "insurance_pct": 0.03, "tariff_pct": 0.10}
        })
        assert result["stressed_margin"].iloc[0] < 20.0
