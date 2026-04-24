"""
Pytest suite for scripts/trend_analysis.py
Covers: rolling stats, volatility, seasonality, growth rates, margin simulation.
"""
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture
def price_series():
    """Daily price series with trend + noise."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(prices, index=dates)


@pytest.fixture
def monthly_import_df():
    """Monthly import data spanning 3 years."""
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    np.random.seed(7)
    return pd.DataFrame({
        "year": dates.year,
        "month": dates.month,
        "date": dates,
        "value": np.random.uniform(50_000, 150_000, 36),
        "net_mass": np.random.uniform(1000, 5000, 36),
    })


@pytest.fixture
def fx_df():
    """Monthly FX rate data."""
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    np.random.seed(7)
    return pd.DataFrame({
        "Date": dates,
        "Rate": np.random.uniform(1.10, 1.20, 36),
    })


# ═══════════════════════════════════
#  ROLLING STATISTICS
# ═══════════════════════════════════

class TestRollingStatistics:
    def test_returns_all_keys(self, price_series):
        from scripts.trend_analysis import calculate_rolling_statistics
        result = calculate_rolling_statistics(price_series, window=10)
        assert set(result.keys()) == {"rolling_mean", "rolling_std", "rolling_min", "rolling_max"}

    def test_rolling_mean_length(self, price_series):
        from scripts.trend_analysis import calculate_rolling_statistics
        result = calculate_rolling_statistics(price_series, window=10)
        assert len(result["rolling_mean"]) == len(price_series)


# ═══════════════════════════════════
#  VOLATILITY
# ═══════════════════════════════════

class TestVolatility:
    def test_annualised_vol_positive(self, price_series):
        from scripts.trend_analysis import calculate_volatility
        vol = calculate_volatility(price_series, window=30, annualize=True)
        assert vol.dropna().iloc[-1] > 0

    def test_flat_series_zero_vol(self):
        from scripts.trend_analysis import calculate_volatility
        flat = pd.Series([100.0] * 100)
        vol = calculate_volatility(flat, window=10, annualize=False)
        assert vol.dropna().iloc[-1] == pytest.approx(0, abs=1e-10)


# ═══════════════════════════════════
#  SEASONALITY DECOMPOSITION
# ═══════════════════════════════════

class TestDecomposeSeasonality:
    def test_insufficient_data(self):
        from scripts.trend_analysis import decompose_seasonality
        short = pd.Series(range(10))
        result = decompose_seasonality(short, period=12)
        assert result["error"] is not None

    def test_sufficient_data(self, price_series):
        from scripts.trend_analysis import decompose_seasonality
        # Resample to monthly for period=12
        monthly = price_series.resample("MS").mean()
        if len(monthly) >= 24:
            result = decompose_seasonality(monthly, period=12)
            assert result["error"] is None
            assert len(result["trend"].dropna()) > 0


# ═══════════════════════════════════
#  GROWTH RATES
# ═══════════════════════════════════

class TestGrowthRates:
    def test_known_growth(self):
        from scripts.trend_analysis import calculate_growth_rates
        s = pd.Series([100, 110, 121, 133.1])  # ~10% each
        result = calculate_growth_rates(s, periods=[1])
        assert result["1_period_growth"] == pytest.approx(10.0, abs=1)

    def test_short_series(self):
        from scripts.trend_analysis import calculate_growth_rates
        s = pd.Series([100])
        result = calculate_growth_rates(s, periods=[1, 3])
        assert result["1_period_growth"] is None


# ═══════════════════════════════════
#  SIMULATE HISTORICAL MARGINS
# ═══════════════════════════════════

class TestSimulateHistoricalMargins:
    def test_basic_simulation(self, monthly_import_df, fx_df):
        from scripts.trend_analysis import simulate_historical_margins
        result = simulate_historical_margins(monthly_import_df, fx_df)
        assert not result.empty
        assert "margin_pct" in result.columns
        assert "landed_cost" in result.columns

    def test_empty_import_data(self, fx_df):
        from scripts.trend_analysis import simulate_historical_margins
        result = simulate_historical_margins(pd.DataFrame(), fx_df)
        assert result.empty

    def test_empty_fx_data(self, monthly_import_df):
        from scripts.trend_analysis import simulate_historical_margins
        result = simulate_historical_margins(monthly_import_df, pd.DataFrame())
        assert result.empty


# ═══════════════════════════════════
#  MARGIN TRENDS
# ═══════════════════════════════════

class TestCalculateMarginTrends:
    def test_valid_data(self, monthly_import_df, fx_df):
        from scripts.trend_analysis import simulate_historical_margins, calculate_margin_trends
        sim = simulate_historical_margins(monthly_import_df, fx_df)
        result = calculate_margin_trends(sim)
        assert "current_margin" in result
        assert "trend_direction" in result

    def test_empty_data(self):
        from scripts.trend_analysis import calculate_margin_trends
        result = calculate_margin_trends(pd.DataFrame())
        assert result.get("error") is not None
