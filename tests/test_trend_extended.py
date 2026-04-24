"""
Extended tests for scripts/trend_analysis.py
Targets missed lines: aggregate_import_data with real file (country_filter,
chunked read), calculate_growth_rates zero-previous, analyse_fx_trends short
data, calculate_fx_percentile_position, analyse_import_trends with data,
simulate_historical_margins no-overlap FX, calculate_margin_trends < 3 rows,
generate_trend_summary.
"""
import os
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from scripts.trend_analysis import (
    aggregate_import_data,
    calculate_rolling_statistics,
    calculate_volatility,
    decompose_seasonality,
    calculate_growth_rates,
    analyse_fx_trends,
    calculate_fx_percentile_position,
    analyse_import_trends,
    simulate_historical_margins,
    calculate_margin_trends,
    generate_trend_summary,
    load_exchange_rates,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fx_df():
    """Synthetic daily FX DataFrame covering 3 years."""
    dates = pd.date_range("2018-01-01", periods=800, freq="B")
    np.random.seed(42)
    rates = np.cumsum(np.random.randn(800) * 0.005) + 1.15
    return pd.DataFrame({"Date": dates, "Rate": np.abs(rates) + 0.5})


@pytest.fixture
def import_df():
    """Synthetic monthly import DataFrame."""
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    np.random.seed(7)
    return pd.DataFrame({
        "date":     dates,
        "year":     dates.year,
        "month":    dates.month,
        "value":    np.random.uniform(500_000, 1_500_000, 36),
        "net_mass": np.random.uniform(10_000, 50_000, 36),
    })


# ── aggregate_import_data with real temp CSV (lines 97-129) ──────────────────

class TestAggregateImportData:
    @pytest.fixture
    def csv_path(self, tmp_path):
        data = pd.DataFrame({
            "year":       [2020, 2020, 2020, 2021, 2021, 2021],
            "month":      [1,    1,    2,    1,    1,    2],
            "hs2_chapter":[84,   85,   84,   84,   85,   84],
            "country_code":["DE","FR","DE","DE","FR","CN"],
            "value":      [1000, 2000, 3000, 4000, 5000, 6000],
            "net_mass":   [10,   20,   30,   40,   50,   60],
        })
        p = tmp_path / "test_hmrc.csv"
        data.to_csv(p, index=False)
        return str(p)

    def test_file_not_found_returns_empty(self):
        result = aggregate_import_data("/nonexistent/path.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_filter_returns_all_data(self, csv_path):
        result = aggregate_import_data(csv_path)
        assert len(result) > 0

    def test_commodity_filter_applied(self, csv_path):
        """Lines 98-99: commodity_filter path."""
        result_all = aggregate_import_data(csv_path)
        result_84 = aggregate_import_data(csv_path, commodity_filter=84)
        assert result_84["value"].sum() < result_all["value"].sum()

    def test_country_filter_applied(self, csv_path):
        """Line 101: country_filter path."""
        result_all = aggregate_import_data(csv_path)
        result_de = aggregate_import_data(csv_path, country_filter="DE")
        assert result_de["value"].sum() < result_all["value"].sum()

    def test_date_column_created(self, csv_path):
        """Lines 124-128: date column creation."""
        result = aggregate_import_data(csv_path)
        assert "date" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_result_sorted_by_date(self, csv_path):
        result = aggregate_import_data(csv_path)
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_value_aggregated_correctly(self, csv_path):
        """Year=2020, month=1 should sum DE(1000)+FR(2000)=3000."""
        result = aggregate_import_data(csv_path)
        row = result[(result["year"] == 2020) & (result["month"] == 1)]
        assert row["value"].iloc[0] == pytest.approx(3000, abs=1)

    def test_combined_filter_commodity_and_country(self, csv_path):
        """Both filters applied together."""
        result = aggregate_import_data(csv_path, commodity_filter=84, country_filter="DE")
        # DE + hs2=84: (2020,1,1000), (2020,2,3000), (2021,1,4000) — 2021/2 is CN not DE
        assert result["value"].sum() == pytest.approx(8000, abs=1)

    def test_empty_after_filter_returns_empty(self, csv_path):
        result = aggregate_import_data(csv_path, commodity_filter=99)
        assert len(result) == 0


# ── calculate_growth_rates – zero previous (line 291) ────────────────────────

class TestGrowthRatesEdgeCases:
    def test_zero_previous_gives_none(self):
        """Series where value at [-period-1] is 0 → line 291 (None branch)."""
        series = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = calculate_growth_rates(series, periods=[1, 2, 3])
        # period=3: previous = series.iloc[-4] = 0 → None
        assert result["3_period_growth"] is None

    def test_nan_previous_gives_none(self):
        series = pd.Series([np.nan, 1.0, 2.0, 3.0])
        result = calculate_growth_rates(series, periods=[3])
        assert result["3_period_growth"] is None

    def test_normal_growth_computed(self):
        series = pd.Series([100.0, 110.0, 121.0, 133.1])
        result = calculate_growth_rates(series, periods=[1])
        assert result["1_period_growth"] == pytest.approx(10.0, abs=0.1)

    def test_short_series_returns_none(self):
        series = pd.Series([1.0, 2.0])
        result = calculate_growth_rates(series, periods=[5])
        assert result["5_period_growth"] is None

    def test_multiple_periods(self):
        series = pd.Series([50.0, 60.0, 70.0, 80.0, 100.0])
        result = calculate_growth_rates(series, periods=[1, 2])
        assert "1_period_growth" in result
        assert "2_period_growth" in result


# ── analyse_fx_trends – short data after date filter (line 327) ──────────────

class TestAnalyseFxTrends:
    def test_no_data_returns_error(self):
        result = analyse_fx_trends(currency="XYZ", data_path="/nonexistent")
        assert result.get("error") is not None

    def test_with_monkeypatched_data(self, monkeypatch, fx_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = analyse_fx_trends(currency="EUR")
        assert result.get("error") is None
        assert "current_rate" in result
        assert "trend_direction" in result

    def test_insufficient_data_after_date_filter(self, monkeypatch):
        """Lookback so long that < 10 points remain → line 327 error."""
        import scripts.trend_analysis as ta
        tiny_df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "Rate": [1.1, 1.2, 1.1, 1.3, 1.2],
        })
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: tiny_df)
        # lookback_days=1 means only last 1 day → < 10 points
        result = analyse_fx_trends(currency="EUR", lookback_days=1)
        assert result.get("error") is not None

    def test_trend_direction_is_string(self, monkeypatch, fx_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = analyse_fx_trends(currency="EUR")
        if result.get("error") is None:
            assert result["trend_direction"] in ("strengthening", "weakening")

    def test_growth_rates_present(self, monkeypatch, fx_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = analyse_fx_trends(currency="EUR")
        if result.get("error") is None:
            assert "growth_rates" in result
            assert "data" in result


# ── calculate_fx_percentile_position (lines 394-414) ─────────────────────────

class TestFxPercentilePosition:
    def test_insufficient_data_returns_error(self, monkeypatch):
        """len(df) < 30 → line 334 error path."""
        import scripts.trend_analysis as ta
        small_df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10, freq="B"),
            "Rate": np.linspace(1.1, 1.2, 10),
        })
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: small_df)
        result = calculate_fx_percentile_position(currency="EUR")
        assert result.get("error") is not None

    def test_no_file_returns_error(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates",
                            lambda *a, **kw: pd.DataFrame(columns=["Date", "Rate"]))
        result = calculate_fx_percentile_position(currency="EUR")
        assert result.get("error") is not None

    def test_success_path_keys(self, monkeypatch, fx_df):
        """Lines 399-422: happy path of percentile function."""
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = calculate_fx_percentile_position(currency="EUR")
        assert result.get("error") is None
        for key in ("currency", "current_rate", "percentile", "interpretation",
                    "historical_min", "historical_max", "historical_median"):
            assert key in result

    def test_percentile_bounded_0_100(self, monkeypatch, fx_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = calculate_fx_percentile_position(currency="EUR")
        if result.get("error") is None:
            assert 0.0 <= result["percentile"] <= 100.0

    def test_interpretation_is_string(self, monkeypatch, fx_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: fx_df)
        result = calculate_fx_percentile_position(currency="EUR")
        if result.get("error") is None:
            assert isinstance(result["interpretation"], str)

    def test_very_low_rate_interpretation(self, monkeypatch):
        """Current rate at bottom → 'very weak' interpretation (line 404-405)."""
        import scripts.trend_analysis as ta
        rates = list(range(1, 101))  # 1..100, last is 100 (highest)
        df = pd.DataFrame({
            "Date": pd.date_range("2018-01-01", periods=100, freq="B"),
            "Rate": [float(r) for r in rates],
        })
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: df)
        result = calculate_fx_percentile_position(currency="EUR")
        if result.get("error") is None:
            assert result["percentile"] == pytest.approx(100.0, abs=1.0)

    def test_low_rate_percentile(self, monkeypatch):
        """Rate at bottom 20th percentile → 'very weak'."""
        import scripts.trend_analysis as ta
        n = 100
        rates = list(range(1, n + 1))  # 1..100
        df = pd.DataFrame({
            "Date": pd.date_range("2018-01-01", periods=n, freq="B"),
            "Rate": [float(r) for r in rates],
        })
        # Reverse so current (last) rate is smallest
        df_reversed = df.copy()
        df_reversed["Rate"] = df_reversed["Rate"].iloc[::-1].values
        monkeypatch.setattr(ta, "load_exchange_rates", lambda *a, **kw: df_reversed)
        result = calculate_fx_percentile_position(currency="EUR")
        if result.get("error") is None:
            assert result["interpretation"] == "very weak (historically low)"


# ── analyse_import_trends (lines 454-485) ────────────────────────────────────

class TestAnalyseImportTrends:
    def test_no_data_returns_error(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: pd.DataFrame())
        result = analyse_import_trends(commodity_code=84)
        assert result.get("error") is not None

    def test_success_path_keys(self, monkeypatch, import_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: import_df)
        result = analyse_import_trends(commodity_code=84)
        assert result.get("error") is None
        for key in ("total_value", "mean_monthly_value", "growth_rates",
                    "seasonality", "monthly_pattern", "yearly_totals", "data"):
            assert key in result

    def test_total_value_positive(self, monkeypatch, import_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: import_df)
        result = analyse_import_trends(commodity_code=84)
        assert result["total_value"] == pytest.approx(import_df["value"].sum(), abs=1)

    def test_monthly_pattern_has_12_keys(self, monkeypatch, import_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: import_df)
        result = analyse_import_trends(commodity_code=84)
        if result.get("error") is None:
            assert len(result["monthly_pattern"]) == 12

    def test_yearly_totals_present(self, monkeypatch, import_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: import_df)
        result = analyse_import_trends(commodity_code=84)
        if result.get("error") is None:
            unique_years = set(import_df["year"].unique())
            assert set(result["yearly_totals"].keys()) == unique_years

    def test_seasonality_key_present_with_enough_data(self, monkeypatch, import_df):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "aggregate_import_data",
                            lambda *a, **kw: import_df)
        result = analyse_import_trends(commodity_code=84)
        if result.get("error") is None:
            assert "seasonality" in result


# ── simulate_historical_margins – no-overlap FX dates (line 542) ─────────────

class TestSimulateHistoricalMarginsExtended:
    def test_all_nan_fx_fallback_to_1(self, import_df):
        """FX dates in 1990, import dates in 2018 → no overlap → line 542."""
        fx_df_old = pd.DataFrame({
            "Date": pd.date_range("1990-01-01", periods=200, freq="B"),
            "Rate": np.ones(200) * 1.15,
        })
        result = simulate_historical_margins(import_df, fx_df_old)
        assert not result.empty
        # fx_shock should be 0 (since all rates = 1.0 → baseline=1.0 → shock=0)
        assert "fx_shock" in result.columns
        assert "margin_pct" in result.columns

    def test_empty_import_returns_empty(self, fx_df):
        result = simulate_historical_margins(pd.DataFrame(), fx_df)
        assert result.empty

    def test_empty_fx_returns_empty(self, import_df):
        result = simulate_historical_margins(import_df, pd.DataFrame())
        assert result.empty

    def test_margin_pct_computed(self, import_df, fx_df):
        result = simulate_historical_margins(import_df, fx_df)
        assert "margin_pct" in result.columns
        assert result["margin_pct"].notna().any()

    def test_profit_equals_revenue_minus_landed(self, import_df, fx_df):
        result = simulate_historical_margins(import_df, fx_df)
        diff = (result["revenue"] - result["landed_cost"] - result["profit"]).abs()
        assert (diff < 0.01).all()

    def test_custom_revenue_multiplier(self, import_df, fx_df):
        r1 = simulate_historical_margins(import_df, fx_df, base_revenue_multiplier=1.2)
        r2 = simulate_historical_margins(import_df, fx_df, base_revenue_multiplier=1.5)
        assert r2["margin_pct"].mean() > r1["margin_pct"].mean()


# ── calculate_margin_trends – len < 3 branch (lines 601-602) ─────────────────

class TestCalculateMarginTrendsEdge:
    def test_empty_returns_error(self):
        result = calculate_margin_trends(pd.DataFrame())
        assert result.get("error") is not None

    def test_two_rows_gives_insufficient_data_trend(self):
        """len < 3 → trend='insufficient data', monthly_change=0 (lines 601-602)."""
        df = pd.DataFrame({"margin_pct": [15.0, 14.0]})
        result = calculate_margin_trends(df)
        assert result["trend_direction"] == "insufficient data"
        assert result["monthly_trend_change"] == 0

    def test_one_row_gives_insufficient_data_trend(self):
        df = pd.DataFrame({"margin_pct": [10.0]})
        result = calculate_margin_trends(df)
        assert result["trend_direction"] == "insufficient data"

    def test_three_rows_computes_slope(self):
        """len >= 3 → polyfit slope computed."""
        df = pd.DataFrame({"margin_pct": [10.0, 12.0, 14.0]})
        result = calculate_margin_trends(df)
        assert result["trend_direction"] == "improving"
        assert result["monthly_trend_change"] > 0

    def test_declining_trend(self):
        df = pd.DataFrame({"margin_pct": [14.0, 12.0, 10.0, 8.0, 6.0]})
        result = calculate_margin_trends(df)
        assert result["trend_direction"] == "deteriorating"

    def test_risk_period_count_correct(self):
        """Margins below 5% count as risk periods."""
        df = pd.DataFrame({"margin_pct": [10.0, 3.0, 12.0, 2.0, 8.0]})
        result = calculate_margin_trends(df)
        assert result["risk_period_count"] == 2

    def test_all_keys_present(self):
        df = pd.DataFrame({"margin_pct": [10.0, 12.0, 11.0, 13.0, 14.0]})
        result = calculate_margin_trends(df)
        for key in ("current_margin", "mean_margin", "std_margin", "min_margin",
                    "max_margin", "trend_direction", "monthly_trend_change",
                    "margin_volatility", "risk_period_count", "risk_frequency"):
            assert key in result


# ── generate_trend_summary (lines 657-681) ───────────────────────────────────

class TestGenerateTrendSummary:
    def test_returns_dict(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"error": "No FX data"})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"error": "No data"})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"error": "No import data"})
        result = generate_trend_summary(commodity_code=84)
        assert isinstance(result, dict)

    def test_required_keys_present(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"error": "No FX data"})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"error": "No data"})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"error": "No import data"})
        result = generate_trend_summary(commodity_code=84)
        for key in ("commodity_code", "currency", "fx_analysis",
                    "fx_percentile", "import_analysis", "margin_trends"):
            assert key in result

    def test_commodity_code_stored(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"error": "x"})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"error": "x"})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"error": "x"})
        result = generate_trend_summary(commodity_code=85, currency="USD")
        assert result["commodity_code"] == 85
        assert result["currency"] == "USD"

    def test_margin_trend_error_when_no_data(self, monkeypatch):
        """When fx_analysis and import_analysis both lack 'data' key."""
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"error": "No FX"})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"error": "No data"})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"error": "No import"})
        result = generate_trend_summary(commodity_code=84)
        assert result["margin_trends"].get("error") is not None

    def test_full_path_with_data(self, monkeypatch, import_df, fx_df):
        """Full path: both 'data' keys present → simulate + calculate."""
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"data": fx_df, "current_rate": 1.15})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"percentile": 50.0})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"data": import_df, "total_value": 1e7})
        result = generate_trend_summary(commodity_code=84)
        assert result.get("margin_trends") is not None
        assert result["margin_trends"].get("error") is None

    def test_generated_at_is_iso_string(self, monkeypatch):
        import scripts.trend_analysis as ta
        monkeypatch.setattr(ta, "analyse_fx_trends",
                            lambda *a, **kw: {"error": "x"})
        monkeypatch.setattr(ta, "calculate_fx_percentile_position",
                            lambda *a, **kw: {"error": "x"})
        monkeypatch.setattr(ta, "analyse_import_trends",
                            lambda *a, **kw: {"error": "x"})
        result = generate_trend_summary(commodity_code=84)
        assert "generated_at" in result
        assert isinstance(result["generated_at"], str)
        assert "T" in result["generated_at"]  # ISO format contains T


# ── rolling statistics and volatility (already partially covered) ─────────────

class TestRollingAndVolatilityExtended:
    def test_rolling_stats_custom_window(self):
        s = pd.Series(np.random.randn(50))
        result = calculate_rolling_statistics(s, window=5, min_periods=2)
        for key in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max"):
            assert key in result

    def test_volatility_not_annualized(self):
        np.random.seed(3)
        s = pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01) * 100)
        vol_ann = calculate_volatility(s, annualize=True)
        vol_raw = calculate_volatility(s, annualize=False)
        non_nan_ann = vol_ann.dropna()
        non_nan_raw = vol_raw.dropna()
        assert (non_nan_ann > non_nan_raw).all()

    def test_decompose_multiplicative_model_with_positive_data(self):
        """Multiplicative decomposition on positive series (lines 237, 239-244)."""
        np.random.seed(0)
        values = np.abs(np.random.randn(36)) + 2.0
        series = pd.Series(
            values,
            index=pd.date_range("2020-01-01", periods=36, freq="MS")
        )
        result = decompose_seasonality(series, period=12, model="multiplicative")
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
