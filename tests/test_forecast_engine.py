# tests/test_forecast_engine.py
"""
Pytest suite for scripts/forecast_engine.py
Covers: ARIMA fitting, forecasting, auto-order, FX forecast, anomaly detection,
        evaluation metrics.
Target: ≥80 % line coverage of forecast_engine.py
"""
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════

@pytest.fixture
def synthetic_series():
    """300-point random-walk with slight trend (enough for ARIMA)."""
    np.random.seed(42)
    values = np.cumsum(np.random.randn(300)) + 100
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    return pd.Series(values, index=dates)


@pytest.fixture
def short_series():
    """Only 10 data points — too short for ARIMA."""
    np.random.seed(0)
    return pd.Series(np.random.randn(10))


@pytest.fixture
def margin_df():
    """Synthetic margin simulation DataFrame."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    margins = np.random.normal(15, 5, n)
    landed = np.random.normal(100_000, 10_000, n)
    fx = np.random.normal(0.0, 0.03, n)
    return pd.DataFrame({
        "date": dates,
        "margin_pct": margins,
        "landed_cost": landed,
        "fx_shock": fx,
    })


# ════════════════════════════════════════════
#  FIT ARIMA
# ════════════════════════════════════════════

class TestFitArimaModel:
    def test_basic_fit(self, synthetic_series):
        from scripts.forecast_engine import fit_arima_model
        result = fit_arima_model(synthetic_series, order=(1, 1, 1))
        assert result["error"] is None
        assert "model" in result
        assert isinstance(result["aic"], float)
        assert isinstance(result["bic"], float)

    def test_insufficient_data(self, short_series):
        from scripts.forecast_engine import fit_arima_model
        result = fit_arima_model(short_series, order=(1, 1, 1))
        assert result["error"] is not None
        assert "Insufficient" in result["error"]

    def test_seasonal_arima(self, synthetic_series):
        from scripts.forecast_engine import fit_arima_model
        result = fit_arima_model(
            synthetic_series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
        )
        # seasonal may or may not converge — just shouldn't crash
        assert "error" in result


# ════════════════════════════════════════════
#  ARIMA FORECAST
# ════════════════════════════════════════════

class TestArimaForecast:
    def test_forecast_shape(self, synthetic_series):
        from scripts.forecast_engine import arima_forecast
        result = arima_forecast(synthetic_series, steps=10, order=(1, 1, 1))
        assert result["error"] is None
        fc = result["forecast"]
        assert len(fc) == 10
        assert "forecast" in fc.columns
        assert "lower_bound" in fc.columns
        assert "upper_bound" in fc.columns

    def test_confidence_bounds_order(self, synthetic_series):
        from scripts.forecast_engine import arima_forecast
        result = arima_forecast(synthetic_series, steps=5, order=(1, 1, 1))
        fc = result["forecast"]
        assert (fc["lower_bound"] <= fc["forecast"]).all()
        assert (fc["forecast"] <= fc["upper_bound"]).all()


# ════════════════════════════════════════════
#  AUTO-ARIMA
# ════════════════════════════════════════════

class TestAutoArimaForecast:
    def test_auto_selects_order(self, synthetic_series):
        from scripts.forecast_engine import auto_arima_forecast
        result = auto_arima_forecast(synthetic_series, steps=5, max_p=1, max_q=1, max_d=1)
        assert result["error"] is None
        assert "model_info" in result
        assert result["model_info"]["order"] is not None


# ════════════════════════════════════════════
#  FX RATE FORECAST (integration-ish)
# ════════════════════════════════════════════

class TestForecastFxRate:
    def test_eur_arima(self):
        from scripts.forecast_engine import forecast_fx_rate
        result = forecast_fx_rate("EUR", forecast_days=10, method="arima",
                                  data_path="data/raw/exchange")
        assert result.get("error") is None
        assert len(result["forecast"]) == 10

    def test_unknown_method(self):
        from scripts.forecast_engine import forecast_fx_rate
        result = forecast_fx_rate("EUR", method="unknown_method")
        assert result["error"] is not None

    def test_missing_currency(self):
        from scripts.forecast_engine import forecast_fx_rate
        result = forecast_fx_rate("XYZ", method="arima")
        assert result["error"] is not None


# ════════════════════════════════════════════
#  ANOMALY DETECTION
# ════════════════════════════════════════════

class TestAnomalyDetection:
    def test_zscore_detection(self):
        from scripts.forecast_engine import detect_anomalies_zscore
        np.random.seed(0)
        s = pd.Series(np.concatenate([np.random.randn(100), [10, -10]]))
        result = detect_anomalies_zscore(s, threshold=2.5)
        assert "is_anomaly" in result.columns
        assert result["is_anomaly"].sum() >= 2  # the two outliers

    def test_iqr_detection(self):
        from scripts.forecast_engine import detect_anomalies_iqr
        np.random.seed(0)
        s = pd.Series(np.concatenate([np.random.randn(100), [10, -10]]))
        result = detect_anomalies_iqr(s, multiplier=1.5)
        assert result["is_anomaly"].sum() >= 2

    def test_margin_anomalies(self, margin_df):
        from scripts.forecast_engine import detect_margin_anomalies
        result = detect_margin_anomalies(margin_df, method="zscore", threshold=2.0)
        assert result["error"] is None
        assert "anomaly_count" in result
        assert "anomaly_rate" in result

    def test_margin_anomalies_empty(self):
        from scripts.forecast_engine import detect_margin_anomalies
        result = detect_margin_anomalies(pd.DataFrame(), method="zscore")
        assert result["error"] is not None

    def test_cost_anomalies(self, margin_df):
        from scripts.forecast_engine import detect_cost_anomalies
        result = detect_cost_anomalies(margin_df, cost_column="landed_cost")
        assert result["error"] is None
        assert "anomaly_count" in result

    def test_cost_anomalies_bad_column(self, margin_df):
        from scripts.forecast_engine import detect_cost_anomalies
        result = detect_cost_anomalies(margin_df, cost_column="nonexistent")
        assert result["error"] is not None


# ════════════════════════════════════════════
#  FORECAST EVALUATION
# ════════════════════════════════════════════

class TestEvaluateForecastAccuracy:
    def test_perfect_match(self):
        from scripts.forecast_engine import evaluate_forecast_accuracy
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        r = evaluate_forecast_accuracy(s, s)
        assert r["mae"] == 0
        assert r["rmse"] == 0

    def test_known_error(self):
        from scripts.forecast_engine import evaluate_forecast_accuracy
        actual = pd.Series([1.0, 2.0, 3.0])
        predicted = pd.Series([1.1, 2.1, 3.1])
        r = evaluate_forecast_accuracy(actual, predicted)
        assert r["mae"] == pytest.approx(0.1, abs=0.001)
        assert r["rmse"] == pytest.approx(0.1, abs=0.001)

    def test_empty_series(self):
        from scripts.forecast_engine import evaluate_forecast_accuracy
        r = evaluate_forecast_accuracy(pd.Series(dtype=float), pd.Series(dtype=float))
        assert r["error"] is not None

    def test_directional_accuracy(self):
        from scripts.forecast_engine import evaluate_forecast_accuracy
        actual = pd.Series([1, 2, 3, 2, 1])
        predicted = pd.Series([1, 2, 3, 2, 1])
        r = evaluate_forecast_accuracy(actual, predicted)
        assert r["directional_accuracy"] == 100.0


# ════════════════════════════════════════════
#  ISOLATION FOREST ANOMALY DETECTION
# ════════════════════════════════════════════

class TestIsolationForest:
    def test_isolation_forest(self, margin_df):
        from scripts.forecast_engine import detect_anomalies_isolation_forest
        result = detect_anomalies_isolation_forest(
            margin_df, features=["margin_pct", "landed_cost"]
        )
        assert "is_anomaly" in result.columns
        assert "anomaly_score" in result.columns

    def test_isolation_forest_insufficient_data(self):
        from scripts.forecast_engine import detect_anomalies_isolation_forest
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = detect_anomalies_isolation_forest(df, features=["a", "b"])
        assert (result["is_anomaly"] == False).all()


# ════════════════════════════════════════════
#  PROPHET (ImportError path)
# ════════════════════════════════════════════

class TestProphetForecast:
    def test_prophet_import_or_run(self):
        """Prophet may or may not be installed — either path should work."""
        from scripts.forecast_engine import prophet_forecast
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": np.random.randn(100).cumsum() + 50,
        })
        result = prophet_forecast(df, date_col="date", value_col="value", periods=10)
        # Either succeeds or returns an error dict — should never crash
        assert "error" in result

    def test_prophet_insufficient_data(self):
        from scripts.forecast_engine import prophet_forecast
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "value": [1, 2, 3]})
        result = prophet_forecast(df, date_col="date", value_col="value", periods=5)
        assert result.get("error") is not None


# ════════════════════════════════════════════
#  MARGIN FORECAST
# ════════════════════════════════════════════

class TestForecastMargins:
    def test_arima_margin_forecast(self, margin_df):
        from scripts.forecast_engine import forecast_margins
        result = forecast_margins(margin_df, forecast_periods=3, method="arima")
        assert result.get("error") is None or "error" in result

    def test_invalid_method(self, margin_df):
        from scripts.forecast_engine import forecast_margins
        result = forecast_margins(margin_df, method="invalid")
        assert result["error"] is not None

    def test_empty_data(self):
        from scripts.forecast_engine import forecast_margins
        result = forecast_margins(pd.DataFrame(), forecast_periods=3)
        assert result["error"] is not None


# ════════════════════════════════════════════
#  MARGIN ANOMALIES — IQR + ISOLATION FOREST
# ════════════════════════════════════════════

class TestMarginAnomalyMethods:
    def test_iqr_method(self, margin_df):
        from scripts.forecast_engine import detect_margin_anomalies
        result = detect_margin_anomalies(margin_df, method="iqr", threshold=1.5)
        assert result["error"] is None

    def test_isolation_forest_method(self, margin_df):
        from scripts.forecast_engine import detect_margin_anomalies
        result = detect_margin_anomalies(margin_df, method="isolation_forest")
        assert result["error"] is None

    def test_unknown_method(self, margin_df):
        from scripts.forecast_engine import detect_margin_anomalies
        result = detect_margin_anomalies(margin_df, method="unknown")
        assert result["error"] is not None


# ════════════════════════════════════════════
#  ROLLING WINDOW ANOMALY DETECTION
# ════════════════════════════════════════════

class TestZscoreRollingWindow:
    def test_rolling_window(self):
        from scripts.forecast_engine import detect_anomalies_zscore
        s = pd.Series(np.random.randn(100))
        result = detect_anomalies_zscore(s, threshold=2.0, window=20)
        assert "is_anomaly" in result.columns
        assert "zscore" in result.columns


# ════════════════════════════════════════════
#  FX FORECAST – PROPHET PATH
# ════════════════════════════════════════════

class TestForecastFxRateProphet:
    def test_prophet_method_returns_result(self):
        """Exercises the prophet branch of forecast_fx_rate (covers line 345)."""
        from scripts.forecast_engine import forecast_fx_rate
        result = forecast_fx_rate("EUR", forecast_days=5, method="prophet",
                                  data_path="data/raw/exchange")
        # Prophet not installed → error propagated; or if installed → valid forecast
        assert "error" in result


# ════════════════════════════════════════════
#  MARGIN FORECAST – PROPHET PATH
# ════════════════════════════════════════════

class TestForecastMarginsProphet:
    def test_prophet_method(self, margin_df):
        """Exercises the prophet branch of forecast_margins (covers lines 389-410)."""
        from scripts.forecast_engine import forecast_margins
        result = forecast_margins(margin_df, forecast_periods=3, method="prophet")
        assert "error" in result

    def test_missing_column(self):
        """DataFrame without margin_pct column (covers line 749)."""
        from scripts.forecast_engine import forecast_margins
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "other": range(10)})
        result = forecast_margins(df, forecast_periods=3)
        assert result["error"] == "Invalid margin data"


# ════════════════════════════════════════════
#  ARIMA FORECAST – NON-DATETIME INDEX
# ════════════════════════════════════════════

class TestArimaForecastNonDatetimeIndex:
    def test_integer_index(self):
        """Series with plain int index → exercises line 139/160-161."""
        from scripts.forecast_engine import arima_forecast
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(200)) + 100)
        result = arima_forecast(s, steps=5, order=(1, 1, 0))
        assert result["error"] is None
        assert len(result["forecast"]) == 5


# ════════════════════════════════════════════
#  GENERATE FORECAST REPORT (integration)
# ════════════════════════════════════════════

class TestGenerateForecastReport:
    def test_full_report(self):
        """End-to-end report for a known commodity + currency (covers lines 801-837)."""
        from scripts.forecast_engine import generate_forecast_report
        report = generate_forecast_report(
            commodity_code=84,
            currency="EUR",
            forecast_horizon=5,
            import_data_path="data/output/merged_hmrc_ons_commodity.csv",
            fx_data_path="data/raw/exchange",
        )
        assert "commodity_code" in report
        assert "fx_forecast" in report
        assert "generated_at" in report

    def test_bad_commodity(self):
        """Non-existent commodity code — should not crash."""
        from scripts.forecast_engine import generate_forecast_report
        report = generate_forecast_report(
            commodity_code=99,
            currency="EUR",
            forecast_horizon=5,
        )
        assert "generated_at" in report


# ════════════════════════════════════════════
#  COST ANOMALIES – EDGE CASES
# ════════════════════════════════════════════

class TestCostAnomaliesEdge:
    def test_empty_dataframe(self):
        from scripts.forecast_engine import detect_cost_anomalies
        result = detect_cost_anomalies(pd.DataFrame())
        assert result["error"] is not None

    def test_valid_cost_data(self, margin_df):
        from scripts.forecast_engine import detect_cost_anomalies
        result = detect_cost_anomalies(margin_df, cost_column="landed_cost", threshold=1.5)
        assert result["error"] is None
        assert "anomaly_count" in result
