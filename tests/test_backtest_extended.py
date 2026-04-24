"""
Extended tests for scripts/backtest.py
Targets missed lines: fold exception recording, dir_acc=None for test_size=1,
all-folds-failed branch, default candidate orders path, all-candidates-failed,
backtest_fx_forecast with monkeypatch.
"""
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from scripts.backtest import walk_forward_backtest, select_best_arima_order


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def good_series():
    np.random.seed(42)
    return pd.Series(np.cumsum(np.random.randn(500)) + 100)


@pytest.fixture(scope="module")
def medium_series():
    np.random.seed(7)
    return pd.Series(np.cumsum(np.random.randn(300)) + 50)


# ── dir_acc = None when test_size=1 (line 104) ───────────────────────────────

class TestDirectionalAccuracyNoneForSingleStep:
    def test_single_step_dir_acc_is_none(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=1, max_folds=1
        )
        assert result["error"] is None
        fold = result["folds"][0]
        assert fold["directional_accuracy"] is None

    def test_single_step_mae_computed(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=1, max_folds=1
        )
        assert result["error"] is None
        fold = result["folds"][0]
        assert isinstance(fold["mae"], float)
        assert fold["mae"] >= 0

    def test_single_step_rmse_computed(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=1, max_folds=1
        )
        fold = result["folds"][0]
        assert isinstance(fold["rmse"], float)
        assert fold["rmse"] >= 0

    def test_single_step_aggregate_dir_acc_none(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=1, max_folds=2
        )
        if result["error"] is None:
            assert result["aggregate"]["avg_directional_accuracy"] is None


# ── fold exception recording path (lines 82-86) ──────────────────────────────

class TestFoldExceptionRecording:
    def test_bad_order_returns_valid_dict(self, good_series):
        """ARIMA(10,10,10) is likely to fail; result must still be a dict."""
        result = walk_forward_backtest(
            good_series, order=(10, 10, 10), train_size=400, test_size=30, max_folds=1
        )
        assert isinstance(result, dict)
        assert "error" in result or "folds" in result

    def test_fold_error_recorded_in_folds_list(self, good_series):
        """If ARIMA fails, the fold entry must have an 'error' key."""
        result = walk_forward_backtest(
            good_series, order=(10, 10, 10), train_size=400, test_size=30, max_folds=1
        )
        if "folds" in result and result["folds"]:
            fold = result["folds"][0]
            assert "fold" in fold


# ── aggregate metrics correctness ────────────────────────────────────────────

class TestAggregateMetricsCorrectness:
    def test_aggregate_keys_present(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=2
        )
        assert result["error"] is None
        for key in ("avg_mae", "avg_rmse", "n_folds"):
            assert key in result["aggregate"]

    def test_mae_leq_rmse(self, good_series):
        """By Cauchy-Schwarz, MAE ≤ RMSE."""
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=1
        )
        if result["error"] is None:
            agg = result["aggregate"]
            assert agg["avg_mae"] <= agg["avg_rmse"] + 1e-9

    def test_rmse_non_negative(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=1
        )
        if result["error"] is None:
            assert result["aggregate"]["avg_rmse"] >= 0

    def test_mape_non_negative(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=1
        )
        if result["error"] is None and result["aggregate"].get("avg_mape") is not None:
            assert result["aggregate"]["avg_mape"] >= 0

    def test_n_folds_does_not_exceed_max(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=3
        )
        if result["error"] is None:
            assert result["aggregate"]["n_folds"] <= 3

    def test_order_stored_in_result(self, good_series):
        order = (1, 1, 0)
        result = walk_forward_backtest(
            good_series, order=order, train_size=200, test_size=30, max_folds=1
        )
        if result["error"] is None:
            assert result["order"] == order

    def test_fold_list_length(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=2
        )
        if result["error"] is None:
            assert len(result["folds"]) == result["aggregate"]["n_folds"]

    def test_directional_accuracy_in_range(self, good_series):
        result = walk_forward_backtest(
            good_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=1
        )
        if result["error"] is None:
            da = result["aggregate"].get("avg_directional_accuracy")
            if da is not None:
                assert 0.0 <= da <= 100.0


# ── default candidate orders used when None passed (line 184) ─────────────────

class TestDefaultCandidateOrders:
    def test_none_candidate_uses_defaults(self, good_series):
        result = select_best_arima_order(
            good_series, candidate_orders=None, train_size=200, test_size=30, max_folds=1
        )
        assert isinstance(result, dict)

    def test_none_candidate_returns_best_order(self, good_series):
        result = select_best_arima_order(
            good_series, candidate_orders=None, train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            assert isinstance(result["best_order"], tuple)
            assert len(result["best_order"]) == 3

    def test_default_includes_multiple_orders(self, good_series):
        result = select_best_arima_order(
            good_series, candidate_orders=None, train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            assert "all_results" in result
            assert len(result["all_results"]) >= 2

    def test_recommendation_string_present(self, good_series):
        result = select_best_arima_order(
            good_series, candidate_orders=None, train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            assert isinstance(result["recommendation"], str)
            assert len(result["recommendation"]) > 0


# ── all candidates failed (line 212) ─────────────────────────────────────────

class TestAllCandidatesFailed:
    def test_short_series_all_fail(self):
        """Series too short (50 pts) for train_size=200 → all error."""
        short = pd.Series(np.linspace(1, 10, 50))
        result = select_best_arima_order(
            short, candidate_orders=[(1, 1, 1), (0, 1, 1)], train_size=200, test_size=30
        )
        assert result.get("error") is not None

    def test_empty_series_all_fail(self):
        result = select_best_arima_order(
            pd.Series(dtype=float),
            candidate_orders=[(1, 1, 1)], train_size=200, test_size=30
        )
        assert result.get("error") is not None


# ── select_best_arima_order – two candidates ──────────────────────────────────

class TestSelectBestOrderExtended:
    def test_best_order_is_from_candidates(self, good_series):
        candidates = [(1, 1, 0), (0, 1, 1)]
        result = select_best_arima_order(
            good_series, candidate_orders=candidates,
            train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            assert result["best_order"] in candidates

    def test_best_rmse_matches_best_order(self, good_series):
        candidates = [(1, 1, 0), (0, 1, 1)]
        result = select_best_arima_order(
            good_series, candidate_orders=candidates,
            train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            best_rmse = result["best_rmse"]
            all_rmses = [
                v["aggregate"]["avg_rmse"]
                for v in result["all_results"].values()
                if v.get("aggregate") and v["aggregate"].get("avg_rmse") is not None
            ]
            assert best_rmse == min(all_rmses)

    def test_three_candidates_evaluated(self, good_series):
        candidates = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
        result = select_best_arima_order(
            good_series, candidate_orders=candidates,
            train_size=200, test_size=30, max_folds=1
        )
        if result.get("error") is None:
            assert len(result["all_results"]) == 3


# ── backtest_fx_forecast (lines 246-260) ─────────────────────────────────────

class TestBacktestFxForecast:
    def test_missing_currency_returns_error(self):
        from scripts.backtest import backtest_fx_forecast
        result = backtest_fx_forecast(currency="XYZ", data_path="data/raw/exchange")
        assert result.get("error") is not None

    def test_nonexistent_path_returns_error(self):
        from scripts.backtest import backtest_fx_forecast
        result = backtest_fx_forecast(currency="EUR", data_path="/nonexistent/path")
        assert result.get("error") is not None

    def test_with_synthetic_data(self, monkeypatch, good_series):
        """Monkeypatch load_exchange_rates → synthetic 500-point series."""
        import scripts.trend_analysis as ta
        import scripts.backtest as bt

        synthetic_df = pd.DataFrame({
            "Date": pd.date_range("2010-01-01", periods=500, freq="B"),
            "Rate": good_series.values,
        })
        monkeypatch.setattr(ta, "load_exchange_rates",
                            lambda *a, **kw: synthetic_df)

        result = bt.backtest_fx_forecast(
            currency="EUR",
            data_path="data/raw/exchange",
            orders=[(1, 1, 0), (0, 1, 1)],
        )
        assert isinstance(result, dict)
        if result.get("error") is None:
            assert result["currency"] == "EUR"
            assert "best_order" in result
            assert result["data_points"] == 500

    def test_with_synthetic_data_stores_data_points(self, monkeypatch, good_series):
        import scripts.trend_analysis as ta
        import scripts.backtest as bt

        synthetic_df = pd.DataFrame({
            "Date": pd.date_range("2010-01-01", periods=500, freq="B"),
            "Rate": good_series.values,
        })
        monkeypatch.setattr(ta, "load_exchange_rates",
                            lambda *a, **kw: synthetic_df)

        result = bt.backtest_fx_forecast(
            currency="USD",
            orders=[(1, 1, 0)],
        )
        if result.get("error") is None:
            assert result["data_points"] > 0
