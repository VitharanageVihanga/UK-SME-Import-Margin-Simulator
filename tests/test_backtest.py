"""
Pytest suite for scripts/backtest.py
Covers: walk-forward backtest, order selection.
"""
import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture
def fx_series():
    np.random.seed(42)
    return pd.Series(np.cumsum(np.random.randn(400)) + 100)


class TestWalkForwardBacktest:
    def test_basic(self, fx_series):
        from scripts.backtest import walk_forward_backtest
        r = walk_forward_backtest(fx_series, order=(1, 1, 0), train_size=200, test_size=30, max_folds=2)
        assert r["error"] is None
        assert len(r["folds"]) == 2

    def test_insufficient_data(self):
        from scripts.backtest import walk_forward_backtest
        r = walk_forward_backtest(pd.Series(range(10)), train_size=200, test_size=30)
        assert r["error"] is not None


class TestSelectBestOrder:
    def test_selects_order(self, fx_series):
        from scripts.backtest import select_best_arima_order
        r = select_best_arima_order(
            fx_series,
            candidate_orders=[(1, 1, 0), (0, 1, 1)],
            train_size=200, test_size=30, max_folds=2,
        )
        assert r["error"] is None
        assert r["best_order"] in [(1, 1, 0), (0, 1, 1)]
