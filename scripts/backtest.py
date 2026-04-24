# backtest.py
"""
Backtesting Module for Forecast Validation

Provides walk-forward backtesting to validate ARIMA forecast accuracy
against held-out historical data. Reports MAE, RMSE, MAPE, and directional
accuracy so the analyst can judge whether the default ARIMA order is adequate.

Usage
-----
    python -m scripts.backtest          # quick summary
    pytest tests/test_backtest.py       # as part of test suite
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ======================
# WALK-FORWARD BACKTEST
# ======================

def walk_forward_backtest(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    train_size: int = 200,
    test_size: int = 30,
    step: int = 30,
    max_folds: int = 5,
) -> Dict:
    """
    Walk-forward validation for ARIMA forecasts.

    Splits *series* into expanding training windows, forecasts
    *test_size* steps ahead, and compares against actuals.

    Parameters
    ----------
    series : pd.Series
        Historical time-series (e.g. FX rates).
    order : tuple
        ARIMA (p, d, q) order to evaluate.
    train_size : int
        Minimum number of observations for the first training window.
    test_size : int
        Forecast horizon per fold (days / periods).
    step : int
        Stride between successive folds.
    max_folds : int
        Maximum number of folds to evaluate.

    Returns
    -------
    dict
        Per-fold and aggregate accuracy metrics.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series_clean = series.dropna()
    n = len(series_clean)

    if n < train_size + test_size:
        return {"error": f"Need at least {train_size + test_size} points, got {n}"}

    fold_results: List[Dict] = []
    fold_idx = 0

    start = train_size
    while start + test_size <= n and fold_idx < max_folds:
        train = series_clean.iloc[:start]
        actual = series_clean.iloc[start : start + test_size]

        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            fc = fitted.get_forecast(steps=test_size)
            predicted = fc.predicted_mean.values
        except Exception as exc:
            fold_results.append({"fold": fold_idx + 1, "error": str(exc)})
            start += step
            fold_idx += 1
            continue

        actual_vals = actual.values
        errors = actual_vals - predicted

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        nonzero = actual_vals != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / actual_vals[nonzero])) * 100)
            if nonzero.any()
            else None
        )
        if len(actual_vals) > 1:
            dir_actual = np.diff(actual_vals) > 0
            dir_pred = np.diff(predicted) > 0
            dir_acc = float(np.mean(dir_actual == dir_pred) * 100)
        else:
            dir_acc = None

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "train_end": start,
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "mape": round(mape, 2) if mape is not None else None,
                "directional_accuracy": round(dir_acc, 1) if dir_acc is not None else None,
                "error": None,
            }
        )

        start += step
        fold_idx += 1

    if not fold_results:
        return {"error": "No folds could be evaluated"}

    # Aggregate across valid folds
    valid = [f for f in fold_results if f.get("error") is None]
    if not valid:
        return {
            "folds": fold_results,
            "error": "All folds failed",
        }

    agg = {
        "avg_mae": round(np.mean([f["mae"] for f in valid]), 6),
        "avg_rmse": round(np.mean([f["rmse"] for f in valid]), 6),
        "avg_mape": round(np.mean([f["mape"] for f in valid if f["mape"] is not None]), 2)
        if any(f["mape"] is not None for f in valid)
        else None,
        "avg_directional_accuracy": round(
            np.mean([f["directional_accuracy"] for f in valid if f["directional_accuracy"] is not None]), 1
        )
        if any(f["directional_accuracy"] is not None for f in valid)
        else None,
        "n_folds": len(valid),
    }

    return {
        "folds": fold_results,
        "aggregate": agg,
        "order": order,
        "error": None,
    }


# ======================
# ORDER SELECTION
# ======================

def select_best_arima_order(
    series: pd.Series,
    candidate_orders: Optional[List[Tuple[int, int, int]]] = None,
    train_size: int = 200,
    test_size: int = 30,
    max_folds: int = 3,
) -> Dict:
    """
    Compare multiple ARIMA orders via walk-forward backtest and return the
    best one (lowest average RMSE).

    Parameters
    ----------
    series : pd.Series
        Historical time-series.
    candidate_orders : list of tuple, optional
        Orders to evaluate.  Defaults to a sensible grid.
    train_size, test_size, max_folds : int
        Forwarded to ``walk_forward_backtest``.

    Returns
    -------
    dict
        Best order, all results, and recommendation.
    """
    if candidate_orders is None:
        candidate_orders = [
            (1, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
            (1, 0, 1),
            (0, 1, 1),
        ]

    results = {}
    for order in candidate_orders:
        bt = walk_forward_backtest(
            series,
            order=order,
            train_size=train_size,
            test_size=test_size,
            max_folds=max_folds,
        )
        results[order] = bt

    # Pick best by RMSE (lower is better)
    scored = {
        o: r["aggregate"]["avg_rmse"]
        for o, r in results.items()
        if r.get("aggregate") and r["aggregate"].get("avg_rmse") is not None
    }

    if not scored:
        return {"error": "All candidate orders failed backtesting", "results": results}

    best_order = min(scored, key=scored.get)
    best_rmse = scored[best_order]

    recommendation = (
        f"Best ARIMA order {best_order} with avg RMSE {best_rmse:.6f}. "
        f"Tested {len(scored)}/{len(candidate_orders)} orders successfully."
    )

    return {
        "best_order": best_order,
        "best_rmse": best_rmse,
        "recommendation": recommendation,
        "all_results": {str(k): v for k, v in results.items()},
        "error": None,
    }


# ======================
# CONVENIENCE: BACKTEST FX
# ======================

def backtest_fx_forecast(
    currency: str = "EUR",
    data_path: str = "data/raw/exchange",
    orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict:
    """
    End-to-end backtest of FX forecasting for a given currency.

    Loads the exchange-rate CSV, runs order selection, and returns a
    human-readable report.
    """
    from scripts.trend_analysis import load_exchange_rates

    df = load_exchange_rates(currency, data_path)
    if df.empty:
        return {"error": f"No exchange-rate data for {currency}"}

    series = df.set_index("Date")["Rate"]

    result = select_best_arima_order(series, candidate_orders=orders)
    if result.get("error"):
        return result

    result["currency"] = currency
    result["data_points"] = len(series)
    return result


# ======================
# CLI ENTRY POINT
# ======================

if __name__ == "__main__":
    for ccy in ("EUR", "USD", "CNY", "JPY"):
        print(f"\n{'='*60}")
        print(f"BACKTESTING FX FORECAST — {ccy}")
        print("=" * 60)
        report = backtest_fx_forecast(ccy)
        if report.get("error"):
            print(f"  ERROR: {report['error']}")
        else:
            print(f"  Data points : {report['data_points']}")
            print(f"  Best order  : {report['best_order']}")
            print(f"  Avg RMSE    : {report['best_rmse']:.6f}")
            print(f"  {report['recommendation']}")
