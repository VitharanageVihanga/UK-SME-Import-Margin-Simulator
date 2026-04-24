# forecast_engine.py
# ARIMA forecasting, anomaly detection, and forecast evaluation.
# Doesn't modify any core model logic.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# --- ARIMA helpers ---

def fit_arima_model(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
) -> Dict:
    """Fits an ARIMA (or SARIMA) model and returns the fitted object with AIC/BIC."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series_clean = series.dropna()

        if len(series_clean) < 20:
            return {"error": "Insufficient data for ARIMA (need at least 20 points)"}
        
        if seasonal_order:
            model = SARIMAX(
                series_clean,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(series_clean, order=order)
        
        fitted = model.fit()

        aic = fitted.aic
        bic = fitted.bic
        
        return {
            "model": fitted,
            "aic": round(aic, 2),
            "bic": round(bic, 2),
            "order": order,
            "seasonal_order": seasonal_order,
            "error": None
        }
        
    except ImportError:
        return {"error": "statsmodels not installed"}
    except Exception as e:
        return {"error": f"Model fitting failed: {str(e)}"}


def arima_forecast(
    series: pd.Series,
    steps: int = 30,
    order: Tuple[int, int, int] = (1, 1, 1),
    confidence_level: float = 0.95
) -> Dict:
    """Fits ARIMA and returns a forecast DataFrame with confidence intervals."""
    model_result = fit_arima_model(series, order=order)

    if model_result.get("error"):
        return model_result

    try:
        fitted = model_result["model"]

        forecast_result = fitted.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1 - confidence_level)

        # build date range for the forecast period
        if isinstance(series.index, pd.DatetimeIndex):
            last_date = series.index[-1]
            freq = pd.infer_freq(series.index) or 'D'
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
        else:
            forecast_dates = list(range(len(series), len(series) + steps))
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast_mean.values,
            "lower_bound": conf_int.iloc[:, 0].values,
            "upper_bound": conf_int.iloc[:, 1].values
        })
        
        return {
            "forecast": forecast_df,
            "model_info": {
                "aic": model_result["aic"],
                "bic": model_result["bic"],
                "order": order
            },
            "confidence_level": confidence_level,
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}


def auto_arima_forecast(
    series: pd.Series,
    steps: int = 30,
    max_p: int = 3,
    max_q: int = 3,
    max_d: int = 2
) -> Dict:
    """Grid searches ARIMA orders up to (max_p, max_d, max_q) and picks the lowest AIC."""
    best_aic = float('inf')
    best_order = (1, 1, 1)

    # brute force the order grid - not fast but good enough for these series sizes
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                
                result = fit_arima_model(series, order=(p, d, q))
                
                if not result.get("error") and result.get("aic", float('inf')) < best_aic:
                    best_aic = result["aic"]
                    best_order = (p, d, q)
    
    # Generate forecast with best order
    return arima_forecast(series, steps=steps, order=best_order)


# --- Prophet ---

def prophet_forecast(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    periods: int = 30,
    freq: str = "D",
    include_seasonality: bool = True
) -> Dict:
    """Forecasts using Facebook Prophet. Falls back gracefully if Prophet isn't installed."""
    try:
        from prophet import Prophet

        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 10:
            return {"error": "Insufficient data for Prophet (need at least 10 points)"}
        
        model = Prophet(
            yearly_seasonality=include_seasonality,
            weekly_seasonality=(freq == "D" and include_seasonality),
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_result.columns = ["date", "forecast", "lower_bound", "upper_bound"]

        historical_end = prophet_df["ds"].max()
        future_forecast = forecast_result[forecast_result["date"] > historical_end]

        # grab trend/seasonality components if we asked for them
        components = {}
        if include_seasonality:
            try:
                components["trend"] = forecast[["ds", "trend"]].copy()
                if "yearly" in forecast.columns:
                    components["yearly"] = forecast[["ds", "yearly"]].copy()
                if "weekly" in forecast.columns:
                    components["weekly"] = forecast[["ds", "weekly"]].copy()
            except Exception:
                pass
        
        return {
            "forecast": future_forecast,
            "full_forecast": forecast_result,
            "components": components,
            "model": model,
            "error": None
        }
        
    except ImportError:
        return {"error": "Prophet not installed. Install with: pip install prophet"}
    except Exception as e:
        return {"error": f"Prophet forecasting failed: {str(e)}"}


# --- FX forecasting ---

def forecast_fx_rate(
    currency: str = "EUR",
    forecast_days: int = 30,
    method: str = "arima",
    data_path: str = "data/raw/exchange"
) -> Dict:
    """Loads the exchange rate CSV for 'currency' and runs ARIMA or Prophet forecasting."""
    from scripts.trend_analysis import load_exchange_rates

    df = load_exchange_rates(currency, data_path)

    if df.empty:
        return {"error": f"No data available for {currency}"}
    
    if method == "arima":
        series = df.set_index("Date")["Rate"]
        result = auto_arima_forecast(series, steps=forecast_days)
    elif method == "prophet":
        result = prophet_forecast(
            df,
            date_col="Date",
            value_col="Rate",
            periods=forecast_days,
            freq="D"
        )
    else:
        return {"error": f"Unknown method: {method}"}
    
    if not result.get("error"):
        result["currency"] = currency
        result["method"] = method
        result["historical_data"] = df
    
    return result


# --- margin forecasting ---

def forecast_margins(
    margin_data: pd.DataFrame,
    forecast_periods: int = 6,
    method: str = "arima"
) -> Dict:
    """Projects margins forward using the same historical simulation output as input."""
    if margin_data.empty or "margin_pct" not in margin_data.columns:
        return {"error": "Invalid margin data"}
    
    if method == "arima":
        series = margin_data.set_index("date")["margin_pct"]
        result = auto_arima_forecast(series, steps=forecast_periods)
    elif method == "prophet":
        result = prophet_forecast(
            margin_data,
            date_col="date",
            value_col="margin_pct",
            periods=forecast_periods,
            freq="MS"
        )
    else:
        return {"error": f"Unknown method: {method}"}
    
    if not result.get("error"):
        result["metric"] = "margin_pct"
        result["method"] = method
    
    return result


# --- anomaly detection ---

def detect_anomalies_zscore(
    series: pd.Series,
    threshold: float = 2.5,
    window: Optional[int] = None
) -> pd.DataFrame:
    """Flags points where the z-score exceeds 'threshold'. Supports both global and rolling stats."""
    df = pd.DataFrame({"value": series})

    if window:
        df["mean"] = series.rolling(window=window, min_periods=1).mean()
        df["std"] = series.rolling(window=window, min_periods=1).std()
    else:
        # use global mean/std across the whole series
        df["mean"] = series.mean()
        df["std"] = series.std()
    
    df["zscore"] = (df["value"] - df["mean"]) / df["std"].replace(0, np.nan)

    df["is_anomaly"] = abs(df["zscore"]) > threshold
    df["anomaly_type"] = "normal"
    df.loc[df["zscore"] > threshold, "anomaly_type"] = "high"
    df.loc[df["zscore"] < -threshold, "anomaly_type"] = "low"
    
    return df


def detect_anomalies_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """IQR-based outlier detection. Less sensitive to extreme values than z-score."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    df = pd.DataFrame({
        "value": series,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    })
    
    df["is_anomaly"] = (df["value"] < lower_bound) | (df["value"] > upper_bound)
    df["anomaly_type"] = "normal"
    df.loc[df["value"] > upper_bound, "anomaly_type"] = "high"
    df.loc[df["value"] < lower_bound, "anomaly_type"] = "low"
    
    return df


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.1
) -> pd.DataFrame:
    """Isolation Forest anomaly detection. Works across multiple features at once."""
    try:
        from sklearn.ensemble import IsolationForest

        X = df[features].dropna()

        if len(X) < 10:
            return df.assign(is_anomaly=False, anomaly_score=0)

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = model.fit_predict(X)
        scores = model.decision_function(X)

        result = df.copy()
        result.loc[X.index, "is_anomaly"] = predictions == -1
        result.loc[X.index, "anomaly_score"] = -scores  # flip sign so higher = more anomalous
        result["is_anomaly"] = result["is_anomaly"].fillna(False)
        result["anomaly_score"] = result["anomaly_score"].fillna(0)

        return result
        
    except ImportError:
        return df.assign(is_anomaly=False, anomaly_score=0)


def detect_margin_anomalies(
    margin_data: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 2.0
) -> Dict:
    """Runs anomaly detection on margin_pct and returns flagged dates with plain-English explanations."""
    if margin_data.empty:
        return {"error": "No margin data provided"}
    
    # Detect anomalies in margin percentage
    if method == "zscore":
        result = detect_anomalies_zscore(margin_data["margin_pct"], threshold=threshold)
    elif method == "iqr":
        result = detect_anomalies_iqr(margin_data["margin_pct"], multiplier=threshold)
    elif method == "isolation_forest":
        features = ["margin_pct", "landed_cost", "fx_shock"]
        valid_features = [f for f in features if f in margin_data.columns]
        if valid_features:
            result = detect_anomalies_isolation_forest(margin_data, valid_features, contamination=0.1)
        else:
            return {"error": "Insufficient features for isolation forest"}
    else:
        return {"error": f"Unknown method: {method}"}
    
    # merge flags back into the original dataframe
    if method != "isolation_forest":
        margin_data = margin_data.copy()
        margin_data["is_anomaly"] = result["is_anomaly"].values
        margin_data["anomaly_type"] = result["anomaly_type"].values
    else:
        margin_data = result

    anomalies = margin_data[margin_data["is_anomaly"] == True]
    
    explanations = []
    for idx, row in anomalies.iterrows():
        if "anomaly_type" in row and row["anomaly_type"] == "high":
            explanations.append({
                "date": row.get("date", idx),
                "type": "Unusually high margin",
                "value": round(row["margin_pct"], 2),
                "explanation": "Margin significantly above historical average - verify revenue assumptions"
            })
        elif "anomaly_type" in row and row["anomaly_type"] == "low":
            explanations.append({
                "date": row.get("date", idx),
                "type": "Unusually low margin",
                "value": round(row["margin_pct"], 2),
                "explanation": "Margin significantly below historical average - investigate cost drivers"
            })
        else:
            explanations.append({
                "date": row.get("date", idx),
                "type": "Anomalous behaviour",
                "value": round(row["margin_pct"], 2),
                "explanation": "Unusual pattern detected - review input data"
            })
    
    return {
        "data": margin_data,
        "anomaly_count": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(margin_data) * 100, 1) if len(margin_data) > 0 else 0,
        "anomalies": explanations,
        "method": method,
        "error": None
    }


def detect_cost_anomalies(
    cost_data: pd.DataFrame,
    cost_column: str = "landed_cost",
    threshold: float = 2.0
) -> Dict:
    """Same as detect_margin_anomalies but applied to a cost column instead."""
    if cost_data.empty or cost_column not in cost_data.columns:
        return {"error": "Invalid cost data"}
    
    result = detect_anomalies_zscore(cost_data[cost_column], threshold=threshold, window=12)
    
    cost_data = cost_data.copy()
    cost_data["is_cost_anomaly"] = result["is_anomaly"].values
    cost_data["cost_anomaly_type"] = result["anomaly_type"].values
    cost_data["cost_zscore"] = result["zscore"].values
    
    anomalies = cost_data[cost_data["is_cost_anomaly"] == True]
    
    explanations = []
    for idx, row in anomalies.iterrows():
        if row["cost_anomaly_type"] == "high":
            explanations.append({
                "date": row.get("date", idx),
                "type": "Cost spike",
                "value": round(row[cost_column], 2),
                "zscore": round(row["cost_zscore"], 2),
                "explanation": "Landed cost significantly higher than normal - check FX and shipping rates"
            })
        else:
            explanations.append({
                "date": row.get("date", idx),
                "type": "Cost drop",
                "value": round(row[cost_column], 2),
                "zscore": round(row["cost_zscore"], 2),
                "explanation": "Landed cost unusually low - verify data accuracy"
            })
    
    return {
        "data": cost_data,
        "anomaly_count": len(anomalies),
        "anomalies": explanations,
        "error": None
    }


# --- forecast accuracy metrics ---

def evaluate_forecast_accuracy(
    actual: pd.Series,
    predicted: pd.Series
) -> Dict:
    """Returns MAE, RMSE, MAPE and directional accuracy for a set of predictions."""
    actual = actual.dropna()
    predicted = predicted.dropna()

    min_len = min(len(actual), len(predicted))
    actual = actual.iloc[:min_len]
    predicted = predicted.iloc[:min_len]

    if len(actual) == 0:
        return {"error": "No data for evaluation"}

    mae = np.mean(np.abs(actual.values - predicted.values))
    mse = np.mean((actual.values - predicted.values) ** 2)
    rmse = np.sqrt(mse)

    # MAPE - skip zeros to avoid division errors
    nonzero_mask = actual.values != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual.values[nonzero_mask] - predicted.values[nonzero_mask]) /
                              actual.values[nonzero_mask])) * 100
    else:
        mape = None

    # how often did we get the direction right?
    if len(actual) > 1:
        actual_direction = np.diff(actual.values) > 0
        predicted_direction = np.diff(predicted.values) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    else:
        directional_accuracy = None
    
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2) if mape else None,
        "directional_accuracy": round(directional_accuracy, 1) if directional_accuracy else None,
        "n_observations": len(actual)
    }


# --- full forecast report ---

def generate_forecast_report(
    commodity_code: int,
    currency: str = "EUR",
    forecast_horizon: int = 30,
    import_data_path: str = "data/output/merged_hmrc_ons_commodity.csv",
    fx_data_path: str = "data/raw/exchange"
) -> Dict:
    """Pulls FX forecast, margin forecast, and anomaly detection into one report dict."""
    from scripts.trend_analysis import (
        analyse_import_trends,
        analyse_fx_trends,
        simulate_historical_margins
    )

    fx_forecast = forecast_fx_rate(
        currency=currency,
        forecast_days=forecast_horizon,
        method="arima",
        data_path=fx_data_path
    )

    import_analysis = analyse_import_trends(commodity_code, import_data_path)
    fx_analysis = analyse_fx_trends(currency, data_path=fx_data_path)

    if "data" in import_analysis and "data" in fx_analysis:
        margin_sim = simulate_historical_margins(
            import_analysis["data"],
            fx_analysis["data"]
        )

        margin_forecast = forecast_margins(margin_sim, forecast_periods=6, method="arima")
        margin_anomalies = detect_margin_anomalies(margin_sim, method="zscore", threshold=2.0)
        cost_anomalies = detect_cost_anomalies(margin_sim, cost_column="landed_cost")
    else:
        margin_forecast = {"error": "Could not generate margin forecast"}
        margin_anomalies = {"error": "Could not detect margin anomalies"}
        cost_anomalies = {"error": "Could not detect cost anomalies"}
    
    return {
        "commodity_code": commodity_code,
        "currency": currency,
        "fx_forecast": fx_forecast,
        "margin_forecast": margin_forecast,
        "margin_anomalies": margin_anomalies,
        "cost_anomalies": cost_anomalies,
        "generated_at": datetime.now().isoformat()
    }
