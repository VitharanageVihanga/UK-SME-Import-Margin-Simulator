# trend_analysis.py
# Loads and analyses historical FX and import data.
# Covers trend direction, volatility, seasonality, and margin simulation.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# --- data loading ---

def load_exchange_rates(currency: str = "EUR", data_path: str = "data/raw/exchange") -> pd.DataFrame:
    """Reads the BoE CSV for a given currency and returns a clean Date/Rate DataFrame."""
    import os

    file_path = os.path.join(data_path, f"exchange_{currency}.csv")

    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Date", "Rate"])

    df = pd.read_csv(file_path)

    # BoE files have long column names - just rename them
    df.columns = ["Date", "Rate"]

    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %y", errors="coerce")
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df


def aggregate_import_data(
    data_path: str = "data/output/merged_hmrc_ons_commodity.csv",
    commodity_filter: Optional[int] = None,
    country_filter: Optional[str] = None
) -> pd.DataFrame:
    """Reads the merged HMRC-ONS file in chunks and aggregates by year/month."""
    # chunked reading because these files can be large
    chunks = []
    
    try:
        for chunk in pd.read_csv(data_path, chunksize=100000):
            if commodity_filter is not None:
                chunk = chunk[chunk["hs2_chapter"] == commodity_filter]
            if country_filter is not None:
                chunk = chunk[chunk["country_code"] == country_filter]
            
            if len(chunk) > 0:
                agg = chunk.groupby(["year", "month"]).agg({
                    "value": "sum",
                    "net_mass": "sum"
                }).reset_index()
                chunks.append(agg)
    except FileNotFoundError:
        return pd.DataFrame(columns=["year", "month", "value", "net_mass"])
    
    if not chunks:
        return pd.DataFrame(columns=["year", "month", "value", "net_mass"])
    
    combined = pd.concat(chunks, ignore_index=True)

    result = combined.groupby(["year", "month"]).agg({
        "value": "sum",
        "net_mass": "sum"
    }).reset_index()

    # build a proper date column for easier plotting
    result["date"] = pd.to_datetime(
        result["year"].astype(str) + "-" + result["month"].astype(str).str.zfill(2) + "-01"
    )
    
    return result.sort_values("date").reset_index(drop=True)


# --- trend calculations ---

def calculate_rolling_statistics(
    series: pd.Series,
    window: int = 12,
    min_periods: int = 3
) -> Dict[str, pd.Series]:
    """Returns rolling mean, std, min, and max for a series."""
    return {
        "rolling_mean": series.rolling(window=window, min_periods=min_periods).mean(),
        "rolling_std": series.rolling(window=window, min_periods=min_periods).std(),
        "rolling_min": series.rolling(window=window, min_periods=min_periods).min(),
        "rolling_max": series.rolling(window=window, min_periods=min_periods).max(),
    }


def calculate_volatility(
    series: pd.Series,
    window: int = 30,
    annualize: bool = True
) -> pd.Series:
    """Annualised rolling volatility from log returns. Assumes daily data by default."""
    returns = np.log(series / series.shift(1))
    vol = returns.rolling(window=window, min_periods=5).std()

    if annualize:
        # 252 trading days in a year
        vol = vol * np.sqrt(252)
    
    return vol


def decompose_seasonality(
    series: pd.Series,
    period: int = 12,
    model: str = "additive"
) -> Dict[str, pd.Series]:
    """Seasonal decomposition via statsmodels. Returns trend, seasonal, and residual components."""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        series_clean = series.dropna()
        
        if len(series_clean) < 2 * period:
            return {
                "trend": pd.Series(dtype=float),
                "seasonal": pd.Series(dtype=float),
                "residual": pd.Series(dtype=float),
                "error": "Insufficient data for seasonal decomposition"
            }
        
        result = seasonal_decompose(series_clean, model=model, period=period)
        
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "error": None
        }
    except ImportError:
        return {
            "trend": pd.Series(dtype=float),
            "seasonal": pd.Series(dtype=float),
            "residual": pd.Series(dtype=float),
            "error": "statsmodels not installed"
        }
    except Exception as e:
        return {
            "trend": pd.Series(dtype=float),
            "seasonal": pd.Series(dtype=float),
            "residual": pd.Series(dtype=float),
            "error": str(e)
        }


def calculate_growth_rates(
    series: pd.Series,
    periods: List[int] = [1, 3, 6, 12]
) -> Dict[str, float]:
    """Returns percentage growth over each of the given look-back periods."""
    growth_rates = {}
    
    for period in periods:
        if len(series) > period:
            current = series.iloc[-1]
            previous = series.iloc[-period - 1]
            
            if previous != 0 and not pd.isna(previous) and not pd.isna(current):
                growth = ((current - previous) / abs(previous)) * 100
                growth_rates[f"{period}_period_growth"] = round(growth, 2)
            else:
                growth_rates[f"{period}_period_growth"] = None
        else:
            growth_rates[f"{period}_period_growth"] = None
    
    return growth_rates


# --- FX trend analysis ---

def analyse_fx_trends(
    currency: str = "EUR",
    lookback_days: int = 365,
    data_path: str = "data/raw/exchange"
) -> Dict:
    """Main function for FX analysis - stats, trend, volatility, growth rates."""
    df = load_exchange_rates(currency, data_path)

    if df.empty:
        return {"error": f"No data available for {currency}"}

    cutoff_date = df["Date"].max() - timedelta(days=lookback_days)
    df = df[df["Date"] >= cutoff_date].copy()

    if len(df) < 10:
        return {"error": "Insufficient data for analysis"}

    current_rate = df["Rate"].iloc[-1]
    mean_rate = df["Rate"].mean()
    std_rate = df["Rate"].std()
    min_rate = df["Rate"].min()
    max_rate = df["Rate"].max()

    volatility = calculate_volatility(df["Rate"], window=30)
    current_vol = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0

    # linear regression slope tells us the direction
    x = np.arange(len(df))
    slope, _ = np.polyfit(x, df["Rate"].values, 1)
    trend_direction = "strengthening" if slope > 0 else "weakening"

    growth_rates = calculate_growth_rates(df["Rate"], [7, 30, 90, 180])
    rolling_stats = calculate_rolling_statistics(df["Rate"], window=30)

    return {
        "currency": currency,
        "current_rate": round(current_rate, 4),
        "mean_rate": round(mean_rate, 4),
        "std_rate": round(std_rate, 4),
        "min_rate": round(min_rate, 4),
        "max_rate": round(max_rate, 4),
        "current_volatility": round(current_vol * 100, 2),
        "trend_direction": trend_direction,
        "trend_slope": round(slope * 1000, 4),
        "growth_rates": growth_rates,
        "data": df,
        "rolling_stats": rolling_stats,
        "volatility_series": volatility
    }


def calculate_fx_percentile_position(
    currency: str = "EUR",
    data_path: str = "data/raw/exchange"
) -> Dict:
    """Shows where the current rate sits in the full historical distribution (0-100th percentile)."""
    df = load_exchange_rates(currency, data_path)
    
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data"}
    
    current_rate = df["Rate"].iloc[-1]
    percentile = (df["Rate"] <= current_rate).mean() * 100
    
    # Interpretation
    if percentile <= 20:
        interpretation = "very weak (historically low)"
    elif percentile <= 40:
        interpretation = "weak (below average)"
    elif percentile <= 60:
        interpretation = "neutral (near average)"
    elif percentile <= 80:
        interpretation = "strong (above average)"
    else:
        interpretation = "very strong (historically high)"
    
    return {
        "currency": currency,
        "current_rate": round(current_rate, 4),
        "percentile": round(percentile, 1),
        "interpretation": interpretation,
        "historical_min": round(df["Rate"].min(), 4),
        "historical_max": round(df["Rate"].max(), 4),
        "historical_median": round(df["Rate"].median(), 4)
    }


# --- import trend analysis ---

def analyse_import_trends(
    commodity_code: Optional[int] = None,
    data_path: str = "data/output/merged_hmrc_ons_commodity.csv"
) -> Dict:
    """Summarises import trends for a commodity - totals, growth, seasonality, yearly breakdown."""
    df = aggregate_import_data(data_path, commodity_filter=commodity_code)

    if df.empty:
        return {"error": "No import data available"}

    total_value = df["value"].sum()
    mean_monthly = df["value"].mean()
    std_monthly = df["value"].std()

    growth_rates = calculate_growth_rates(df["value"])

    # need at least 2 years of data for seasonal decomposition
    if len(df) >= 24:
        df_indexed = df.set_index("date")["value"]
        seasonality = decompose_seasonality(df_indexed, period=12)
    else:
        seasonality = {"error": "Insufficient data for seasonality"}

    df["month_num"] = df["month"]
    monthly_pattern = df.groupby("month_num")["value"].mean().to_dict()
    yearly_totals = df.groupby("year")["value"].sum().to_dict()
    
    return {
        "commodity_code": commodity_code,
        "total_value": round(total_value, 2),
        "mean_monthly_value": round(mean_monthly, 2),
        "std_monthly_value": round(std_monthly, 2),
        "growth_rates": growth_rates,
        "seasonality": seasonality,
        "monthly_pattern": monthly_pattern,
        "yearly_totals": yearly_totals,
        "data": df
    }


# --- margin simulation ---

def simulate_historical_margins(
    import_data: pd.DataFrame,
    fx_data: pd.DataFrame,
    base_revenue_multiplier: float = 1.35,
    shipping_pct: float = 0.05,
    insurance_pct: float = 0.01,
    tariff_pct: float = 0.02
) -> pd.DataFrame:
    """Replays historical import/FX data through the margin model to simulate past margins."""
    if import_data.empty or fx_data.empty:
        return pd.DataFrame()

    # resample FX to monthly to match the import data frequency
    fx_monthly = fx_data.set_index("Date")["Rate"].resample("MS").mean().reset_index()
    fx_monthly.columns = ["date", "fx_rate"]

    merged = pd.merge(import_data, fx_monthly, on="date", how="left")
    merged["fx_rate"] = merged["fx_rate"].ffill().bfill()

    if merged["fx_rate"].isna().all():
        merged["fx_rate"] = 1.0  # fallback if no FX data

    # FX shock is relative to the first available rate
    baseline_fx = merged["fx_rate"].iloc[0]
    merged["fx_shock"] = (merged["fx_rate"] - baseline_fx) / baseline_fx
    
    # Simulate margins
    merged["import_value"] = merged["value"]
    merged["goods_cost"] = merged["import_value"] * (1 + merged["fx_shock"])
    merged["shipping_cost"] = merged["goods_cost"] * shipping_pct
    merged["insurance_cost"] = merged["goods_cost"] * insurance_pct
    merged["tariff_cost"] = merged["goods_cost"] * tariff_pct
    merged["landed_cost"] = (
        merged["goods_cost"] + 
        merged["shipping_cost"] + 
        merged["insurance_cost"] + 
        merged["tariff_cost"]
    )
    
    merged["revenue"] = merged["import_value"] * base_revenue_multiplier
    merged["profit"] = merged["revenue"] - merged["landed_cost"]
    merged["margin_pct"] = (merged["profit"] / merged["revenue"]) * 100
    
    return merged


def calculate_margin_trends(
    margin_data: pd.DataFrame
) -> Dict:
    """Summarises the margin simulation output - mean, range, trend direction, risk periods."""
    if margin_data.empty:
        return {"error": "No margin data available"}

    current_margin = margin_data["margin_pct"].iloc[-1]
    mean_margin = margin_data["margin_pct"].mean()
    std_margin = margin_data["margin_pct"].std()
    min_margin = margin_data["margin_pct"].min()
    max_margin = margin_data["margin_pct"].max()

    if len(margin_data) >= 3:
        x = np.arange(len(margin_data))
        slope, _ = np.polyfit(x, margin_data["margin_pct"].values, 1)
        trend = "improving" if slope > 0 else "deteriorating"
        monthly_change = slope
    else:
        trend = "insufficient data"
        monthly_change = 0

    margin_vol = margin_data["margin_pct"].std()

    # count months where margin dropped below 5% as a risk indicator
    risk_periods = (margin_data["margin_pct"] < 5).sum()
    total_periods = len(margin_data)
    risk_frequency = risk_periods / total_periods if total_periods > 0 else 0
    
    return {
        "current_margin": round(current_margin, 2),
        "mean_margin": round(mean_margin, 2),
        "std_margin": round(std_margin, 2),
        "min_margin": round(min_margin, 2),
        "max_margin": round(max_margin, 2),
        "trend_direction": trend,
        "monthly_trend_change": round(monthly_change, 4),
        "margin_volatility": round(margin_vol, 2),
        "risk_period_count": risk_periods,
        "risk_frequency": round(risk_frequency * 100, 1),
        "data": margin_data
    }


# --- summary wrapper ---

def generate_trend_summary(
    commodity_code: int,
    currency: str = "EUR",
    import_data_path: str = "data/output/merged_hmrc_ons_commodity.csv",
    fx_data_path: str = "data/raw/exchange"
) -> Dict:
    """Convenience wrapper that runs all trend analyses and returns them in one dict."""
    fx_analysis = analyse_fx_trends(currency, data_path=fx_data_path)
    fx_percentile = calculate_fx_percentile_position(currency, data_path=fx_data_path)

    import_analysis = analyse_import_trends(commodity_code, data_path=import_data_path)

    if "data" in import_analysis and "data" in fx_analysis:
        margin_sim = simulate_historical_margins(
            import_analysis["data"],
            fx_analysis["data"]
        )
        margin_trends = calculate_margin_trends(margin_sim)
    else:
        margin_trends = {"error": "Could not simulate margins"}
    
    return {
        "commodity_code": commodity_code,
        "currency": currency,
        "fx_analysis": fx_analysis,
        "fx_percentile": fx_percentile,
        "import_analysis": import_analysis,
        "margin_trends": margin_trends,
        "generated_at": datetime.now().isoformat()
    }
