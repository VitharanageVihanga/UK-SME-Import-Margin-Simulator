# advanced_risk_metrics.py
# VaR, volatility, correlation, stress testing, and risk decomposition.
# All standalone - nothing here touches the core margin or ONS coverage logic.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# --- value at risk ---

def calculate_var_historical(
    returns: pd.Series,
    confidence_level: float = 0.95,
    holding_period: int = 1
) -> Dict:
    """Historical simulation VaR. Returns the loss threshold at the given confidence level."""
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return {"error": "Insufficient data for VaR calculation (need at least 30 observations)"}
    
    # scale by sqrt of time (standard approach)
    scaled_returns = returns_clean * np.sqrt(holding_period)

    var_percentile = 1 - confidence_level
    var_value = np.percentile(scaled_returns, var_percentile * 100)

    # expected shortfall = average loss in the tail beyond VaR
    es_value = scaled_returns[scaled_returns <= var_value].mean()

    mean_return = returns_clean.mean()
    std_return = returns_clean.std()
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()

    if abs(var_value) < 0.02:
        risk_level = "LOW"
        interpretation = "Limited potential for significant losses"
    elif abs(var_value) < 0.05:
        risk_level = "MODERATE"
        interpretation = "Moderate potential for losses under adverse conditions"
    else:
        risk_level = "HIGH"
        interpretation = "Significant potential for losses - consider risk mitigation"
    
    return {
        "var_value": round(var_value * 100, 2),  # As percentage
        "var_interpretation": f"With {confidence_level*100:.0f}% confidence, daily loss will not exceed {abs(var_value)*100:.2f}%",
        "expected_shortfall": round(es_value * 100, 2) if not np.isnan(es_value) else None,
        "es_interpretation": f"Average loss when VaR is breached: {abs(es_value)*100:.2f}%" if not np.isnan(es_value) else "N/A",
        "confidence_level": confidence_level,
        "holding_period": holding_period,
        "risk_level": risk_level,
        "interpretation": interpretation,
        "statistics": {
            "mean_return": round(mean_return * 100, 4),
            "std_return": round(std_return * 100, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "n_observations": len(returns_clean)
        },
        "error": None
    }


def calculate_var_parametric(
    returns: pd.Series,
    confidence_level: float = 0.95,
    holding_period: int = 1
) -> Dict:
    """Parametric (variance-covariance) VaR. Assumes returns are normally distributed."""
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return {"error": "Insufficient data for VaR calculation"}
    
    mu = returns_clean.mean()
    sigma = returns_clean.std()
    z_score = stats.norm.ppf(1 - confidence_level)

    var_value = mu + z_score * sigma * np.sqrt(holding_period)
    es_value = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence_level) * np.sqrt(holding_period)
    
    return {
        "var_value": round(var_value * 100, 2),
        "expected_shortfall": round(es_value * 100, 2),
        "method": "parametric",
        "assumptions": "Assumes returns are normally distributed",
        "mu": round(mu * 100, 4),
        "sigma": round(sigma * 100, 4),
        "z_score": round(z_score, 4),
        "error": None
    }


def calculate_var_montecarlo(
    returns: pd.Series,
    confidence_level: float = 0.95,
    holding_period: int = 1,
    simulations: int = 10000
) -> Dict:
    """Monte Carlo VaR - runs 'simulations' paths and picks the percentile loss."""
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 30:
        return {"error": "Insufficient data for Monte Carlo VaR"}
    
    mu = returns_clean.mean()
    sigma = returns_clean.std()

    # fix seed so results are reproducible
    np.random.seed(42)
    simulated_returns = np.random.normal(mu, sigma, (simulations, holding_period))
    portfolio_returns = simulated_returns.sum(axis=1)

    var_percentile = (1 - confidence_level) * 100
    var_value = np.percentile(portfolio_returns, var_percentile)
    es_value = portfolio_returns[portfolio_returns <= var_value].mean()
    
    return {
        "var_value": round(var_value * 100, 2),
        "expected_shortfall": round(es_value * 100, 2),
        "method": "monte_carlo",
        "simulations": simulations,
        "confidence_interval_95": [
            round(np.percentile(portfolio_returns, 2.5) * 100, 2),
            round(np.percentile(portfolio_returns, 97.5) * 100, 2)
        ],
        "error": None
    }


def calculate_margin_var(
    margin_data: pd.DataFrame,
    confidence_level: float = 0.95,
    value_at_risk_horizon: int = 30
) -> Dict:
    """Calculates VaR on margin percentages rather than raw returns."""
    if margin_data.empty or "margin_pct" not in margin_data.columns:
        return {"error": "Invalid margin data"}
    
    margin_returns = margin_data["margin_pct"].pct_change().dropna()

    if len(margin_returns) < 10:
        # not enough data for pct change - fall back to level differences
        margin_changes = margin_data["margin_pct"].diff().dropna()
        if len(margin_changes) < 10:
            return {"error": "Insufficient data for margin VaR"}

        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(margin_changes, var_percentile)
        current_margin = margin_data["margin_pct"].iloc[-1]
        
        return {
            "current_margin": round(current_margin, 2),
            "margin_var": round(var_value, 2),
            "worst_case_margin": round(current_margin + var_value, 2),
            "interpretation": f"With {confidence_level*100:.0f}% confidence, margin won't fall by more than {abs(var_value):.2f}pp",
            "method": "level_changes",
            "error": None
        }
    
    # Calculate VaR on returns
    var_result = calculate_var_historical(margin_returns, confidence_level)
    
    if var_result.get("error"):
        return var_result
    
    current_margin = margin_data["margin_pct"].iloc[-1]
    var_pct = var_result["var_value"] / 100
    worst_case_margin = current_margin * (1 + var_pct)
    
    return {
        "current_margin": round(current_margin, 2),
        "margin_var_pct": var_result["var_value"],
        "worst_case_margin": round(worst_case_margin, 2),
        "risk_level": var_result["risk_level"],
        "interpretation": var_result["interpretation"],
        "statistics": var_result["statistics"],
        "error": None
    }


# --- volatility ---

def calculate_historical_volatility(
    series: pd.Series,
    window: int = 30,
    annualization_factor: int = 252
) -> pd.DataFrame:
    """Computes rolling, EWMA, and Parkinson volatility estimates for a price series."""
    log_returns = np.log(series / series.shift(1))

    rolling_vol = log_returns.rolling(window=window, min_periods=5).std() * np.sqrt(annualization_factor)
    ewma_vol = log_returns.ewm(span=window).std() * np.sqrt(annualization_factor)

    # Parkinson uses the high-low range; we approximate with rolling max/min of log returns
    range_proxy = log_returns.rolling(window=5).max() - log_returns.rolling(window=5).min()
    parkinson_vol = range_proxy.rolling(window=window).mean() * np.sqrt(annualization_factor) / (2 * np.sqrt(np.log(2)))
    
    result = pd.DataFrame({
        "value": series,
        "log_return": log_returns,
        "rolling_vol": rolling_vol,
        "ewma_vol": ewma_vol,
        "parkinson_vol": parkinson_vol
    })
    
    return result


def volatility_regime_detection(
    volatility_series: pd.Series,
    low_threshold: float = 0.10,
    high_threshold: float = 0.20
) -> Dict:
    """Labels the current vol environment as LOW / NORMAL / HIGH based on thresholds."""
    vol_clean = volatility_series.dropna()
    
    if len(vol_clean) == 0:
        return {"error": "No volatility data"}
    
    current_vol = vol_clean.iloc[-1]
    mean_vol = vol_clean.mean()
    percentile = (vol_clean <= current_vol).mean() * 100
    
    if current_vol < low_threshold:
        regime = "LOW"
        regime_description = "Market conditions are calm with low volatility"
        trading_implication = "Good conditions for standard import planning"
    elif current_vol > high_threshold:
        regime = "HIGH"
        regime_description = "Market conditions are volatile with elevated uncertainty"
        trading_implication = "Consider hedging strategies or delayed commitments"
    else:
        regime = "NORMAL"
        regime_description = "Market conditions are within normal volatility range"
        trading_implication = "Standard risk management approaches apply"

    # look at distribution of past regimes for context
    regime_series = pd.cut(
        vol_clean,
        bins=[0, low_threshold, high_threshold, float('inf')],
        labels=['LOW', 'NORMAL', 'HIGH']
    )
    regime_distribution = regime_series.value_counts(normalize=True).to_dict()
    
    return {
        "current_volatility": round(current_vol * 100, 2),
        "mean_volatility": round(mean_vol * 100, 2),
        "percentile": round(percentile, 1),
        "current_regime": regime,
        "regime_description": regime_description,
        "trading_implication": trading_implication,
        "regime_distribution": {k: round(v * 100, 1) for k, v in regime_distribution.items()},
        "error": None
    }


# --- correlation ---

def calculate_correlation_matrix(
    data_dict: Dict[str, pd.Series],
    method: str = "pearson"
) -> pd.DataFrame:
    """Builds a correlation matrix from a dict of named series."""
    # align on index first so we're comparing the same dates
    df = pd.DataFrame(data_dict)
    df = df.dropna()
    
    if len(df) < 10:
        return pd.DataFrame()
    
    return df.corr(method=method)


def analyse_commodity_fx_correlation(
    import_data: pd.DataFrame,
    fx_data: pd.DataFrame,
    window: int = 12
) -> Dict:
    """Checks how much import volumes and FX moves track each other over time."""
    if import_data.empty or fx_data.empty:
        return {"error": "Insufficient data for correlation analysis"}

    # resample FX to monthly so it lines up with the import data
    fx_monthly = fx_data.set_index("Date")["Rate"].resample("MS").mean()
    import_monthly = import_data.set_index("date")["value"]

    aligned = pd.DataFrame({
        "import_value": import_monthly,
        "fx_rate": fx_monthly
    }).dropna()

    if len(aligned) < 10:
        return {"error": "Insufficient aligned data points"}

    aligned["import_change"] = aligned["import_value"].pct_change()
    aligned["fx_change"] = aligned["fx_rate"].pct_change()

    static_corr = aligned["import_change"].corr(aligned["fx_change"])
    rolling_corr = aligned["import_change"].rolling(window=window).corr(aligned["fx_change"])

    abs_corr = abs(static_corr)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.6:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if static_corr > 0 else "negative"
    
    interpretation = f"Import values show {strength} {direction} correlation with FX movements"
    
    if static_corr > 0.3:
        implication = "Weaker GBP tends to coincide with higher import values - natural hedge"
    elif static_corr < -0.3:
        implication = "Weaker GBP tends to coincide with lower import values - amplified risk"
    else:
        implication = "Import values and FX rates move relatively independently"
    
    return {
        "static_correlation": round(static_corr, 4),
        "correlation_strength": strength,
        "correlation_direction": direction,
        "interpretation": interpretation,
        "implication": implication,
        "rolling_correlation": rolling_corr,
        "latest_rolling_corr": round(rolling_corr.iloc[-1], 4) if len(rolling_corr) > 0 and not pd.isna(rolling_corr.iloc[-1]) else None,
        "error": None
    }


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series
) -> Dict:
    """OLS beta of asset returns against market returns, plus alpha and R²."""
    aligned = pd.DataFrame({
        "asset": asset_returns,
        "market": market_returns
    }).dropna()

    if len(aligned) < 20:
        return {"error": "Insufficient data for beta calculation"}

    covariance = aligned["asset"].cov(aligned["market"])
    variance = aligned["market"].var()

    if variance == 0:
        return {"error": "Market variance is zero"}

    beta = covariance / variance

    # Jensen's alpha
    mean_asset = aligned["asset"].mean()
    mean_market = aligned["market"].mean()
    alpha = mean_asset - beta * mean_market

    correlation = aligned["asset"].corr(aligned["market"])
    r_squared = correlation ** 2
    
    # Interpretation
    if beta > 1.2:
        sensitivity = "HIGH"
        interpretation = "Highly sensitive to market movements - amplified risk"
    elif beta > 0.8:
        sensitivity = "MODERATE"
        interpretation = "Moves in line with market"
    elif beta > 0:
        sensitivity = "LOW"
        interpretation = "Less sensitive to market movements - defensive"
    else:
        sensitivity = "NEGATIVE"
        interpretation = "Moves opposite to market - potential hedge"
    
    return {
        "beta": round(beta, 4),
        "alpha": round(alpha * 100, 4),  # As percentage
        "r_squared": round(r_squared, 4),
        "sensitivity": sensitivity,
        "interpretation": interpretation,
        "error": None
    }


# --- risk decomposition ---

def decompose_margin_risk(
    margin_data: pd.DataFrame
) -> Dict:
    """Attributes margin variance to FX, goods cost, shipping, and tariffs."""
    if margin_data.empty:
        return {"error": "No margin data provided"}

    required_cols = ["margin_pct", "fx_shock", "goods_cost", "shipping_cost", "tariff_cost"]
    missing_cols = [col for col in required_cols if col not in margin_data.columns]

    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    margin_var = margin_data["margin_pct"].var()

    if margin_var == 0:
        return {"error": "Zero variance in margin data"}

    # use squared correlation as a proxy for explained variance
    factors = {
        "FX Impact": margin_data["fx_shock"].corr(margin_data["margin_pct"]) ** 2,
        "Goods Cost": margin_data["goods_cost"].corr(margin_data["margin_pct"]) ** 2,
        "Shipping Cost": margin_data["shipping_cost"].corr(margin_data["margin_pct"]) ** 2,
        "Tariff Cost": margin_data["tariff_cost"].corr(margin_data["margin_pct"]) ** 2
    }

    # normalise so contributions sum to 100%
    total_contribution = sum(factors.values())
    if total_contribution > 0:
        factors = {k: round(v / total_contribution * 100, 1) for k, v in factors.items()}

    primary_driver = max(factors.items(), key=lambda x: x[1])
    
    return {
        "risk_contributions": factors,
        "primary_risk_driver": primary_driver[0],
        "primary_contribution": primary_driver[1],
        "total_margin_volatility": round(np.sqrt(margin_var), 2),
        "interpretation": f"The primary driver of margin risk is {primary_driver[0]} ({primary_driver[1]:.1f}% contribution)",
        "error": None
    }


# --- stress testing ---

def run_stress_test(
    base_margin: float,
    scenarios: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Runs each scenario through the margin model and returns a summary DataFrame."""
    from scripts.margin_model import compute_margin

    results = []

    # use 1M as a representative base so results are easy to read
    base_import_value = 1_000_000
    base_revenue = base_import_value * (1 + base_margin / 100)
    
    for scenario_name, params in scenarios.items():
        result = compute_margin(
            import_value_gbp=base_import_value,
            revenue_gbp=base_revenue,
            fx_shock_pct=params.get("fx_shock", 0),
            shipping_pct=params.get("shipping_pct", 0.05),
            insurance_pct=params.get("insurance_pct", 0.01),
            tariff_pct=params.get("tariff_pct", 0.02)
        )
        
        stressed_margin = result["margin_pct"]
        margin_impact = stressed_margin - base_margin
        
        results.append({
            "scenario": scenario_name,
            "fx_shock": params.get("fx_shock", 0) * 100,
            "shipping_pct": params.get("shipping_pct", 0.05) * 100,
            "tariff_pct": params.get("tariff_pct", 0.02) * 100,
            "stressed_margin": round(stressed_margin, 2),
            "margin_impact": round(margin_impact, 2),
            "profit_impact": round(result["profit"] - (base_revenue - result["landed_cost"]), 2)
        })
    
    return pd.DataFrame(results)


def get_predefined_stress_scenarios() -> Dict[str, Dict[str, float]]:
    """Returns a dict of named stress scenarios with their parameter overrides."""
    return {
        "Base Case": {
            "fx_shock": 0.0,
            "shipping_pct": 0.05,
            "insurance_pct": 0.01,
            "tariff_pct": 0.02
        },
        "Mild FX Depreciation": {
            "fx_shock": 0.05,
            "shipping_pct": 0.05,
            "insurance_pct": 0.01,
            "tariff_pct": 0.02
        },
        "Severe FX Depreciation": {
            "fx_shock": 0.15,
            "shipping_pct": 0.05,
            "insurance_pct": 0.01,
            "tariff_pct": 0.02
        },
        "Shipping Crisis": {
            "fx_shock": 0.0,
            "shipping_pct": 0.20,
            "insurance_pct": 0.02,
            "tariff_pct": 0.02
        },
        "Trade War": {
            "fx_shock": 0.05,
            "shipping_pct": 0.08,
            "insurance_pct": 0.015,
            "tariff_pct": 0.15
        },
        "Perfect Storm": {
            "fx_shock": 0.15,
            "shipping_pct": 0.15,
            "insurance_pct": 0.025,
            "tariff_pct": 0.10
        },
        "Economic Recovery": {
            "fx_shock": -0.05,
            "shipping_pct": 0.03,
            "insurance_pct": 0.008,
            "tariff_pct": 0.015
        }
    }


# --- comprehensive risk report ---

def generate_risk_report(
    commodity_code: int,
    currency: str = "EUR",
    import_data_path: str = "data/output/merged_hmrc_ons_commodity.csv",
    fx_data_path: str = "data/raw/exchange"
) -> Dict:
    """Pulls everything together - VaR, vol, correlation, margin risk, and stress tests for one commodity."""
    from scripts.trend_analysis import (
        load_exchange_rates,
        analyse_import_trends,
        simulate_historical_margins
    )

    fx_data = load_exchange_rates(currency, fx_data_path)

    if fx_data.empty:
        return {"error": f"No FX data for {currency}"}

    fx_returns = np.log(fx_data["Rate"] / fx_data["Rate"].shift(1))

    fx_var = calculate_var_historical(fx_returns, confidence_level=0.95)

    vol_data = calculate_historical_volatility(fx_data["Rate"], window=30)
    current_vol = vol_data["rolling_vol"].iloc[-1] if not vol_data.empty else None
    vol_regime = volatility_regime_detection(vol_data["rolling_vol"])

    import_analysis = analyse_import_trends(commodity_code, import_data_path)

    if "data" in import_analysis:
        correlation = analyse_commodity_fx_correlation(import_analysis["data"], fx_data)
    else:
        correlation = {"error": "No import data for correlation"}

    if "data" in import_analysis:
        margin_sim = simulate_historical_margins(import_analysis["data"], fx_data)
        margin_var = calculate_margin_var(margin_sim)
        risk_decomposition = decompose_margin_risk(margin_sim)
    else:
        margin_var = {"error": "No data for margin VaR"}
        risk_decomposition = {"error": "No data for risk decomposition"}

    stress_scenarios = get_predefined_stress_scenarios()
    current_margin = margin_var.get("current_margin", 20)
    stress_results = run_stress_test(current_margin, stress_scenarios)
    
    return {
        "commodity_code": commodity_code,
        "currency": currency,
        "fx_var": fx_var,
        "volatility_analysis": {
            "current_volatility": round(current_vol * 100, 2) if current_vol and not np.isnan(current_vol) else None,
            "regime": vol_regime
        },
        "correlation_analysis": correlation,
        "margin_var": margin_var,
        "risk_decomposition": risk_decomposition,
        "stress_test_results": stress_results.to_dict("records"),
        "generated_at": datetime.now().isoformat()
    }
