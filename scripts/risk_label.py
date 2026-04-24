# risk_label.py
"""Risk classification based on profit margin thresholds."""

# Named constants for risk thresholds
MARGIN_THRESHOLD_HIGH: float = 5.0
MARGIN_THRESHOLD_MODERATE: float = 10.0
MARGIN_THRESHOLD_LOW: float = 20.0


def risk_label(margin_pct: float) -> str:
    """
    Classify financial risk based on margin percentage.

    Args:
        margin_pct (float): Profit margin as a percentage.

    Returns:
        str: Risk label — 'HIGH', 'MODERATE', or 'LOW'.
    """
    try:
        if margin_pct is not None:
            margin_pct = float(margin_pct)
    except (TypeError, ValueError):
        return "HIGH"

    if margin_pct is None or margin_pct < MARGIN_THRESHOLD_HIGH:
        return "HIGH"
    elif margin_pct < MARGIN_THRESHOLD_MODERATE:
        return "MODERATE"
    elif margin_pct < MARGIN_THRESHOLD_LOW:
        return "LOW"
    else:
        return "LOW"
