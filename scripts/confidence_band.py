# confidence_band.py
"""Confidence band calculations based on ONS data coverage quality."""

from typing import Tuple

# Coverage-to-uncertainty mapping
COVERAGE_UNCERTAINTY = {
    "No coverage": 0.40,       # ±40%
    "Low coverage": 0.25,      # ±25%
    "Partial coverage": 0.15,  # ±15%
    "High coverage": 0.05,     # ±5%
}
DEFAULT_UNCERTAINTY: float = 0.30


def confidence_multiplier(coverage_class: str) -> float:
    """
    Map ONS coverage quality to an uncertainty multiplier.

    Args:
        coverage_class (str): One of 'No coverage', 'Low coverage',
            'Partial coverage', or 'High coverage'.

    Returns:
        float: Uncertainty fraction (e.g. 0.15 means ±15%).
    """
    return COVERAGE_UNCERTAINTY.get(
        str(coverage_class) if coverage_class else "", DEFAULT_UNCERTAINTY
    )


def compute_confidence_band(profit: float, coverage_class: str) -> Tuple[float, float]:
    """
    Return lower and upper profit bounds.

    Args:
        profit (float): Base profit value in GBP.
        coverage_class (str): ONS coverage classification.

    Returns:
        tuple[float, float]: (lower_bound, upper_bound) of profit.
    """
    try:
        profit = float(profit)
    except (TypeError, ValueError):
        return 0.0, 0.0
    m = confidence_multiplier(coverage_class)
    lower = profit * (1 - m)
    upper = profit * (1 + m)
    return lower, upper
