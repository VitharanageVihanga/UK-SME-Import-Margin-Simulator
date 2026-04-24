# risk_adjuster.py
"""Adjust financial risk labels using ONS data coverage quality."""


def adjust_risk(margin_risk: str, coverage_class: str) -> str:
    """
    Adjust financial risk using ONS data coverage.

    Args:
        margin_risk (str): Base risk label ('HIGH', 'MODERATE', 'LOW').
        coverage_class (str): ONS coverage classification.

    Returns:
        str: Adjusted risk label.
    """
    if not isinstance(margin_risk, str) or not isinstance(coverage_class, str):
        return str(margin_risk) if margin_risk else "HIGH"

    if margin_risk == "HIGH":
        return "HIGH"

    if margin_risk == "MODERATE":
        if coverage_class in ["No coverage", "Low coverage"]:
            return "HIGH"
        return "MODERATE"

    if margin_risk == "LOW":
        if coverage_class in ["No coverage", "Low coverage"]:
            return "MODERATE"
        return "LOW"

    return margin_risk
