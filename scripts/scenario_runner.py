# scenario_runner.py

import pandas as pd
from scripts.margin_model import compute_margin

def run_sensitivity_scenarios(
    import_value_gbp: float,
    revenue_gbp: float,
    fx_range=(-0.1, 0.1),
    shipping_range=(0.0, 0.3),
    steps=11,
    tariff_pct=0.0,
    insurance_pct=0.0,
):
    """
    Run FX × shipping sensitivity analysis.
    """
    try:
        import_value_gbp = float(import_value_gbp)
        revenue_gbp = float(revenue_gbp)
        steps = max(int(steps), 2)
    except (TypeError, ValueError) as exc:
        return pd.DataFrame(columns=["fx_shock_pct", "shipping_pct", "profit", "margin_pct"])

    fx_values = [fx_range[0] + i * (fx_range[1] - fx_range[0]) / (steps - 1) for i in range(steps)]
    ship_values = [shipping_range[0] + i * (shipping_range[1] - shipping_range[0]) / (steps - 1) for i in range(steps)]

    rows = []

    for fx in fx_values:
        for ship in ship_values:
            result = compute_margin(
                import_value_gbp=import_value_gbp,
                revenue_gbp=revenue_gbp,
                fx_shock_pct=fx,
                shipping_pct=ship,
                insurance_pct=insurance_pct,
                tariff_pct=tariff_pct,
            )
            rows.append({
                "fx_shock_pct": fx * 100,
                "shipping_pct": ship * 100,
                "profit": result["profit"],
                "margin_pct": result["margin_pct"],
            })

    return pd.DataFrame(rows)
