# margin_model.py

def compute_margin(
    import_value_gbp: float,
    revenue_gbp: float,
    fx_shock_pct: float = 0.0,
    shipping_pct: float = 0.0,
    insurance_pct: float = 0.0,
    tariff_pct: float = 0.0,
):
    """
    Calculates landed cost and profit margin for a UK SME import scenario.
    FX shock adjusts the goods cost upward (positive = weaker GBP).
    """

    try:
        import_value_gbp = float(import_value_gbp)
        revenue_gbp = float(revenue_gbp)
        fx_shock_pct = float(fx_shock_pct)
        shipping_pct = float(shipping_pct)
        insurance_pct = float(insurance_pct)
        tariff_pct = float(tariff_pct)
    except (TypeError, ValueError) as exc:
        return {
            "goods_cost": 0, "shipping_cost": 0, "insurance_cost": 0,
            "tariff_cost": 0, "landed_cost": 0, "profit": 0,
            "margin_pct": None, "error": f"Invalid input: {exc}",
        }

    # goods cost after FX adjustment
    goods_cost = import_value_gbp * (1 + fx_shock_pct)

    shipping_cost = goods_cost * shipping_pct
    insurance_cost = goods_cost * insurance_pct
    tariff_cost = goods_cost * tariff_pct

    landed_cost = (
        goods_cost
        + shipping_cost
        + insurance_cost
        + tariff_cost
    )

    # cap landed cost at 200% of import value - avoids absurd results from extreme inputs
    MAX_COST_MULTIPLIER = 2.0
    landed_cost = min(landed_cost, import_value_gbp * MAX_COST_MULTIPLIER)

    # clamp downside - can't lose more than the import value
    profit = revenue_gbp - landed_cost
    profit = max(profit, -import_value_gbp)

    if revenue_gbp > 0:
        margin_pct = (profit / revenue_gbp) * 100
        margin_pct = max(margin_pct, -100)
    else:
        margin_pct = None


    return {
        "goods_cost": round(goods_cost, 2),
        "shipping_cost": round(shipping_cost, 2),
        "insurance_cost": round(insurance_cost, 2),
        "tariff_cost": round(tariff_cost, 2),
        "landed_cost": round(landed_cost, 2),
        "profit": round(profit, 2),
        "margin_pct": round(margin_pct, 2) if margin_pct is not None else None,
    }
