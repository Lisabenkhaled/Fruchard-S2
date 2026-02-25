# pricing_tree/adaptateur.py
from __future__ import annotations

from typing import Dict, Any

from .pricer import price_tree_backward_direct


def tree_price_from_mc(
    mc_market,
    mc_trade,
    N: int = 2000,
    optimize: bool = False,
    threshold: float = 1e-14,
    return_tree: bool = False,
) -> Dict[str, Any]:
    """
    Adapter: call the Tree Backward pricer using MC objects.

    Dividend mapping is handled inside price_tree_backward_direct:
        rho = div_amount / S0
        lam = 0.0
    """

    # --- Extract MC market values ---
    S0 = float(mc_market.S0)
    r = float(mc_market.r)
    sigma = float(mc_market.sigma)

    # --- Extract MC trade values ---
    exercise = mc_trade.exercise.lower()

    # ex-div info (dates, not year fractions)
    ex_div_date = getattr(mc_trade, "ex_div_date", None)
    div_amount = float(getattr(mc_trade, "div_amount", 0.0))

    # --- Call the direct tree pricer ---
    return price_tree_backward_direct(
        S0=S0,
        r=r,
        sigma=sigma,
        K=float(mc_trade.strike),
        is_call=bool(mc_trade.is_call),
        exercise=exercise,
        pricing_date=mc_trade.pricing_date,
        maturity_date=mc_trade.maturity_date,
        N=int(N),
        ex_div_date=ex_div_date,
        div_amount=div_amount,
        optimize=bool(optimize),
        threshold=float(threshold),
        return_tree=bool(return_tree),
    )