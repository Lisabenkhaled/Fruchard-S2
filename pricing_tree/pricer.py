# pricing_tree/tree_pricer_backward.py
from __future__ import annotations

import time
import datetime as dt
from typing import Dict, Any, Literal, Optional

from .models.market import Market
from .models.option_trade import Option
from .models.tree import TrinomialTree
from .models.backward_pricing import price_backward
from .utils.utils_date import datetime_to_years

Exercise = Literal["european", "american"]


def price_tree_backward_direct(
    *,
    S0: float,
    r: float,
    sigma: float,
    K: float,
    is_call: bool,
    exercise: Exercise,
    pricing_date: dt.date,
    maturity_date: dt.date,
    N: int = 10_000,
    ex_div_date: Optional[dt.date] = None,
    div_amount: float = 0.0,
    optimize: bool = False,
    threshold: float = 1e-14,
    return_tree: bool = False,
) -> Dict[str, Any]:
    """
    Trinomial Tree pricer (Backward only), fed by direct numeric values.

    Dividend convention (aligned with your adaptateur.py):
        rho = div_amount / S0
        lam = 0.0
        exdivdate = year-fraction(pricing_date -> ex_div_date)
    """

    # ---- checks ----
    exercise = exercise.lower()
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")

    if N <= 0:
        raise ValueError("N must be >= 1")

    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    if div_amount < 0:
        raise ValueError("div_amount must be >= 0")

    # ---- dates -> year fractions ----
    T = float(datetime_to_years(maturity_date, pricing_date))
    if T <= 0:
        raise ValueError("Invalid maturity: maturity_date must be after pricing_date")

    exdivdate = None
    if ex_div_date is not None:
        ex_t = float(datetime_to_years(ex_div_date, pricing_date))
        # If ex-date is outside (0, T], ignore (or raise). Here: raise for safety.
        if ex_t <= 0 or ex_t > T:
            raise ValueError(f"ex_div_date produces exdivtime={ex_t:.6f} outside (0, T].")
        if div_amount == 0.0:
            # ex-date given but no amount: ok, but effectively no dividend
            exdivdate = None
        else:
            exdivdate = ex_t

    # ---- dividend mapping (your correction) ----
    rho = (float(div_amount) / float(S0)) if (div_amount != 0.0) else 0.0
    lam = 0.0

    # ---- build objects ----
    market = Market(
        S0=float(S0),
        r=float(r),
        sigma=float(sigma),
        T=float(T),
        exdivdate=exdivdate,
        pricing_date=pricing_date,
        rho=float(rho),
        lam=float(lam),
    )

    option = Option(K=float(K), is_call=bool(is_call))

    # ---- run backward pricing ----
    t0 = time.time()

    tree = TrinomialTree(market, option, int(N), exercise)
    tree.build_tree()
    tree.compute_reach_probabilities()

    if optimize:
        tree.prune_tree(float(threshold))

    price = float(price_backward(tree))
    elapsed = float(time.time() - t0)

    out: Dict[str, Any] = {
        "tree_price": price,
        "tree_time": elapsed,
    }
    if return_tree:
        out["tree"] = tree
    return out


# Example usage
if __name__ == "__main__":
    res = price_tree_backward_direct(
        S0=100.0,
        r=0.05,
        sigma=0.20,
        K=100.0,
        is_call=False,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        ex_div_date=dt.date(2026, 10, 30),
        div_amount=3.0,     # rho = 3/100
        N=5000,
        optimize=False,
    )
    print(f"Tree backward price: {res['tree_price']:.6f} (time={res['tree_time']:.4f}s)")