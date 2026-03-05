# pricing_monte_carlo/core_greeks.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# ---- repo root so we can import sibling pricing_tree ----
def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pricing_monte_carlo").is_dir() and (parent / "pricing_tree").is_dir():
            p = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        f"Repo root not found from {here}. Expected sibling folders pricing_monte_carlo/ and pricing_tree/."
    )

_add_repo_root_to_syspath()

# ---- imports (now stable) ----
from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.core_pricer import CorePricingParams, core_price
from pricing_monte_carlo.greek.greeks import compute_greeks_vector
from pricing_tree.greek_adaptateur import tree_greeks_from_mc


def _price_from_core(market: Market, trade: OptionTrade, params: CorePricingParams) -> float:
    price, _, _, _ = core_price(market, trade, params)
    return float(price)


def core_greeks(
    market: Market,
    trade: OptionTrade,
    params: CorePricingParams,
    *,
    eps_spot: Optional[float] = None,
    eps_vol: float = 0.01,
    tree_N: int = 300,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (mc_greeks, tree_greeks).
    """
    if eps_spot is None:
        eps_spot = float(market.S0) * 0.001  # 0.1% spot

    mc = compute_greeks_vector(
        price_fn=_price_from_core,
        market=market,
        trade=trade,
        params=params,
        eps_spot=eps_spot,
        eps_vol=eps_vol,
    )
    mc = {k: float(v) for k, v in mc.items()}

    tree = tree_greeks_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=tree_N,
    )
    tree = {k: float(v) for k, v in tree.items()}

    return mc, tree


def print_greeks(title: str, g: Dict[str, float]) -> None:
    print(title)
    for k, v in g.items():
        print(f"{k}: {v:.6f}")

import datetime as dt

from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.core_pricer import CorePricingParams
from pricing_monte_carlo.core_greeks import core_greeks, print_greeks

pricing_date = dt.date(2026, 3, 1)
maturity_date = dt.date(2026, 12, 25)

market = Market(S0=100, r=0.05, sigma=0.2)

trade = OptionTrade(
    strike=100,
    is_call=True,
    exercise="american",
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    ex_div_date=dt.date(2026, 11, 30),
    div_amount=3.0
)

params = CorePricingParams(
    n_paths=100_000,
    n_steps=300,
    seed=2,
    antithetic=True,
    method="vector",
    american_algo="ls",
    basis="laguerre",
    degree=2
)

mc, tree = core_greeks(market, trade, params, tree_N=300)

print_greeks("MC Greeks", mc)
print_greeks("\nTree Greeks", tree)