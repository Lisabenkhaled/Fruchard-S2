# pricing_monte_carlo/core_greeks.py
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, Optional


# Repo root
def _add_repo_root_to_syspath() -> None:
    here: Path = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pricing_monte_carlo").is_dir() and (parent / "pricing_tree").is_dir():
            p: str = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        f"Repo root not found from {here}. Expected pricing_monte_carlo/ and pricing_tree/."
    )


_add_repo_root_to_syspath()

# importation
from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.core_pricer import CorePricingParams, core_price
from pricing_monte_carlo.greek.greeks import compute_greeks_vector

from pricing_tree.greek_adaptateur import tree_greeks_from_mc
from pricing_monte_carlo.utils.utils_bs import bs_greeks
from pricing_monte_carlo.utils.utils_date import datetime_to_years


def _price_from_core(market: Market, trade: OptionTrade, params: CorePricingParams) -> float:
    price, _, _, _ = core_price(market, trade, params)
    return float(price)


def _bench_greeks(market: Market, trade: OptionTrade, tree_N: int) -> Dict[str, float]:
    """
    Benchmark:
    - EU sans div : Black-Scholes
    - EU avec div / AM : Tree
    """
    if trade.exercise.lower() == "european" and getattr(trade, "div_amount", 0.0) == 0.0:
        T: float = float(datetime_to_years(trade.maturity_date, trade.pricing_date))
        bs: Dict[str, float] = bs_greeks(
            S=market.S0,
            K=trade.strike,
            r=market.r,
            sigma=market.sigma,
            T=T,
            is_call=trade.is_call,
        )
        return {k.lower(): float(v) for k, v in bs.items()}

    tree: Dict[str, float] = tree_greeks_from_mc(mc_market=market, mc_trade=trade, N=tree_N)
    return {k.lower(): float(v) for k, v in tree.items()}


def core_greeks(market: Market, trade: OptionTrade, params: CorePricingParams, shift_spot: Optional[float] = None, 
                shift_vol: float = 0.01, tree_N: int = 300) -> Tuple[Dict[str, float], Dict[str, float]]:

    # default bump
    if shift_spot is None:
        shift_spot = float(market.S0) * 0.001

    # Monte Carlo greeks
    mc_raw: Dict[str, float] = compute_greeks_vector(
        price_fn=_price_from_core,
        market=market,
        trade=trade,
        params=params,
        shift_spot=shift_spot,
        shift_vol=shift_vol,
    )
    mc: Dict[str, float] = {k.lower(): float(v) for k, v in mc_raw.items()}

    # Benchmark greeks (BS or Tree)
    ref: Dict[str, float] = _bench_greeks(market, trade, tree_N)

    return mc, ref


def print_greeks(title: str, g: Dict[str, float]) -> None:
    print(title)
    for k, v in g.items():
        print(f"{k:>6s}: {v:.6f}")


def print_comparison(mc: Dict[str, float], ref: Dict[str, float]) -> None:
    print("\nComparison (MC vs Benchmark)")
    for k, v in mc.items():
        if k in ref:
            diff: float = v - ref[k]
            print(f"{k:>6s} | MC: {v:.6f} | Bench: {ref[k]:.6f} | Diff: {diff:.6f}")


# Example usage
if __name__ == "__main__":
    # dates
    pricing_date: dt.date = dt.date(2026, 3, 1)  # pricing date
    maturity_date: dt.date = dt.date(2026, 12, 25)  # maturity

    # market
    market: Market = Market(S0=100, r=0.10, sigma=0.20)  # inputs

    # trades
    trade: OptionTrade = OptionTrade(
        strike=100,  # strike
        is_call=False,  # call/put
        exercise="european",  # european/american
        pricing_date=pricing_date,  # pricing date
        maturity_date=maturity_date,  # maturity
        ex_div_date=dt.date(2026, 11, 30),  # dividend date
        div_amount=0.0,  # dividend amount
    )

    # MC param2s
    params: CorePricingParams = CorePricingParams(
        n_paths=100_000,  # paths
        n_steps=100,  # steps
        seed=1,  # seed
        antithetic=True,  # antithetic
        method="vector",  # vector only
        american_algo="ls",  # for AM
        basis="laguerre",  # basis
        degree=2,  # degree
    )

    # run
    mc, ref = core_greeks(market, trade, params)  # compute

    # print
    print_greeks("Monte Carlo Greeks", mc)  # mc
    print_greeks("\nBenchmark Greeks", ref)  # bench
    print_comparison(mc, ref)  # compare