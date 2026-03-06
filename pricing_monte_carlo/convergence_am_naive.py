from __future__ import annotations

import sys
import os
import datetime as dt
from typing import List, Sequence, TypedDict

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc
from core_pricer import CorePricingParams, core_price
from model.market import Market
from model.option import OptionTrade
class NaiveAmVsTreeResult(TypedDict):
    tree_price: float
    naive_prices: List[float]
    n_paths_list: List[int]


def _plot(
    x: Sequence[int],
    y: Sequence[float],
    tree_price: float,
    seed: int
) -> None:
    xx = np.array(x, dtype=float)
    yy = np.array(y, dtype=float)

    # Plot
    plt.figure()
    plt.plot(xx, yy, "o-", label=f"Naive American MC (seed={seed})")
    plt.axhline(tree_price, linestyle="--", label=f"Tree = {tree_price:.6f}")
    plt.xscale("log")
    plt.xlabel("Number of paths (log)")
    plt.ylabel("Option price")
    plt.title("Naive American MC vs Tree")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def naive_am_vs_tree(
    market: "Market",
    trade: "OptionTrade",
    n_paths_list: Sequence[int] = (2_000, 5_000, 10_000),
    n_steps_mc: int = 200,
    tree_steps: int = 2_000,
    fixed_seed: int = 42,
    antithetic: bool = True,
    method_mc: str = "vector"
) -> NaiveAmVsTreeResult:
    
    if trade.exercise.lower() != "american":
        raise ValueError("trade.exercise must be 'american'")

    # Convert path counts to integers
    paths: List[int] = [int(n) for n in n_paths_list]

    # Antithetic sampling needs an even number of paths
    if antithetic:
        paths = [n if n % 2 == 0 else n + 1 for n in paths]

    # Tree benchmark price
    tree_res = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_steps),
        optimize=False,
    )
    tree_price: float = float(tree_res["tree_price"])

    # Monte Carlo prices
    naive_prices: List[float] = []

    for n_paths in paths:
        params = CorePricingParams(
            n_paths=int(n_paths),
            n_steps=int(n_steps_mc),
            seed=int(fixed_seed),
            antithetic=bool(antithetic),
            method=method_mc,
            american_algo="naive",
            basis="laguerre",
            degree=2,
        )

        price, _, _, _ = core_price(market, trade, params)
        naive_prices.append(float(price))

    _plot(paths, naive_prices, tree_price, fixed_seed)

    return {
        "tree_price": tree_price,
        "naive_prices": naive_prices,
        "n_paths_list": paths,
    }

# Exemple
market: Market = Market(S0=100, r=0.10, sigma=0.20)

trade: OptionTrade = OptionTrade(
    strike=100,
    is_call=False,
    exercise="american",
    pricing_date=dt.date(2026, 3, 1),
    maturity_date=dt.date(2026, 12, 26),
    ex_div_date=None,
    div_amount=0.0
)

results: NaiveAmVsTreeResult = naive_am_vs_tree(
    market=market,
    trade=trade,
    n_paths_list=(100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000),
    n_steps_mc=250,
    tree_steps=250,
    fixed_seed=42,
    antithetic=True,
    method_mc="vector"
)

print("Tree Price:", results["tree_price"])
print("Naive Prices:", results["naive_prices"])