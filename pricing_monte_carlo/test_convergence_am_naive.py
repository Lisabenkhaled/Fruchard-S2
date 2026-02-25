import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc

import numpy as np
import matplotlib.pyplot as plt

from core_pricer import CorePricingParams, core_price


def naive_am_vs_tree(
    market,
    trade,
    n_paths_list=(2_000, 5_000, 10_000),
    n_steps_mc=200,
    tree_steps=2_000,
    fixed_seed=42,
    antithetic=True,
    method_mc="vector",
):

    if trade.exercise.lower() != "american":
        raise ValueError("trade.exercise must be 'american'")

    n_paths_list = [int(n) for n in n_paths_list]
    if antithetic:
        n_paths_list = [n if n % 2 == 0 else n + 1 for n in n_paths_list]

    tree_res = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_steps),
        optimize=False,
    )
    tree_price = float(tree_res["tree_price"])
    naive_prices = []

    for n_paths in n_paths_list:

        params = CorePricingParams(
            n_paths=int(n_paths),
            n_steps=int(n_steps_mc),
            seed=int(fixed_seed),
            antithetic=bool(antithetic),
            method=method_mc,
            american_algo="naive",
            basis="power",
            degree=2
        )

        price, elapsed_time = core_price(market, trade, params)
        naive_prices.append(float(price))


    # Plot
    x = np.array(n_paths_list, dtype=float)
    y = np.array(naive_prices, dtype=float)

    plt.figure()
    plt.plot(
        x, y, "o-",
        label=f"Naive American MC (seed={fixed_seed})"
    )
    plt.axhline(
        tree_price, linestyle="--",
        label=f"Tree American (Backward) = {tree_price:.6f}"
    )

    plt.xscale("log")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Option price")
    plt.title("Naive American MC vs Trinomial Tree (Backward)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    return {
        "tree_price": tree_price,
        "naive_prices": naive_prices,
        "n_paths_list": n_paths_list
    }

import datetime as dt
from model.market import Market
from model.option import OptionTrade

market = Market(S0=100, r=0.10, sigma=0.20)

trade = OptionTrade(
    strike=100,
    is_call=False,
    exercise="american",
    pricing_date=dt.date(2026, 3, 1),
    maturity_date=dt.date(2026, 12, 26),
    ex_div_date=None,
    div_amount=0.0
)

results = naive_am_vs_tree(
    market=market,
    trade=trade,
    n_paths_list=(100, 500, 1000,),
    n_steps_mc=100,
    tree_steps=500,
    fixed_seed=42,
    antithetic=True,
    method_mc="vector",
)

print("Tree Price:", results["tree_price"])
print("Naive Prices:", results["naive_prices"])