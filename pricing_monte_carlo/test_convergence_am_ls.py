import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc

import numpy as np
import matplotlib.pyplot as plt

from core_pricer import CorePricingParams, core_price


def ls_am_vs_tree(
    market,
    trade,
    n_paths_list=(2_000, 5_000, 10_000, 20_000, 50_000, 100_000),
    n_steps_mc=200,
    tree_steps=2_000,
    fixed_seed=42,
    antithetic=True,
    method_mc="vector",
    basis="laguerre",
    degree=2,
):
    """
    Price comparison only:
        Longstaff–Schwartz American MC (fixed seed) vs Tree Backward
    """

    if trade.exercise.lower() != "american":
        raise ValueError("trade.exercise must be 'american'")

    # Ensure even paths if antithetic=True
    n_paths_list = [int(n) for n in n_paths_list]
    if antithetic:
        n_paths_list = [n if n % 2 == 0 else n + 1 for n in n_paths_list]

    # -------------------------
    # Tree price (Backward only)
    # -------------------------
    tree_res = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_steps),
        optimize=False,
    )
    tree_price = float(tree_res["tree_price"])

    ls_prices = []

    for n_paths in n_paths_list:
        params = CorePricingParams(
            n_paths=int(n_paths),
            n_steps=int(n_steps_mc),
            seed=int(fixed_seed),
            antithetic=bool(antithetic),
            method=method_mc,
            american_algo="ls",
            basis=basis,
            degree=int(degree),
        )

        out = core_price(market, trade, params)
        price = out[0]

        if isinstance(price, tuple):
            price = price[0]

        ls_prices.append(float(price))

    x = np.array(n_paths_list, dtype=float)
    y = np.array(ls_prices, dtype=float)

    plt.figure()
    plt.plot(
        x, y, "o-",
        label=f"AM LS MC (seed={fixed_seed}, basis={basis}, deg={degree})"
    )
    plt.axhline(
        tree_price, linestyle="--",
        label=f"Tree American (Backward) = {tree_price:.6f}"
    )

    plt.xscale("log")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Option price")
    plt.title("American Option with Discrete Dividend:\nLS MC vs Trinomial Tree (Backward)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    return {
        "tree_price": tree_price,
        "ls_prices": ls_prices,
        "n_paths_list": n_paths_list,
    }


if __name__ == "__main__":
    import datetime as dt
    from model.market import Market
    from model.option import OptionTrade

    market = Market(S0=100, r=0.05, sigma=0.20)

    trade = OptionTrade(
        strike=100,
        is_call=False,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        ex_div_date=dt.date(2026, 10, 30),
        div_amount=3.0
    )

    results = ls_am_vs_tree(
        market=market,
        trade=trade,
        n_paths_list=list(range(1000, 20001, 500)),
        n_steps_mc=300,
        tree_steps=300,
        fixed_seed=42,
        antithetic=True,
        method_mc="vector",
        basis="laguerre",  # or "power"
        degree=2,
    )

    print("Tree Price:", results["tree_price"])
    print("LS Prices:", results["ls_prices"])