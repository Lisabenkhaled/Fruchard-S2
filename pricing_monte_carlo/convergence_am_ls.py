# ls_convergence_to_tree_simple.py
# ------------------------------------------------------------
# Goal:
#   Check if American LS Monte Carlo price converges to the Tree price as N increases.
#   Reduce noise by averaging across seeds for each N.
#   Plot mean MC price vs N and the Tree price (horizontal line).
# ------------------------------------------------------------

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc
from core_pricer import CorePricingParams, core_price


def ls_converges_to_tree(
    market,
    trade,
    n_paths_list,
    seeds,
    n_steps_mc=200,
    tree_steps=2000,
    antithetic=True,
    method_mc="vector",
    basis="laguerre",
    degree=2,
):
    """
    For each N in n_paths_list:
      - run LS Monte Carlo for each seed in seeds
      - take the mean price across seeds
    Then plot mean price vs N and the tree benchmark.
    """

    if trade.exercise.lower() != "american":
        raise ValueError("trade.exercise must be 'american'")

    # make sure N is even if antithetic=True
    n_paths_list = [int(n) for n in n_paths_list]
    if antithetic:
        n_paths_list = [n if n % 2 == 0 else n + 1 for n in n_paths_list]

    seeds = [int(s) for s in seeds]

    # Tree benchmark (Backward)
    tree_res = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_steps),
        optimize=False,
    )
    tree_price = float(tree_res["tree_price"])

    mean_prices = []

    print("\nAmerican LS MC vs Tree (mean across seeds)")
    print("------------------------------------------------------")
    print("N paths | mean LS price | tree price")
    print("------------------------------------------------------")

    for N in n_paths_list:
        prices = []

        for sd in seeds:
            params = CorePricingParams(
                n_paths=int(N),
                n_steps=int(n_steps_mc),
                seed=int(sd),
                antithetic=bool(antithetic),
                method=method_mc,
                american_algo="ls",
                basis=basis,
                degree=int(degree),
            )
            price, _, _, _ = core_price(market, trade, params)
            prices.append(float(price))

        mean_p = float(np.mean(prices))
        mean_prices.append(mean_p)

        print(f"{N:>6d} | {mean_p:>13.6f} | {tree_price:>10.6f}")

    # Plot
    x = np.asarray(n_paths_list, dtype=float)
    y = np.asarray(mean_prices, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o-", label="LS MC mean price (across seeds)")
    plt.axhline(tree_price, linestyle="--", label=f"Tree price = {tree_price:.6f}")

    plt.xscale("log")
    plt.xlabel("Number of paths N (log scale)")
    plt.ylabel("Option price")
    plt.title("Convergence of American LS Monte Carlo price to Tree benchmark")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "tree_price": tree_price,
        "n_paths_list": n_paths_list,
        "mean_prices": mean_prices,
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import datetime as dt
    from model.market import Market
    from model.option import OptionTrade

    market = Market(S0=100.0, r=0.05, sigma=0.20)

    trade = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        ex_div_date=dt.date(2026, 10, 30),
        div_amount=3.0,
    )

    n_paths_list = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
    seeds = list(range(1, 51))  # K=50 seeds

    res = ls_converges_to_tree(
        market=market,
        trade=trade,
        n_paths_list=n_paths_list,
        seeds=seeds,
        n_steps_mc=300,
        tree_steps=300,
        antithetic=True,
        method_mc="vector",
        basis="laguerre",
        degree=2,
    )

    print("\nTree price:", res["tree_price"])
    print("Mean prices:", res["mean_prices"])