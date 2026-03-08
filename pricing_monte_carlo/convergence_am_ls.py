# ls_convergence_to_tree_simple.py
# ------------------------------------------------------------
# Goal:
#   Check if American LS Monte Carlo price converges to the Tree price as N increases.
#   Reduce noise by averaging across seeds for each N.
#   Plot mean MC price vs N and the Tree price (horizontal line).
# ------------------------------------------------------------

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc
from core_pricer import CorePricingParams, core_price

def _tree_benchmark_price(market: Any, trade: Any, tree_steps: int) -> float:
    """Compute Tree benchmark price"""
    tree_res = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_steps),
        optimize=False,
    )
    return float(tree_res["tree_price"])

def _ls_price_one_seed(market: Any, trade: Any, n_paths: int, n_steps_mc: int,
    seed: int, antithetic: bool, method_mc: str, basis: str, degree: int) -> float:
    """Compute one LS Monte Carlo price for one seed."""
    params = CorePricingParams(
        n_paths=int(n_paths),
        n_steps=int(n_steps_mc),
        seed=int(seed),
        antithetic=bool(antithetic),
        method=method_mc,
        american_algo="ls",
        basis=basis,
        degree=int(degree),
    )
    price, _, _, _ = core_price(market, trade, params)
    return float(price)


def _mean_ls_price_for_n(market: Any, trade: Any, n_paths: int, seeds: Sequence[int], n_steps_mc: int,
    antithetic: bool, method_mc: str, basis: str, degree: int) -> float:
    """Average LS price across seeds for a fixed N."""
    prices: List[float] = []

    # Reduce MC noise by repeating across many random seeds
    for sd in seeds:
        p = _ls_price_one_seed(
            market=market,
            trade=trade,
            n_paths=n_paths,
            n_steps_mc=n_steps_mc,
            seed=int(sd),
            antithetic=antithetic,
            method_mc=method_mc,
            basis=basis,
            degree=degree,
        )
        prices.append(p)

    return float(np.mean(prices))


def _print_header() -> None:
    """Pretty print header for console output."""
    print("\nAmerican LS MC vs Tree (mean across seeds)")
    print("------------------------------------------------------")
    print("N paths | mean LS price | tree price")
    print("------------------------------------------------------")


def _print_row(n_paths: int, mean_price: float, tree_price: float) -> None:
    """Pretty print one result line."""
    print(f"{n_paths:>6d} | {mean_price:>13.6f} | {tree_price:>10.6f}")


def _plot_convergence(n_paths_list: Sequence[int], mean_prices: Sequence[float], tree_price: float) -> None:
    """Plot mean MC price vs N + tree benchmark."""
    x = np.asarray(n_paths_list, dtype=float)
    y = np.asarray(mean_prices, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o-", label="LS MC mean price (across seeds)")
    plt.axhline(tree_price, linestyle="--", label=f"Tree price = {tree_price:.6f}")

    # Log scale helps visualize convergence across wide N range
    plt.xscale("log")
    plt.xlabel("Number of paths N (log scale)")
    plt.ylabel("Option price")
    plt.title("Convergence of American LS Monte Carlo price to Tree benchmark")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def ls_converges_to_tree(market: Any, trade: Any, n_paths_list: Sequence[int], seeds: Sequence[int],
    n_steps_mc: int = 200, tree_steps: int = 2000, antithetic: bool = True,
    method_mc: str = "vector", basis: str = "laguerre", degree: int = 2) -> Dict[str, Any]:
    if trade.exercise.lower() != "american":
        raise ValueError("trade.exercise must be 'american'")

    # Clean inputs 
    n_list_clean = [int(n) for n in n_paths_list]
    seeds_clean = [int(s) for s in seeds]

    # Compute the tree benchmark once
    tree_price = _tree_benchmark_price(market, trade, int(tree_steps))

    mean_prices: List[float] = []
    _print_header()

    # Main loop: compute mean LS price for each N
    for n_paths in n_list_clean:
        mean_p = _mean_ls_price_for_n(
            market=market,
            trade=trade,
            n_paths=int(n_paths),
            seeds=seeds_clean,
            n_steps_mc=int(n_steps_mc),
            antithetic=bool(antithetic),
            method_mc=str(method_mc),
            basis=str(basis),
            degree=int(degree),
        )
        mean_prices.append(mean_p)
        _print_row(int(n_paths), mean_p, tree_price)

    # Plot the convergence chart
    _plot_convergence(n_list_clean, mean_prices, tree_price)

    return {
        "tree_price": tree_price,
        "n_paths_list": n_list_clean,
        "mean_prices": mean_prices,
    }


# Example usage
def main() -> None:
    """Run a small convergence experiment"""
    import datetime as dt
    from model.market import Market
    from model.option import OptionTrade

    # Market inputs
    market = Market(S0=100.0, r=0.05, sigma=0.20)

    # American put with dividend (common LS test case)
    trade = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        ex_div_date=dt.date(2026, 10, 30),
        div_amount=3.0,
    )

    # Increasing N to check convergence
    n_paths_list = [1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 50_000, 100_000]

    # Use many seeds to reduce noise (mean estimate becomes smoother)
    seeds = list(range(1, 51))  # K=50 seeds

    res = ls_converges_to_tree(
        market=market,
        trade=trade,
        n_paths_list=n_paths_list,
        seeds=seeds,
        n_steps_mc=100,
        tree_steps=100,
        antithetic=True,
        method_mc="vector",
        basis="laguerre",
        degree=2,
    )

    print("\nTree price:", res["tree_price"])
    print("Mean prices:", res["mean_prices"])


if __name__ == "__main__":
    main()