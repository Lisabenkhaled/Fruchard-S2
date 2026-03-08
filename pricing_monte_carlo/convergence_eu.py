import sys
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# Project path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price
from utils.utils_stats import (
    sample_mean,
    sample_std,
    standard_error,
    standard_error_anti,
)


# Helpers
def has_discrete_dividend(trade: OptionTrade) -> bool:
    """True if the trade has a discrete dividend"""
    return (trade.ex_div_date is not None) and (trade.div_amount != 0.0)


def benchmark_price(market: Market, trade: OptionTrade, tree_N: int = 1000) -> tuple[float, str]:
    """Return BS price if possible, otherwise tree benchmark"""
    if not has_discrete_dividend(trade) and getattr(trade, "q", 0.0) == 0.0:
        p = bs_price(
            S=market.S0,
            K=trade.strike,
            r=market.r,
            sigma=market.sigma,
            T=trade.T,
            is_call=trade.is_call,
        )
        return float(p), "BS"

    out = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=tree_N,
        optimize=False,
        threshold=0.0,
        return_tree=False,
    )
    return float(out["tree_price"]), "Tree"


def normal_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Gaussian pdf used only for visual overlay."""
    if sig <= 0:
        return np.zeros_like(x)
    z = (x - mu) / sig
    return (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def _run_prices(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seeds: list[int], antithetic: bool) -> np.ndarray:
    """Run MC for all seeds and store one price per seed"""
    prices = np.empty(len(seeds), dtype=float)

    # Loop over seeds
    for i, sd in enumerate(seeds):
        p, _ = price_european_naive_mc_vector(
            market=market,
            trade=trade,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=int(sd),
            antithetic=bool(antithetic),
        )
        prices[i] = float(p)

    return prices


def _compute_summary(prices: np.ndarray, bench: float, antithetic: bool) -> tuple[float, float, float, float, bool]:
    """Compute mean, SE, CI and pass/fail"""
    mc_mean = float(sample_mean(prices))
    se = float(standard_error_anti(prices) if antithetic else standard_error(prices))

    ci_left = mc_mean - 1.96 * se
    ci_right = mc_mean + 1.96 * se

    passed = abs(mc_mean - bench) <= 1.96 * se if se > 0 else (mc_mean == bench)
    return mc_mean, se, ci_left, ci_right, passed


def _print_summary(bench_name: str, bench: float, n_paths: int, n_steps: int,
    seeds: list[int], antithetic: bool, mc_mean: float, se: float,
    ci_left: float, ci_right: float, passed: bool) -> None:
    """Print a simple console summary"""

    print("\n" + "=" * 60)
    print("MC histogram check (European)")
    print("=" * 60)
    print(f"Benchmark      : {bench_name} = {bench:.6f}")
    print(f"N paths        : {n_paths}")
    print(f"n_steps        : {n_steps}")
    print(f"n_seeds        : {len(seeds)}")
    print(f"antithetic     : {antithetic}")
    print(f"MC mean        : {mc_mean:.6f}")
    print(f"SE (used)      : {se:.6e}")
    print(f"95% CI         : [{ci_left:.6f}, {ci_right:.6f}]")
    print(f"RESULT         : {'PASS' if passed else 'FAIL'}")


def _plot_histogram(prices: np.ndarray, bins: int, mc_mean: float,
    ci_left: float, ci_right: float, 
    bench: float, bench_name: str, n_paths: int) -> None:
    """Plot histogram and reference lines"""

    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()

    # Histogram
    ax.hist(
        prices,
        bins=bins,
        density=True,
        alpha=0.55,
        label="MC prices (across seeds)",
    )

    # Gaussian overlay
    sig = float(sample_std(prices)) if len(prices) > 1 else 0.0
    x = np.linspace(
        min(prices.min(), ci_left, bench),
        max(prices.max(), ci_right, bench),
        600,
    )
    ax.plot(x, normal_pdf(x, mc_mean, sig), linewidth=2.2, label="Gaussian fit")

    # Vertical lines
    ax.axvline(mc_mean, linestyle="--", linewidth=2.0, label=f"MC mean = {mc_mean:.4f}")
    ax.axvline(ci_left, linestyle=":", linewidth=2.0, label="MC mean ± 1.96 SE")
    ax.axvline(ci_right, linestyle=":", linewidth=2.0)
    ax.axvline(bench, color="black", linewidth=2.4, label=f"{bench_name} = {bench:.4f}")

    # Labels
    ax.set_title(f"MC prices across seeds (N={n_paths})", fontsize=13)
    ax.set_xlabel("Monte Carlo price")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# Main plot
def plot_mc_histogram_with_ci(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seeds: list[int], antithetic: bool, bins: int = 30, tree_N: int = 300) -> dict[str, Any]:
    """Run MC across seeds, compute CI, print summary, and plot histogram"""

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    # Antithetic requires even N
    if antithetic and (n_paths % 2 == 1):
        n_paths += 1

    # Benchmark price
    bench, bench_name = benchmark_price(market, trade, tree_N=tree_N)

    # Run MC across seeds
    prices = _run_prices(
        market=market,
        trade=trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seeds=seeds,
        antithetic=antithetic,
    )

    # Compute summary stats
    mc_mean, se, ci_left, ci_right, passed = _compute_summary(
        prices=prices,
        bench=bench,
        antithetic=antithetic,
    )

    # Print results
    _print_summary(
        bench_name=bench_name,
        bench=bench,
        n_paths=n_paths,
        n_steps=n_steps,
        seeds=seeds,
        antithetic=antithetic,
        mc_mean=mc_mean,
        se=se,
        ci_left=ci_left,
        ci_right=ci_right,
        passed=passed,
    )

    # Plot histogram
    _plot_histogram(
        prices=prices,
        bins=bins,
        mc_mean=mc_mean,
        ci_left=ci_left,
        ci_right=ci_right,
        bench=bench,
        bench_name=bench_name,
        n_paths=n_paths,
    )

    # Return all useful outputs
    return {
        "prices": prices,
        "mc_mean": mc_mean,
        "se": se,
        "ci_95": (ci_left, ci_right),
        "bench": bench,
        "bench_name": bench_name,
        "passed": passed,
    }


# Example
if __name__ == "__main__":
    import datetime as dt

    # Dates
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    # Market inputs
    market = Market(S0=100.0, r=0.10, sigma=0.20)

    # Option
    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    # Seeds
    seeds = list(range(1, 501))

    # Run test
    plot_mc_histogram_with_ci(
        market=market,
        trade=trade,
        n_paths=100000,
        n_steps=100,
        seeds=seeds,
        antithetic=True,
        bins=28,
        tree_N=100,
    )