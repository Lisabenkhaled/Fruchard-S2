import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from pricing_tree.adaptateur import tree_price_from_mc


import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price
from utils.utils_stats import sample_mean, sample_std, sample_variance, standard_error


def _norm_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    if sig <= 0:
        return np.zeros_like(x)
    z = (x - mu) / sig
    return (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def _has_dividends(trade: OptionTrade) -> bool:
    return (trade.q != 0.0) or (trade.ex_div_date is not None) or (trade.div_amount != 0.0)

def _benchmark_price(
    market: Market,
    trade: OptionTrade,
    tree_N: int = 2000,
    optimize: bool = False,
    threshold: float = 0.0,
) -> tuple[float, str]:
    
    if not _has_dividends(trade):
        bench = bs_price(
            S=market.S0,
            K=trade.strike,
            r=market.r,
            sigma=market.sigma,
            T=trade.T,
            is_call=trade.is_call,
        )
        return float(bench), "BS"

    out = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=tree_N,
        optimize=optimize,
        threshold=threshold,
        return_tree=False,
    )
    return float(out["tree_price"]), "Tree"


def plot_histogram(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seeds: list[int],
    antithetic: bool = True,
    bins: int = 30,
    tree_N: int = 2000,
    optimize_tree: bool = False,
    threshold_tree: float = 0.0,
):
    
    if trade.exercise.lower() != "european":
        raise ValueError("This convergence test is for EUROPEAN options only.")

    bench, bench_name = _benchmark_price(
        market=market,
        trade=trade,
        tree_N=tree_N,
        optimize=optimize_tree,
        threshold=threshold_tree,
    )

    prices = np.empty(len(seeds), dtype=float)

    for i, sd in enumerate(seeds):
        price, discounted_payoffs = price_european_naive_mc_vector(
            market=market,
            trade=trade,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=sd,
            antithetic=antithetic,
        )
        prices[i] = float(price)

    mc_mean = sample_mean(prices)
    se_bar = standard_error(prices)

    left = mc_mean - 2.0 * se_bar
    right = mc_mean + 2.0 * se_bar
    passed = abs(mc_mean - bench) <= 2.0 * se_bar if se_bar > 0 else (mc_mean == bench)

    print("\n===============================")
    print("Convergence check (EU)")
    print("===============================")
    print(f"Benchmark      : {bench_name}")
    print(f"N paths        : {n_paths}")
    print(f"n_steps        : {n_steps}")
    print(f"n_seeds        : {len(seeds)}")
    print(f"antithetic     : {antithetic}")
    print(f"MC mean        : {mc_mean:.6f}")
    print(f"Avg SE (utils) : {se_bar:.6f}")
    print(f"{bench_name} price   : {bench:.6f}")
    print(f"|mean-bench|   : {abs(mc_mean - bench):.6f}")
    print(f"2*SE           : {2.0*se_bar:.6f}")
    print("RESULT         :", "PASS" if passed else "FAIL")

    # Plot
    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()

    ax.hist(prices, bins=bins, density=True, alpha=0.55, label="MC prices (across seeds)")

    # Gaussian overlay uses std across seeds
    std_prices = sample_std(prices) if len(prices) > 1 else 0.0
    x_min = min(prices.min(), left, bench) - 0.02 * max(1.0, abs(mc_mean))
    x_max = max(prices.max(), right, bench) + 0.02 * max(1.0, abs(mc_mean))
    x = np.linspace(x_min, x_max, 600)
    ax.plot(x, _norm_pdf(x, mc_mean, std_prices), linewidth=2.2, label="Gaussian fit (across seeds)")

    ax.axvline(mc_mean, linestyle="--", linewidth=2.0, label=f"MC mean = {mc_mean:.4f}")
    ax.axvline(bench, color="black", linewidth=2.2, label=f"{bench_name} = {bench:.4f}")

    # 2SE band 
    ax.axvline(left, linestyle=":", linewidth=1.5, label="2SE (avg over seeds)")
    ax.axvline(right, linestyle=":", linewidth=1.5)

    ax.set_title(f"MC prices across seeds (N={n_paths})", fontsize=13)
    ax.set_xlabel("Monte Carlo price")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return {
        "prices": prices,
        "mc_mean": mc_mean,
        "se_bar": se_bar,
        "bench": bench,
        "bench_name": bench_name,
        "passed": passed,
    }

import datetime as dt

from model.market import Market
from model.option import OptionTrade

# Dates
pricing_date = dt.date(2026, 2, 18)
maturity_date = dt.date(2027, 2, 18)

# Market
market = Market(
    S0=100.0,
    r=0.05,
    sigma=0.30
)

trade = OptionTrade(
    strike=102.0,
    is_call=False,
    exercise="european",
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    q=0.0,
    ex_div_date=None,
    div_amount=0.0
)

seeds = list(range(1, 501)) 

plot_histogram(
    market=market,
    trade=trade,
    n_paths=10_000,
    n_steps=100,
    seeds=seeds,
    antithetic=True,
    bins=28
)