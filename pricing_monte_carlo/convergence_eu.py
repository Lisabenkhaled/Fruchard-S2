import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Project path (keep if needed in your repo) ---
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


# -------------------------
# Helpers
# -------------------------
def has_discrete_dividend(trade: OptionTrade) -> bool:
    """True if the trade has a discrete dividend (ex-div date + amount)."""
    return (trade.ex_div_date is not None) and (trade.div_amount != 0.0)


def benchmark_price(
    market: Market,
    trade: OptionTrade,
    tree_N: int = 1000,
) -> tuple[float, str]:
    """
    If no discrete dividend -> use Black-Scholes (European).
    If discrete dividend -> use your tree benchmark.
    """
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
    if sig <= 0:
        return np.zeros_like(x)
    z = (x - mu) / sig
    return (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


# -------------------------
# Main plot
# -------------------------
def plot_mc_histogram_with_ci(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seeds: list[int],
    antithetic: bool,
    bins: int = 30,
    tree_N: int = 300,
):

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    # Antithetic requires even paths
    if antithetic and (n_paths % 2 == 1):
        n_paths += 1

    bench, bench_name = benchmark_price(market, trade, tree_N=tree_N)

    # --- Run MC across seeds ---
    prices = np.empty(len(seeds), dtype=float)
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

    mc_mean = float(sample_mean(prices))

    # CI width uses YOUR formulas (anti vs no-anti)
    se = float(standard_error_anti(prices) if antithetic else standard_error(prices))
    ci_left = mc_mean - 1.96 * se
    ci_right = mc_mean + 1.96 * se

    passed = abs(mc_mean - bench) <= 1.96 * se if se > 0 else (mc_mean == bench)

    # --- Print summary ---
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

    # --- Plot ---
    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()

    ax.hist(prices, bins=bins, density=True, alpha=0.55, label="MC prices (across seeds)")

    # Gaussian overlay uses std across seeds (just for visual fit)
    sig = float(sample_std(prices)) if len(prices) > 1 else 0.0
    x = np.linspace(min(prices.min(), ci_left, bench), max(prices.max(), ci_right, bench), 600)
    ax.plot(x, normal_pdf(x, mc_mean, sig), linewidth=2.2, label="Gaussian fit")

    # Lines: mean, CI, benchmark
    ax.axvline(mc_mean, linestyle="--", linewidth=2.0, label=f"MC mean = {mc_mean:.4f}")
    ax.axvline(ci_left, linestyle=":", linewidth=2.0, label="MC mean ± 1.96 SE")
    ax.axvline(ci_right, linestyle=":", linewidth=2.0)
    ax.axvline(bench, color="black", linewidth=2.4, label=f"{bench_name} = {bench:.4f}")

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
        "se": se,
        "ci_95": (ci_left, ci_right),
        "bench": bench,
        "bench_name": bench_name,
        "passed": passed,
    }


# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 2, 18)
    maturity_date = dt.date(2027, 2, 18)

    market = Market(S0=100.0, r=0.05, sigma=0.30)

    trade = OptionTrade(
        strike=102.0,
        is_call=False,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 9, 18),
        div_amount=3.0,
    )

    seeds = list(range(1, 501))

    plot_mc_histogram_with_ci(
        market=market,
        trade=trade,
        n_paths=10_000,
        n_steps=500,
        seeds=seeds,
        antithetic=True,
        bins=28,
        tree_N=500,
    )