"""
benchmark_scalar_vs_vector_all.py

Simple one-file benchmark: Scalar vs Vector for
  1) European naive MC
  2) American naive MC
  3) American Longstaff–Schwartz (LS) MC

ONLY antithetic is used.
SE is computed with standard_error_anti (and std with sample_std_anti).

Expected pricer outputs (as in your code):
- European: (price, discounted_payoffs)
- American naive: (price, best_pv_by_path)
- American LS: (price, discounted_cashflows)
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import (
    # European
    price_european_naive_mc_vector,
    price_european_naive_mc_scalar,
    # American naive
    price_american_naive_mc_vector,
    price_american_naive_mc_scalar,
    # American LS
    price_american_ls_vector,
    price_american_ls_scalar,
)
from utils.utils_stats import sample_std_anti, standard_error_anti


# -------------------------
# Helpers
# -------------------------
def make_n_paths_grid(n_min=1_000, n_max=100_000, n_points=25):
    """Log-spaced grid, forced EVEN because we always use antithetic."""
    grid = np.unique(
        np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int)
    )
    grid = np.where(grid % 2 == 0, grid, grid + 1)  # force even
    return np.unique(grid)


def run_benchmark(title, n_paths_grid, run_vec, run_sca, plot=True):
    """
    run_vec(n_paths) -> (price, samples_for_stats)
    run_sca(n_paths) -> (price, samples_for_stats)
    """
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("Paths | Price(Vec) | Std(Vec)  | SE(Vec)   | Time(Vec) | Price(Sca) | Std(Sca)  | SE(Sca)   | Time(Sca) | Speedup")
    print("-" * 100)

    t_vec_list, t_sca_list = [], []

    for n_paths in n_paths_grid:
        # Vector
        t0 = time.perf_counter()
        p_vec, x_vec = run_vec(int(n_paths))
        t1 = time.perf_counter()
        std_vec = float(sample_std_anti(x_vec))
        se_vec = float(standard_error_anti(x_vec))
        t_vec = float(t1 - t0)

        # Scalar
        t0 = time.perf_counter()
        p_sca, x_sca = run_sca(int(n_paths))
        t1 = time.perf_counter()
        std_sca = float(sample_std_anti(x_sca))
        se_sca = float(standard_error_anti(x_sca))
        t_sca = float(t1 - t0)

        speedup = t_sca / t_vec if t_vec > 0 else np.nan

        print(
            f"{int(n_paths):>6d} | "
            f"{float(p_vec):>10.6f} | {std_vec:>9.6f} | {se_vec:>9.6f} | {t_vec:>8.4f} | "
            f"{float(p_sca):>10.6f} | {std_sca:>9.6f} | {se_sca:>9.6f} | {t_sca:>8.4f} | "
            f"{speedup:>7.2f}x"
        )

        t_vec_list.append(t_vec)
        t_sca_list.append(t_sca)

    print("-" * 100)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(n_paths_grid, t_vec_list, marker="o", label="Vector")
        plt.plot(n_paths_grid, t_sca_list, marker="s", label="Scalar")
        plt.xscale("log")
        plt.title(title)
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Execution time (seconds)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# -------------------------
# Benchmarks
# -------------------------
def bench_european(market, trade, n_paths_grid, n_steps, seed=42, plot=True):
    if trade.exercise.lower() != "european":
        raise ValueError("European trade required")

    antithetic = True

    run_benchmark(
        title="EUROPEAN NAIVE MC — Scalar vs Vector (Antithetic)",
        n_paths_grid=n_paths_grid,
        run_vec=lambda n: price_european_naive_mc_vector(market, trade, n, n_steps, seed, antithetic),
        run_sca=lambda n: price_european_naive_mc_scalar(market, trade, n, n_steps, seed, antithetic),
        plot=plot,
    )


def bench_american_naive(market, trade, n_paths_grid, n_steps, seed=42, plot=True):
    if trade.exercise.lower() != "american":
        raise ValueError("American trade required")

    antithetic = True  # ALWAYS

    run_benchmark(
        title="AMERICAN NAIVE MC — Scalar vs Vector (Antithetic)",
        n_paths_grid=n_paths_grid,
        run_vec=lambda n: price_american_naive_mc_vector(market, trade, n, n_steps, seed, antithetic),
        run_sca=lambda n: price_american_naive_mc_scalar(market, trade, n, n_steps, seed, antithetic),
        plot=plot,
    )


def bench_american_ls(market, trade, n_paths_grid, n_steps, seed=42, basis="laguerre", degree=2, plot=True):
    if trade.exercise.lower() != "american":
        raise ValueError("American trade required")

    antithetic = True  # ALWAYS

    run_benchmark(
        title=f"AMERICAN LS MC — Scalar vs Vector (Antithetic, {basis=}, {degree=})",
        n_paths_grid=n_paths_grid,
        run_vec=lambda n: price_american_ls_vector(market, trade, n, n_steps, seed, antithetic, basis, degree),
        run_sca=lambda n: price_american_ls_scalar(market, trade, n, n_steps, seed, antithetic, basis, degree),
        plot=plot,
    )


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import datetime as dt

    n_paths_grid = make_n_paths_grid(n_min=1_000, n_max=100_000, n_points=30)

    market = Market(S0=100.0, r=0.05, sigma=0.20)

    eu_trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="European",
        pricing_date=dt.date(2026, 2, 18),
        maturity_date=dt.date(2026, 8, 18),
        q=0.0,
        ex_div_date=dt.date(2026, 3, 18),
        div_amount=3.0,
    )

    am_trade_naive = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    am_trade_ls = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        q=0.0,
        ex_div_date=None,
        div_amount=0.0,
    )

    bench_european(market, eu_trade, n_paths_grid, n_steps=300, seed=42, plot=True)
    bench_american_naive(market, am_trade_naive, n_paths_grid, n_steps=200, seed=42, plot=True)
    bench_american_ls(market, am_trade_ls, n_paths_grid, n_steps=200, seed=42, basis="laguerre", degree=2, plot=True)