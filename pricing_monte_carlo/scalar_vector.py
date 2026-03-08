from __future__ import annotations

import time
from typing import Callable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import (
    price_european_naive_mc_scalar,
    price_european_naive_mc_vector,
    price_american_naive_mc_vector,
    price_american_naive_mc_scalar,
    price_american_ls_vector,
    price_american_ls_scalar,
)

from utils.utils_stats import sample_std_anti, standard_error_anti

# Types
Samples = NDArray[np.float64]
RunFn = Callable[[int], Tuple[float, Samples]]


# Helpers
def make_n_paths_grid(n_min: int = 1_000, n_max: int = 100_000, n_points: int = 25) -> NDArray[np.int64]:
    grid = np.unique(
        np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int)
    )
    grid = np.where(grid % 2 == 0, grid, grid + 1)  # force even
    return np.unique(grid).astype(np.int64)


def _print_header(title: str) -> None:
    """Print the benchmark table header"""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(
        "Paths | Price(Vec) | Std(Vec)  | SE(Vec)   | Time(Vec) | "
        "Price(Sca) | Std(Sca)  | SE(Sca)   | Time(Sca) | Speedup"
    )
    print("-" * 100)


def _run_one(pricer: RunFn, n_paths: int) -> Tuple[float, float, float, float]:
    """
    Run one pricer and return:
      price, std, se, elapsed_time
    """
    t0 = time.perf_counter()
    price, samples = pricer(int(n_paths))
    t1 = time.perf_counter()

    std = float(sample_std_anti(samples))
    se = float(standard_error_anti(samples))
    elapsed = float(t1 - t0)

    return float(price), std, se, elapsed


def _print_row(n_paths: int, p_vec: float, std_vec: float, se_vec: float,t_vec: float, 
               p_sca: float, std_sca: float, se_sca: float, t_sca: float) -> None:
    """Print one line of the benchmark table"""
    speedup = (t_sca / t_vec) if t_vec > 0 else float("nan")

    print(
        f"{int(n_paths):>6d} | "
        f"{p_vec:>10.6f} | {std_vec:>9.6f} | {se_vec:>9.6f} | {t_vec:>8.4f} | "
        f"{p_sca:>10.6f} | {std_sca:>9.6f} | {se_sca:>9.6f} | {t_sca:>8.4f} | "
        f"{speedup:>7.2f}x"
    )


def _plot_times(title: str, n_paths_grid: Sequence[int], t_vec: Sequence[float], t_sca: Sequence[float]) -> None:
    """Plot execution time vs N"""
    plt.figure(figsize=(10, 6))
    plt.plot(n_paths_grid, t_vec, marker="o", label="Vector")
    plt.plot(n_paths_grid, t_sca, marker="s", label="Scalar")
    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Execution time (seconds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_benchmark(title: str, n_paths_grid: Sequence[int], run_vec: RunFn, run_sca: RunFn, plot: bool = True) -> None:
    _print_header(title)

    t_vec_list: List[float] = []
    t_sca_list: List[float] = []

    # Main loop over N
    for n_paths in n_paths_grid:
        p_vec, std_vec, se_vec, t_vec = _run_one(run_vec, int(n_paths))
        p_sca, std_sca, se_sca, t_sca = _run_one(run_sca, int(n_paths))

        _print_row(
            n_paths=int(n_paths),
            p_vec=p_vec,
            std_vec=std_vec,
            se_vec=se_vec,
            t_vec=t_vec,
            p_sca=p_sca,
            std_sca=std_sca,
            se_sca=se_sca,
            t_sca=t_sca,
        )

        t_vec_list.append(t_vec)
        t_sca_list.append(t_sca)

    print("-" * 100)

    if plot:
        _plot_times(title, n_paths_grid, t_vec_list, t_sca_list)


# EU Options
def bench_european(market: Market, trade: OptionTrade, n_paths_grid: Sequence[int], n_steps: int,
    seed: int = 42, plot: bool = True) -> None:
    """European naive MC: scalar vs vector"""

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

# AM Naive
def bench_american_naive(market: Market, trade: OptionTrade, n_paths_grid: Sequence[int], n_steps: int,
    seed: int = 42, plot: bool = True) -> None:
    """American naive MC: scalar vs vector"""

    if trade.exercise.lower() != "american":
        raise ValueError("American trade required")

    antithetic = True

    run_benchmark(
        title="AMERICAN NAIVE MC — Scalar vs Vector (Antithetic)",
        n_paths_grid=n_paths_grid,
        run_vec=lambda n: price_american_naive_mc_vector(market, trade, n, n_steps, seed, antithetic),
        run_sca=lambda n: price_american_naive_mc_scalar(market, trade, n, n_steps, seed, antithetic),
        plot=plot,
    )

# AM LS
def bench_american_ls(market: Market, trade: OptionTrade, n_paths_grid: Sequence[int], n_steps: int,
    seed: int = 42, basis: str = "laguerre", degree: int = 2, plot: bool = True) -> None:
    """American LS MC: scalar vs vector"""

    if trade.exercise.lower() != "american":
        raise ValueError("American trade required")

    antithetic = True

    run_benchmark(
        title=f"AMERICAN LS MC — Scalar vs Vector (Antithetic, {basis=}, {degree=})",
        n_paths_grid=n_paths_grid,
        run_vec=lambda n: price_american_ls_vector(market, trade, n, n_steps, seed, antithetic, basis, degree),
        run_sca=lambda n: price_american_ls_scalar(market, trade, n, n_steps, seed, antithetic, basis, degree),
        plot=plot,
    )


# Example usage
def main() -> None:
    import datetime as dt

    n_paths_grid = make_n_paths_grid(n_min=1_000, n_max=100_000, n_points=30)
    market = Market(S0=100.0, r=0.10, sigma=0.20)

    # EU Options
    eu_trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=dt.date(2026, 3, 1),
        maturity_date=dt.date(2026, 12, 25),
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    # AM Naive
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

    # AM LS
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

    # Calculation and Plot
    bench_european(market, eu_trade, n_paths_grid, n_steps=300, seed=42, plot=True)
    bench_american_naive(market, am_trade_naive, n_paths_grid, n_steps=200, seed=42, plot=True)
    bench_american_ls(market, am_trade_ls, n_paths_grid, n_steps=200, seed=42, basis="laguerre", degree=2, plot=True)

if __name__ == "__main__":
    main()