import time
import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import (
    price_european_naive_mc_vector,
    price_european_naive_mc_scalar,
)
from utils.utils_stats import standard_error


def make_smooth_n_paths_grid(n_min=2_000, n_max=200_000, n_points=25, force_even=True):
    grid = np.unique(
        np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int)
    )
    if force_even:
        grid = np.where(grid % 2 == 0, grid, grid + 1)
        grid = np.unique(grid)
    return grid


def benchmark_scalar_vs_vector_eu(
    market: Market,
    trade: OptionTrade,
    n_paths_grid,
    n_steps: int,
    seed: int = 42,
    antithetic: bool = True,
    plot: bool = True,
):
    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    n_paths_grid = np.array(list(map(int, n_paths_grid)), dtype=int)
    if antithetic:
        n_paths_grid = np.where(n_paths_grid % 2 == 0, n_paths_grid, n_paths_grid + 1)
        n_paths_grid = np.unique(n_paths_grid)

    print("\n" + "=" * 92)
    print("EU Scalar vs Vector Benchmark (Price + SE + Time)")
    print("=" * 92)
    print("Paths | Price(Vec) | SE(Vec)   | Time(Vec) | Price(Sca) | SE(Sca)   | Time(Sca) | Speedup")
    print("-" * 92)

    times_vec, times_sca = [], []

    for n_paths in n_paths_grid:

        # Vectorized version
        t0 = time.perf_counter()
        price_vec, disc_payoff_vec = price_european_naive_mc_vector(
            market, trade, n_paths, n_steps, seed, antithetic
        )
        t1 = time.perf_counter()

        se_vec = standard_error(disc_payoff_vec)
        time_vec = t1 - t0

        # Scalar version
        t0 = time.perf_counter()
        price_sca, disc_payoff_sca = price_european_naive_mc_scalar(
            market, trade, n_paths, n_steps, seed, antithetic
        )
        t1 = time.perf_counter()

        se_sca = standard_error(disc_payoff_sca)
        time_sca = t1 - t0

        speedup = time_sca / time_vec if time_vec > 0 else np.nan

        print(f"{int(n_paths):>6d} | "
              f"{price_vec:>10.6f} | {se_vec:>9.6f} | {time_vec:>8.4f} | "
              f"{price_sca:>10.6f} | {se_sca:>9.6f} | {time_sca:>8.4f} | "
              f"{speedup:>7.2f}x")

        times_vec.append(time_vec)
        times_sca.append(time_sca)

    print("-" * 92)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(n_paths_grid, times_vec, marker="o", linewidth=2, label="Vector")
        plt.plot(n_paths_grid, times_sca, marker="s", linewidth=2, label="Scalar")
        plt.xscale("log")
        plt.title("Execution Time vs Number of Paths (European MC)")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Execution time (seconds)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 2, 18)
    maturity_date = dt.date(2026, 8, 18)

    market = Market(
        S0=100.0,
        r=0.03,
        sigma=0.25
    )

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="European",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0
    )

    n_paths_grid = make_smooth_n_paths_grid(
        n_min=1_000,
        n_max=100_000,
        n_points=30,
        force_even=True 
    )

    benchmark_scalar_vs_vector_eu(
        market=market,
        trade=trade,
        n_paths_grid=n_paths_grid,
        n_steps=300, 
        seed=42,
        antithetic=True,
        plot=True
    )
    