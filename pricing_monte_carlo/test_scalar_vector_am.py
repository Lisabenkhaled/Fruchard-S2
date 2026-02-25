import time
import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import (
    price_american_ls_vector,
    price_american_ls_scalar,
)
from utils.utils_stats import standard_error
from test_scalar_vector_eu import make_smooth_n_paths_grid


def benchmark_scalar_vs_vector_am_ls(
    market,
    trade,
    n_paths_grid,
    n_steps,
    seed=42,
    antithetic=True,
    basis="laguerre",
    degree=2,
    plot=True,
):
    if trade.exercise.lower() != "american":
        raise ValueError("AMERICAN options only.")

    n_paths_grid = np.array(list(map(int, n_paths_grid)), dtype=int)
    if antithetic:
        n_paths_grid = np.where(n_paths_grid % 2 == 0, n_paths_grid, n_paths_grid + 1)
        n_paths_grid = np.unique(n_paths_grid)

    print("\n" + "=" * 95)
    print("AM LS Scalar vs Vector Benchmark (Price + SE + Time)")
    print("=" * 95)
    print("Paths | Price(Vec) | SE(Vec)   | Time(Vec) | Price(Sca) | SE(Sca)   | Time(Sca) | Speedup")
    print("-" * 95)

    times_vec, times_sca = [], []

    for n_paths in n_paths_grid:

        # VECTOR
        t0 = time.perf_counter()
        price_vec, cf_vec = price_american_ls_vector(
            market, trade, n_paths, n_steps,
            seed, antithetic, basis, degree
        )
        t1 = time.perf_counter()
        se_vec = standard_error(cf_vec)
        time_vec = t1 - t0

        # SCALAR
        t0 = time.perf_counter()
        price_sca, cf_sca = price_american_ls_scalar(
            market, trade, n_paths, n_steps,
            seed, antithetic, basis, degree
        )
        t1 = time.perf_counter()
        se_sca = standard_error(cf_sca)
        time_sca = t1 - t0

        speedup = time_sca / time_vec if time_vec > 0 else np.nan

        print(f"{int(n_paths):>6d} | "
              f"{price_vec:>10.6f} | {se_vec:>9.6f} | {time_vec:>8.4f} | "
              f"{price_sca:>10.6f} | {se_sca:>9.6f} | {time_sca:>8.4f} | "
              f"{speedup:>7.2f}x")

        times_vec.append(time_vec)
        times_sca.append(time_sca)

    print("-" * 95)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(n_paths_grid, times_vec, marker="o", label="Vector")
        plt.plot(n_paths_grid, times_sca, marker="s", label="Scalar")
        plt.xscale("log")
        plt.title("Execution Time vs Number of Paths (AM LS)")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Execution time (seconds)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    market = Market(S0=100.0, r=0.05, sigma=0.20)

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0
    )

    n_paths_grid = make_smooth_n_paths_grid(
        n_min=1_000,
        n_max=100_000,
        n_points=25,
        force_even=True
    )

    benchmark_scalar_vs_vector_am_ls(
        market,
        trade,
        n_paths_grid,
        n_steps=200,
        seed=42,
        antithetic=True,
        basis="laguerre",
        degree=2,
        plot=True
    )