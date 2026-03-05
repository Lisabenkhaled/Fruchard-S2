import time
import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import (
    price_european_naive_mc_vector,
    price_american_ls_vector,
)

from utils.utils_stats import (
    sample_std,
    standard_error,
    sample_std_anti,
    standard_error_anti,
)


def make_smooth_n_paths_grid(n_min=2_000, n_max=200_000, n_points=25, force_even=True):
    grid = np.unique(
        np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int)
    )
    if force_even:
        grid = np.where(grid % 2 == 0, grid, grid + 1)
        grid = np.unique(grid)
    return grid


def benchmark_noanti_vs_anti(
    market: Market,
    trade: OptionTrade,
    n_paths_grid,
    n_steps: int,
    seed: int = 42,
    basis: str = "laguerre",   # only used for AM LS
    degree: int = 2,          # only used for AM LS
    plot: bool = True,
):
    ex = trade.exercise.lower()
    if ex not in ("european", "american"):
        raise ValueError("trade.exercise must be 'European' or 'American'.")

    n_paths_grid = np.array(list(map(int, n_paths_grid)), dtype=int)
    n_paths_grid = np.where(n_paths_grid % 2 == 0, n_paths_grid, n_paths_grid + 1)  # even for antithetic
    n_paths_grid = np.unique(n_paths_grid)

    title = "EU" if ex == "european" else f"AM LS (basis={basis}, degree={degree})"

    print("\n" + "=" * 120)
    print(f"{title} — Non-Antithetic vs Antithetic (Price + Std + SE + Time + VRF)")
    print("=" * 120)
    print(
        "Paths | Price(No)  Std(No)    SE(No)     Time(No) | "
        "Price(Anti) Std(Anti) SE(Anti)  Time(Anti) | VRF"
    )
    print("-" * 120)

    Ns = []
    se_no_list, se_anti_list = [], []
    time_no_list, time_anti_list = [], []
    vrf_list = []

    for n_paths in n_paths_grid:

        # -------------------------
        # Non-antithetic
        # -------------------------
        t0 = time.perf_counter()
        if ex == "european":
            price_no, payoff_no = price_european_naive_mc_vector(
                market, trade, int(n_paths), n_steps, seed, False
            )
        else:
            price_no, payoff_no = price_american_ls_vector(
                market=market,
                trade=trade,
                n_paths=int(n_paths),
                n_steps=n_steps,
                seed=seed,
                antithetic=False,
                basis=basis,
                degree=degree,
            )
        t1 = time.perf_counter()

        std_no = sample_std(payoff_no)
        se_no = standard_error(payoff_no)
        time_no = t1 - t0

        # -------------------------
        # Antithetic
        # -------------------------
        t0 = time.perf_counter()
        if ex == "european":
            price_anti, payoff_anti = price_european_naive_mc_vector(
                market, trade, int(n_paths), n_steps, seed, True
            )
        else:
            price_anti, payoff_anti = price_american_ls_vector(
                market=market,
                trade=trade,
                n_paths=int(n_paths),
                n_steps=n_steps,
                seed=seed,
                antithetic=True,
                basis=basis,
                degree=degree,
            )
        t1 = time.perf_counter()

        std_anti = sample_std_anti(payoff_anti)
        se_anti = standard_error_anti(payoff_anti)
        time_anti = t1 - t0

        vrf = (se_no / se_anti) ** 2 if se_anti > 0 else np.nan

        print(
            f"{int(n_paths):>6d} | "
            f"{price_no:>9.6f} {std_no:>9.3e} {se_no:>9.3e} {time_no:>9.4f} | "
            f"{price_anti:>9.6f} {std_anti:>9.3e} {se_anti:>9.3e} {time_anti:>9.4f} | "
            f"{vrf:>6.2f}"
        )

        Ns.append(n_paths)
        se_no_list.append(se_no)
        se_anti_list.append(se_anti)
        time_no_list.append(time_no)
        time_anti_list.append(time_anti)
        vrf_list.append(vrf)

    print("-" * 120)

    if plot:
        Ns = np.array(Ns)
        se_no_arr = np.array(se_no_list)
        se_anti_arr = np.array(se_anti_list)
        time_no_arr = np.array(time_no_list)
        time_anti_arr = np.array(time_anti_list)
        vrf_arr = np.array(vrf_list)

        # SE vs N
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, se_no_arr, marker="o", label="SE (No antithetic)")
        plt.plot(Ns, se_anti_arr, marker="s", label="SE (Antithetic)")
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"{title} — Standard Error vs N")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Standard Error")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # VRF vs N
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, vrf_arr, marker="o")
        plt.axhline(1.0, linestyle="--")
        plt.xscale("log")
        plt.title(f"{title} — VRF vs N")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("VRF = (SE_no / SE_anti)^2")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Time vs N
        plt.figure(figsize=(10, 6))
        plt.plot(Ns, time_no_arr, marker="o", label="No antithetic")
        plt.plot(Ns, time_anti_arr, marker="s", label="Antithetic")
        plt.xscale("log")
        plt.title(f"{title} — Time vs N")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Execution time (seconds)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import datetime as dt

    n_paths_grid = make_smooth_n_paths_grid(n_min=1_000, n_max=100_000, n_points=30)

    market = Market(S0=100.0, r=0.03, sigma=0.25)

    # -------------------------
    # Example 1: European
    # -------------------------
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

    benchmark_noanti_vs_anti(
        market=market,
        trade=eu_trade,
        n_paths_grid=n_paths_grid,
        n_steps=300,
        seed=42,
        plot=True,
    )

    # -------------------------
    # Example 2: American LS
    # -------------------------
    am_trade = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=dt.date(2026, 2, 18),
        maturity_date=dt.date(2027, 2, 18),
        q=0.0,
        ex_div_date=None,
        div_amount=0.0,
    )

    benchmark_noanti_vs_anti(
        market=market,
        trade=am_trade,
        n_paths_grid=n_paths_grid,
        n_steps=100,
        seed=42,
        basis="laguerre",
        degree=2,
        plot=True,
    )