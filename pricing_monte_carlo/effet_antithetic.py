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


def make_smooth_n_paths_grid(
    n_min: int = 2000,
    n_max: int = 200000,
    n_points: int = 25,
    force_even: bool = True
) -> np.ndarray:
    """Create a logarithmic grid of Monte Carlo path counts."""
    grid = np.unique(
        np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int)
    )
    if force_even:
        grid = np.where(grid % 2 == 0, grid, grid + 1)
        grid = np.unique(grid)
    return grid


def run_mc_simulation(
    ex: str,
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool,
    basis: str,
    degree: int
) -> tuple[float, np.ndarray, float]:
    """Run one Monte Carlo pricing simulation."""
    t0 = time.perf_counter()

    if ex == "european":
        price, payoff = price_european_naive_mc_vector(
            market, trade, int(n_paths), n_steps, seed, antithetic
        )
    else:
        price, payoff = price_american_ls_vector(
            market=market,
            trade=trade,
            n_paths=int(n_paths),
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
            basis=basis,
            degree=degree,
        )

    t1 = time.perf_counter()
    return float(price), payoff, float(t1 - t0)


def _print_header(title: str) -> None:
    """Print the benchmark table header."""
    print("\n" + "=" * 120)
    print(f"{title} — Non-Antithetic vs Antithetic")
    print("=" * 120)
    print(
        "Paths | Price(No)  Std(No)    SE(No)     Time(No) | "
        "Price(Anti) Std(Anti) SE(Anti)  Time(Anti) | VRF"
    )
    print("-" * 120)


def _plot_se(
    Ns: np.ndarray,
    se_no_arr: np.ndarray,
    se_anti_arr: np.ndarray,
    title: str
) -> None:
    """Plot standard errors."""
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


def _plot_vrf(
    Ns: np.ndarray,
    vrf_arr: np.ndarray,
    title: str
) -> None:
    """Plot variance reduction factor."""
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


def _plot_time(
    Ns: np.ndarray,
    time_no_arr: np.ndarray,
    time_anti_arr: np.ndarray,
    title: str
) -> None:
    """Plot execution times."""
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


def plot_benchmark_results(
    Ns: np.ndarray,
    se_no_arr: np.ndarray,
    se_anti_arr: np.ndarray,
    time_no_arr: np.ndarray,
    time_anti_arr: np.ndarray,
    vrf_arr: np.ndarray,
    title: str
) -> None:
    """Plot all benchmark results."""
    _plot_se(Ns, se_no_arr, se_anti_arr, title)

    # Second chart: VRF.
    _plot_vrf(Ns, vrf_arr, title)

    # Third chart: execution time.
    _plot_time(Ns, time_no_arr, time_anti_arr, title)


def _prepare_grid(n_paths_grid: np.ndarray) -> np.ndarray:
    """Ensure the path grid is integer, unique, and even."""
    grid = np.array(list(map(int, n_paths_grid)), dtype=int)
    grid = np.where(grid % 2 == 0, grid, grid + 1)
    return np.unique(grid)


def _benchmark_one_n(
    ex: str,
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int,
    basis: str,
    degree: int
) -> tuple[float, float, float, float, float, float, float]:
    """Run both simulations for one value of n_paths."""
    price_no, payoff_no, time_no = run_mc_simulation(
        ex, market, trade, n_paths, n_steps, seed, False, basis, degree
    )
    std_no = sample_std(payoff_no)
    se_no = standard_error(payoff_no)

    # Antithetic run.
    price_anti, payoff_anti, time_anti = run_mc_simulation(
        ex, market, trade, n_paths, n_steps, seed, True, basis, degree
    )
    std_anti = sample_std_anti(payoff_anti)
    se_anti = standard_error_anti(payoff_anti)

    vrf = (se_no / se_anti) ** 2 if se_anti > 0 else np.nan
    return price_no, std_no, se_no, time_no, price_anti, std_anti, se_anti, time_anti, vrf


def benchmark_noanti_vs_anti(
    market: Market,
    trade: OptionTrade,
    n_paths_grid: np.ndarray,
    n_steps: int,
    seed: int = 42,
    basis: str = "laguerre",
    degree: int = 2
) -> None:
    """Compare non-antithetic vs antithetic Monte Carlo."""
    ex = trade.exercise.lower()

    # Prepare the grid of path counts.
    n_paths_grid = _prepare_grid(n_paths_grid)
    title = "EU" if ex == "european" else f"AM LS (basis={basis}, degree={degree})"
    _print_header(title)

    Ns = []
    se_no_list, se_anti_list = [], []
    time_no_list, time_anti_list = [], []
    vrf_list = []

    for n_paths in n_paths_grid:
        result = _benchmark_one_n(ex, market, trade, n_paths, n_steps, seed, basis, degree)
        price_no, std_no, se_no, time_no = result[0], result[1], result[2], result[3]
        price_anti, std_anti, se_anti, time_anti, vrf = result[4], result[5], result[6], result[7], result[8]

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

    # Plot
    plot_benchmark_results(
        Ns=np.array(Ns),
        se_no_arr=np.array(se_no_list),
        se_anti_arr=np.array(se_anti_list),
        time_no_arr=np.array(time_no_list),
        time_anti_arr=np.array(time_anti_list),
        vrf_arr=np.array(vrf_list),
        title=title,
    )


if __name__ == "__main__":
    import datetime as dt

    n_paths_grid = make_smooth_n_paths_grid(
        n_min=1000,
        n_max=100000,
        n_points=30,
    )

    market = Market(S0=100.0, r=0.03, sigma=0.25)

    # European example
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
        seed=42
    )

    # American LS example
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
        degree=2
    )