from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path


import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Repo root
# =========================================================
def _add_repo_root_to_syspath() -> None:
    here: Path = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pricing_monte_carlo").is_dir() and (parent / "pricing_tree").is_dir():
            p: str = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        f"Repo root not found from {here}. Expected pricing_monte_carlo/ and pricing_tree/."
    )


_add_repo_root_to_syspath()


# =========================================================
# Imports projet
# =========================================================
from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.core_pricer import CorePricingParams
from pricing_monte_carlo.core_greeks import core_greeks


# =========================================================
# Main function
# =========================================================
def delta_convergence_table(
    market: Market,
    trade: OptionTrade,
    base_params: CorePricingParams,
    n_paths_list: list[int],
    shift_spot: float | None = None,
    shift_vol: float = 0.01,
    tree_N: int = 300,
    plot: bool = True
) -> pd.DataFrame:
    """
    Convergence du delta avec :
    - une seule seed
    - plusieurs N
    - tableau final : N, Delta MC, Delta Bench, Diff
    """
    rows = []

    for n_paths in n_paths_list:
        # Override only n_paths; all other params stay identical across runs
        params = CorePricingParams(
            n_paths=n_paths,
            n_steps=base_params.n_steps,
            seed=base_params.seed,
            antithetic=base_params.antithetic,
            method=base_params.method,
            american_algo=base_params.american_algo,
            basis=base_params.basis,
            degree=base_params.degree,
        )

        # Compute MC delta and benchmark delta (Black-Scholes or tree)
        mc_greeks, bench_greeks = core_greeks(
            market=market,
            trade=trade,
            params=params,
            shift_spot=shift_spot,
            shift_vol=shift_vol,
            tree_N=tree_N,
        )

        # Extract deltas and compute signed difference MC - Bench
        delta_mc: float = float(mc_greeks["delta"])
        delta_bench: float = float(bench_greeks["delta"])
        diff: float = delta_mc - delta_bench

        # Append one row per N value
        rows.append({
            "N": n_paths,
            "Delta MC": delta_mc,
            "Delta Bench": delta_bench,
            "Diff": diff,
        })

    # Build results DataFrame from collected rows
    df = pd.DataFrame(rows)

    # Print formatted convergence table to stdout
    print("\n" + "=" * 70)
    print(f"Delta convergence table | {trade.exercise.upper()} | div={getattr(trade, 'div_amount', 0.0)}")
    print("=" * 70)
    print(df.to_string(index=False))

    if plot:
        plot_delta_convergence(df, trade)

    return df


# =========================================================
# Plot
# =========================================================
def _plot_delta_levels(df: pd.DataFrame, trade: OptionTrade) -> None:
    """Plot MC delta and benchmark delta against number of paths (log scale)."""
    plt.figure(figsize=(8, 5))
    plt.plot(df["N"], df["Delta MC"], marker="o", label="Delta MC")
    plt.plot(df["N"], df["Delta Bench"], marker="o", label="Delta Bench")
    plt.xscale("log")
    plt.xlabel("Number of paths N")
    plt.ylabel("Delta")
    plt.title(f"Delta convergence - {trade.exercise}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_delta_diff(df: pd.DataFrame, trade: OptionTrade) -> None:
    """Plot signed difference (MC delta - benchmark delta) against N (log scale)."""
    plt.figure(figsize=(8, 5))
    plt.plot(df["N"], df["Diff"], marker="o", label="Diff = MC - Bench")
    plt.xscale("log")
    plt.xlabel("Number of paths N")
    plt.ylabel("Difference")
    plt.title(f"Delta difference - {trade.exercise}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_delta_convergence(df: pd.DataFrame, trade: OptionTrade) -> None:
    """Plot delta levels and delta difference across path counts."""
    _plot_delta_levels(df, trade)   # MC vs Bench delta levels
    _plot_delta_diff(df, trade)     # signed difference MC - Bench


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    n_paths_list = [1_000, 2_000, 5_000, 10_000, 12_000, 15_000, 20_000, 50_000, 100_000]

    base_params = CorePricingParams(
        n_paths=100_000,      # overwritten in loop
        n_steps=100,
        seed=1,              # one seed only
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2,
    )

    # =====================================================
    # CASE 1 : EU non div -> BS
    # =====================================================
    market_eu_no_div = Market(
        S0=100,
        r=0.10,
        sigma=0.20,
    )

    trade_eu_no_div = OptionTrade(
        strike=100,
        is_call=False,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        ex_div_date=None,
        div_amount=0.0,
    )

    df_eu_no_div = delta_convergence_table(
        market=market_eu_no_div,
        trade=trade_eu_no_div,
        base_params=base_params,
        n_paths_list=n_paths_list,
        shift_spot=0.01,
        shift_vol=0.01,
        tree_N=100,
        plot=True,
    )

    # =====================================================
    # CASE 2 : EU avec div -> Tree
    # =====================================================
    market_eu_div = Market(
        S0=100,
        r=0.10,
        sigma=0.20,
    )

    trade_eu_div = OptionTrade(
        strike=100,
        is_call=False,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    df_eu_div = delta_convergence_table(
        market=market_eu_div,
        trade=trade_eu_div,
        base_params=base_params,
        n_paths_list=n_paths_list,
        shift_spot=0.01,
        shift_vol=0.01,
        tree_N=100,
        plot=True,
    )

    # =====================================================
    # CASE 3 : AM non div -> Tree
    # =====================================================
    market_am_no_div = Market(
        S0=100,
        r=0.10,
        sigma=0.20,
    )

    trade_am_no_div = OptionTrade(
        strike=100,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        ex_div_date=None,
        div_amount=0.0,
    )

    df_am_no_div = delta_convergence_table(
        market=market_am_no_div,
        trade=trade_am_no_div,
        base_params=base_params,
        n_paths_list=n_paths_list,
        shift_spot=0.01,
        shift_vol=0.01,
        tree_N=100,
        plot=True,
    )

    # =====================================================
    # CASE 4 : AM avec div -> Tree
    # =====================================================
    market_am_div = Market(
        S0=100,
        r=0.10,
        sigma=0.20,
    )

    trade_am_div = OptionTrade(
        strike=100,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    df_am_div = delta_convergence_table(
        market=market_am_div,
        trade=trade_am_div,
        base_params=base_params,
        n_paths_list=n_paths_list,
        shift_spot=0.01,
        shift_vol=0.01,
        tree_N=100,
        plot=True,
    )