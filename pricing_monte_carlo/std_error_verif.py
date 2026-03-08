from __future__ import annotations

import datetime as dt
import numpy as np

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector, price_american_ls_vector


def _run_seeds(
    price_function: object,
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    n_seeds: int,
    **kwargs: object
) -> tuple[list[float], list[float]]:
    """
    Run the pricer for each seed, collect per-seed prices and standard errors.
    SE per seed = std(discounted_payoffs, ddof=1) / sqrt(n_paths).
    """
    prices: list[float] = []
    ses: list[float] = []

    for seed in range(n_seeds):
        # Fresh random paths for each seed
        price, discounted_payoffs = price_function(
            market=market, trade=trade,
            n_paths=n_paths, n_steps=n_steps,
            seed=seed, **kwargs,
        )
        prices.append(price)
        # SE = std(payoffs) / sqrt(N) — Bessel correction for unbiased std
        se: float = float(np.std(discounted_payoffs, ddof=1)) / np.sqrt(n_paths)
        ses.append(se)
        print(f"Seed {seed:3d}  |  Price = {price:.6f}  |  SE = {se:.6f}")

    return prices, ses


def _print_summary(prices: list[float], ses: list[float]) -> tuple[float, float]:
    """
    Compute and print avg SE vs std of prices
    Returns (avg_se, std_price)
    """
    avg_se: float = float(np.mean(ses))
    std_price: float = float(np.std(prices, ddof=1))
    rel_diff: float = abs(avg_se - std_price) / std_price

    print("\nSummary")
    print("----------------------")
    print(f"Average SE            : {avg_se:.6f}")
    print(f"Std of price estimator: {std_price:.6f}")
    print(f"Relative difference   : {rel_diff:.2%}")

    return avg_se, std_price


def test_mc_standard_error(
    price_function: object,
    market: Market,
    trade: OptionTrade,
    n_paths: int = 10000,
    n_steps: int = 50,
    n_seeds: int = 20,
    label: str = "",
    **kwargs: object
) -> tuple[float, float]:
    """
    Validate the MC standard error estimator across multiple seeds
    """
    header: str = f"SE for each seed — {label}" if label else "SE for each seed"
    print(f"\n{header}")
    print("----------------------")

    # Collect per-seed prices and SEs
    prices, ses = _run_seeds(
        price_function, market, trade, n_paths, n_steps, n_seeds, **kwargs
    )

    # Summarise and return
    return _print_summary(prices, ses)


def main() -> None:
    """
    Run SE validation for:
      1. European call with discrete dividend  — vectorised pricer
      2. American put without dividend         — Longstaff-Schwartz vectorised
    """
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    # Shared market parameters
    market = Market(
        S0=100.0,
        r=0.10,
        sigma=0.20,
    )

    # =========================================================
    # Case 1 — European call with discrete dividend
    # =========================================================
    trade_eu = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )

    print("\n" + "=" * 60)
    print("CASE 1 — European call | div = 3.0")
    print("=" * 60)
    test_mc_standard_error(
        price_function=price_european_naive_mc_vector,
        market=market,
        trade=trade_eu,
        n_paths=100_000,
        n_steps=100,
        n_seeds=50,
        label="EU call with dividend",
    )

    # =========================================================
    # Case 2 — American put without dividend (Longstaff-Schwartz)
    # =========================================================
    trade_am = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0,
    )

    print("\n" + "=" * 60)
    print("CASE 2 — American put | no dividend | LS vector")
    print("=" * 60)
    test_mc_standard_error(
        price_function=price_american_ls_vector,
        market=market,
        trade=trade_am,
        n_paths=100_000,
        n_steps=100,
        n_seeds=50,
        label="AM put LS vector",
        basis="laguerre",   # forwarded to price_american_ls_vector via **kwargs
        degree=2,
    )


if __name__ == "__main__":
    main()