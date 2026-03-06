from __future__ import annotations

import datetime as dt
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector
from model.regression import design_matrix, ols_fit_predict


def _discount_factor(rate: float, maturity: float, n_steps: int) -> float:
    """Return the one-step discount factor"""
    dt_step = maturity / n_steps
    return float(np.exp(-rate * dt_step))


def _basis_matrix(spots: np.ndarray, trade: OptionTrade, degree: int, basis: str) -> np.ndarray:
    """Build the regression design matrix for the chosen basis"""
    if basis == "laguerre":
        return design_matrix(
            spots,
            degree=degree,
            basis="laguerre",
            scale=trade.strike,
        )
    return design_matrix(
        spots,
        degree=degree,
        basis="power",
    )

def _update_cashflows(trade: OptionTrade, spots_t: np.ndarray, values: np.ndarray, 
                      df: float, degree: int, basis: str) -> np.ndarray:
    """Update continuation/exercise values at one exercise date"""
    exercise = trade.payoff_vector(spots_t)
    itm = exercise > 0.0

    # OTMy paths cannot be exercised now
    values[~itm] = df * values[~itm]
    if not np.any(itm):
        return values

    # Regress discounted continuation values on ITM states only
    y = df * values[itm]
    x = _basis_matrix(spots_t[itm], trade, degree, basis)
    continuation = ols_fit_predict(x, y, x)

    # Choose between immediate exercise and continuation
    ex_now = exercise[itm] > continuation
    hold_values = df * values[itm]
    values[itm] = np.where(ex_now, exercise[itm], hold_values)
    return values


def lsm_price_from_paths(market: Market,trade: OptionTrade, paths: np.ndarray, n_steps: int,
                         degree: int, basis: str) -> float:
    """Price an American option with the Longstaff-Schwartz method"""
    df = _discount_factor(market.r, trade.T, n_steps)

    # Start from terminal payoff at maturity
    values = trade.payoff_vector(paths[:, -1])

    # Move backward through the exercise dates
    for t in range(n_steps - 1, 0, -1):
        spots_t = paths[:, t]
        values = _update_cashflows(trade, spots_t, values, df, degree, basis)

    # Discount one last step back to time 0
    return float(df * values.mean())


def _prices_for_degree_list(market: Market, trade: OptionTrade, paths: np.ndarray, n_steps: int,
                            degrees: Sequence[int]) -> Dict[str, np.ndarray]:
    """Compute LSM prices for power and Laguerre bases"""
    power_prices = []
    laguerre_prices = []

    # Use the same simulated paths for all degrees and both bases
    for degree in degrees:
        power_prices.append(
            lsm_price_from_paths(market, trade, paths, n_steps, degree, "power")
        )
        laguerre_prices.append(
            lsm_price_from_paths(market, trade, paths, n_steps, degree, "laguerre")
        )

    return {
        "power": np.array(power_prices, dtype=float),
        "laguerre": np.array(laguerre_prices, dtype=float),
    }


def _print_comparison_table(degrees: Sequence[int], results: Dict[str, np.ndarray], seed: int, 
                            n_paths: int,n_steps: int, antithetic: bool) -> None:
    """Print a comparison table for the two regression bases"""

    print("\n============================================")
    print("LSM Regression Degree Comparison (1 seed)")
    print("============================================")
    print(
        f"seed={seed}, n_paths={n_paths}, "
        f"n_steps={n_steps}, antithetic={antithetic}\n"
    )
    print("Degree | Power Price | Laguerre Price | Diff (Lag - Pow)")
    print("---------------------------------------------------------")

    # Print one line per polynomial degree.
    for i, degree in enumerate(degrees):
        diff = results["laguerre"][i] - results["power"][i]
        print(
            f"{degree:>6d} | {results['power'][i]:>10.6f} | "
            f"{results['laguerre'][i]:>13.6f} | {diff:>13.6f}"
        )


def _plot_comparison(degrees: Sequence[int], results: Dict[str, np.ndarray], seed: int) -> None:
    """Plot option prices as a function of regression degree"""

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot both basis families on the same figure
    ax.plot(degrees, results["power"], marker="o", linewidth=2, label="Power basis")
    ax.plot(
        degrees,
        results["laguerre"],
        marker="s",
        linewidth=2,
        label="Laguerre basis",
    )

    ax.set_title(f"LSM Price vs Regression Degree (Single Seed = {seed})")
    ax.set_xlabel("Regression Degree")
    ax.set_ylabel("Option Price")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def compare_bases_one_seed(market: Market, trade: OptionTrade, degrees: Sequence[int],
                           seed: int, n_paths: int, n_steps: int,
                           antithetic: bool = True, plot: bool = True) -> Dict[str, np.ndarray]:
    """Compare Power and Laguerre bases for one Monte Carlo seed"""

    if trade.exercise.lower() != "american":
        raise ValueError("This test is for American options (LSM).")

    # Simulate once so both bases use identical paths.
    _, paths = simulate_gbm_paths_vector(
        market,
        trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )

    # Compute the prices for each regression degree.
    results = _prices_for_degree_list(market, trade, paths, n_steps, degrees)

    # Display the numerical comparison.
    _print_comparison_table(degrees, results, seed, n_paths, n_steps, antithetic)

    # Optionally show the graph.
    if plot:
        _plot_comparison(degrees, results, seed)

    return results


def _build_example_market() -> Market:
    """Create the market object used in the script example"""
    return Market(S0=100.0, r=0.10, sigma=0.20)


def _build_example_trade() -> OptionTrade:
    """Create the American option used in the script example"""
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 26)

    return OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 10, 30),
        div_amount=3.0,
    )


def main() -> None:
    """Run a simple comparison between basis functions"""
    market = _build_example_market()
    trade = _build_example_trade()

    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    seed = 42

    compare_bases_one_seed(
        market=market,
        trade=trade,
        degrees=degrees,
        seed=seed,
        n_paths=100000,
        n_steps=100,
        antithetic=True,
        plot=True,
    )


if __name__ == "__main__":
    main()