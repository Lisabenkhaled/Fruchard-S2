from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector, simulate_gbm_paths_scalar
from model.regression import design_matrix, ols_fit_predict

# Discount helpers
def _discount_factor(rate: float, T: float) -> float:
    """Discount factor exp(-rT)."""
    return float(np.exp(-rate * T))

def _one_step_discount(rate: float, T: float, n_steps: int) -> float:
    """One-step discount factor."""
    dt = float(T) / int(n_steps)
    return float(np.exp(-rate * dt))


# Intrinsic values
def _intrinsic_vector(trade: OptionTrade, paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute intrinsic values on full path matrix."""
    try:
        return trade.payoff_vector(paths)
    except Exception:
        intrinsic = np.empty_like(paths)
        for j in range(paths.shape[1]):
            intrinsic[:, j] = trade.payoff_vector(paths[:, j])
        return intrinsic


# Regression helper
def _ls_regression_values(stock_values: NDArray[np.float64], cont_values: NDArray[np.float64],
    strike: float, basis: str, degree: int) -> NDArray[np.float64]:

    X = design_matrix(stock_values, degree=degree, basis=basis, scale=strike)
    return ols_fit_predict(X, cont_values, X)

# Helpers for scalar LS
def _terminal_cashflows_scalar(trade: OptionTrade, paths: NDArray[np.float64], n_paths: int) -> NDArray[np.float64]:
    """Terminal payoff per path"""
    V = np.empty(int(n_paths), dtype=float)

    for i in range(int(n_paths)):
        V[i] = trade.payoff_scalar(paths[i, -1])

    return V

def _exercise_values_scalar(trade: OptionTrade, paths: NDArray[np.float64],
    n_paths: int, step: int) -> NDArray[np.float64]:
    """Intrinsic value at time step"""

    ex = np.empty(int(n_paths), dtype=float)

    for i in range(int(n_paths)):
        ex[i] = trade.payoff_scalar(paths[i, step])

    return ex


# European MC : Scalar
def price_european_naive_mc_scalar(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seed: int = 0, antithetic: bool = False) -> tuple[float, NDArray[np.float64]]:
    _, paths = simulate_gbm_paths_scalar(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    disc = _discount_factor(float(market.r), float(trade.T))

    discounted_payoffs = np.empty(int(n_paths), dtype=float)

    # Terminal payoff per path
    for i in range(int(n_paths)):
        discounted_payoffs[i] = disc * trade.payoff_scalar(paths[i, -1])

    price = float(discounted_payoffs.mean())
    return price, discounted_payoffs

# European MC : Vector
def price_european_naive_mc_vector(market: Market, trade: OptionTrade, n_paths: int,
    n_steps: int, seed: int = 0, antithetic: bool = False) -> tuple[float, NDArray[np.float64]]:
    _, paths = simulate_gbm_paths_vector(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    ST = paths[:, -1]
    payoff = trade.payoff_vector(ST)

    disc = _discount_factor(float(market.r), float(trade.T))
    discounted_payoffs = disc * payoff

    price = float(discounted_payoffs.mean())
    return price, discounted_payoffs


# Naive American MC : Scalar
def price_american_naive_mc_scalar(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seed: int = 0, antithetic: bool = False) -> tuple[float, NDArray[np.float64]]:
    times, paths = simulate_gbm_paths_scalar(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    disc = np.exp(-float(market.r) * times)
    best_pv = np.empty(int(n_paths), dtype=float)

    # Search best exercise time
    for i in range(int(n_paths)):
        best = -np.inf
        for j in range(paths.shape[1]):
            ex = trade.payoff_scalar(paths[i, j])
            pv = ex * disc[j]
            if pv > best:
                best = pv
        best_pv[i] = best

    price = float(best_pv.mean())
    return price, best_pv

# Naive American MC : Vector
def price_american_naive_mc_vector(market: Market, trade: OptionTrade, n_paths: int,
    n_steps: int, seed: int = 0, antithetic: bool = False) -> tuple[float, NDArray[np.float64]]:
    times, paths = simulate_gbm_paths_vector(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    disc = np.exp(-float(market.r) * times)
    intrinsic = _intrinsic_vector(trade, paths)
    pv = intrinsic * disc[None, :]
    best_pv = np.max(pv, axis=1)
    price = float(best_pv.mean())
    return price, best_pv


# Longstaff-Schwartz scalar
def price_american_ls_scalar(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seed: int = 0, antithetic: bool = False, basis: str = "laguerre", degree: int = 2
) -> tuple[float, NDArray[np.float64]]:
    _, S = simulate_gbm_paths_scalar(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    df = _one_step_discount(float(market.r), float(trade.T), int(n_steps))

    # Terminal payoff
    V = _terminal_cashflows_scalar(trade, S, int(n_paths))

    # Backward induction
    for j in range(int(n_steps) - 1, 0, -1):
        V = df * V
        exercise = _exercise_values_scalar(trade, S, int(n_paths), int(j))
        itm_idx = np.where(exercise > 0.0)[0]
        if itm_idx.size == 0:
            continue

        Sj_itm = S[itm_idx, j]
        Y = V[itm_idx]
        cont = _ls_regression_values(
            Sj_itm,
            Y,
            float(trade.strike),
            basis,
            int(degree),
        )

        for k, idx in enumerate(itm_idx):
            if exercise[idx] >= cont[k]:
                V[idx] = exercise[idx]

    discounted_cf = df * V
    price = float(discounted_cf.mean())

    return price, discounted_cf


# Longstaff-Schwartz vector
def price_american_ls_vector(market: Market, trade: OptionTrade, n_paths: int,n_steps: int,
                             seed: int = 0, antithetic: bool = False, 
                             basis: str = "laguerre", degree: int = 2) -> tuple[float, NDArray[np.float64]]:
    _, S = simulate_gbm_paths_vector(
        market,
        trade,
        n_paths,
        n_steps,
        seed,
        antithetic,
    )

    df = _one_step_discount(float(market.r), float(trade.T), int(n_steps))
    V = trade.payoff_vector(S[:, -1])

    # Backward induction
    for j in range(int(n_steps) - 1, 0, -1):
        Sj = S[:, j]
        exercise = trade.payoff_vector(Sj)
        V = df * V
        itm = exercise > 0.0
        if not np.any(itm):
            continue

        Y = V[itm]
        cont = _ls_regression_values(
            Sj[itm],
            Y,
            float(trade.strike),
            basis,
            int(degree),
        )

        ex_now = exercise[itm] >= cont
        V_itm = V[itm]
        V_itm = np.where(ex_now, exercise[itm], V_itm)
        V[itm] = V_itm

    discounted_cf = df * V
    price = float(discounted_cf.mean())

    return price, discounted_cf