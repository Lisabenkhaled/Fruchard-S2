from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from model.market import Market
from model.option import OptionTrade
from model.brownian import BrownianMotion
from model.path_simulator import simulate_gbm_paths_scalar, simulate_gbm_paths_vector, _simulate_gbm_paths_vector_from_dW
from model.regression import design_matrix, ols_fit_predict


def _ls_regression_values(
    stock_values: NDArray[np.float64],
    cont_values: NDArray[np.float64],
    strike: float,
    basis: str,
    degree: int
) -> NDArray[np.float64]:
    """
    Build the regression design matrix from ITM stock prices (normalised by
    the strike for numerical stability), then fit OLS and return the fitted
    continuation values at the same points.
    """
    X = design_matrix(stock_values, degree=degree, basis=basis, scale=strike)
    return ols_fit_predict(X, cont_values, X)



# European MC — Scalar
def _european_scalar_price_paths(
    paths: NDArray[np.float64],
    n_paths: int,
    n_steps: int,
    disc: float,
    payoff_scalar: object
) -> NDArray[np.float64]:
    """
    Walk every simulated path step by step and
    return the array of discounted payoffs
    """
    discounted_payoffs = np.empty(n_paths, dtype=np.float64)

    for i in range(n_paths):           # iterate over paths one at a time
        row = paths[i]                 # extract the i-th path as a 1-D view
        ST = 0.0
        for j in range(n_steps + 1):  # walk every time step explicitly
            ST = float(row[j])         # overwrite ST each step; last value = S_T
        # apply payoff and discount back to t=0
        discounted_payoffs[i] = disc * payoff_scalar(ST)

    return discounted_payoffs

def price_european_naive_mc_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False
) -> tuple[float, NDArray[np.float64]]:
    """
    Monte Carlo price of a European option using a fully scalar pricer.
    """
    r: float = float(market.r)
    T: float = float(trade.T)
    N: int = int(n_paths)
    M: int = int(n_steps)

    disc: float = math.exp(-r * T)    # risk-neutral discount factor

    # Simulate GBM paths scalar 
    _, paths = simulate_gbm_paths_scalar(market, trade, N, M, seed, antithetic)

    # Price each path step by step via the scalar helper
    discounted_payoffs = _european_scalar_price_paths(paths, N, M, disc, trade.payoff_scalar)

    # Accumulate the mean 
    price: float = 0.0
    for i in range(N):
        price += discounted_payoffs[i]
    price /= N

    return float(price), discounted_payoffs

# European MC — Vector
def _european_vector_terminal_prices(
    market: Market,
    trade: OptionTrade,
    dW: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate full GBM paths from dW and return
    Both the no-dividend and dividend cases go through
    _simulate_gbm_paths_vector_from_dW so the full (N, M+1) path matrix
    is always available.
    """
    _, paths = _simulate_gbm_paths_vector_from_dW(market, trade, dW)
    return paths, paths[:, -1]

def price_european_naive_mc_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False
) -> tuple[float, NDArray[np.float64]]:
    """
    Monte Carlo price of a European option using vectorised pricer.

    Always simulates full GBM paths via _simulate_gbm_paths_vector_from_dW
    (same route as the dividend case) so LAST_PATHS is set and the excel
    diagnostic can reuse the exact same paths.
    """
    import model.path_simulator as path_sim

    r: float = float(market.r)
    T: float = float(trade.T)
    N: int = int(n_paths)
    M: int = int(n_steps)

    disc: float = math.exp(-r * T)

    # Draw Brownian increments — same dW as scalar pricer for same seed
    bm = BrownianMotion(seed)
    dW = bm.dW(N, M, T / M, antithetic=antithetic)    # shape (N, M)

    # Simulate full paths and extract terminal prices
    paths, ST = _european_vector_terminal_prices(market, trade, dW)

    # Expose paths for diagnostics (same interface as simulate_gbm_paths_vector)
    times = np.linspace(0.0, T, M + 1)
    path_sim.LAST_PATHS = paths
    path_sim.LAST_TIMES = times

    # Apply payoff and discount in one vectorised multiply
    discounted_payoffs = disc * trade.payoff_vector(ST)

    return float(np.mean(discounted_payoffs)), discounted_payoffs


# American Naive MC — Scalar
def _american_naive_scalar_best_pv(
    paths: NDArray[np.float64],
    n_paths: int,
    n_steps: int,
    step_df: float,
    payoff_scalar: object
) -> NDArray[np.float64]:
    best_pv = np.empty(n_paths, dtype=np.float64)

    for i in range(n_paths):               # path by path
        row = paths[i]
        best: float = payoff_scalar(float(row[0]))   # payoff at t=0
        disc_j: float = step_df
        for j in range(1, n_steps + 1):    # step by step
            pv: float = payoff_scalar(float(row[j])) * disc_j
            if pv > best:                  # keep the best discounted payoff
                best = pv
            disc_j *= step_df              # accumulate discount
        best_pv[i] = best
    return best_pv


def price_american_naive_mc_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False
) -> tuple[float, NDArray[np.float64]]:
    """
    Naive American MC price 
    """
    r: float = float(market.r)
    T: float = float(trade.T)
    N: int = int(n_paths)
    M: int = int(n_steps)

    # Simulate paths scalar
    _, paths = simulate_gbm_paths_scalar(market, trade, N, M, seed, antithetic)

    # One-step discount factor
    step_df: float = math.exp(-r * (T / M))
    best_pv = _american_naive_scalar_best_pv(paths, N, M, step_df, trade.payoff_scalar)
    return float(best_pv.sum() / N), best_pv


# American Naive MC — Vector
def price_american_naive_mc_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False
) -> tuple[float, NDArray[np.float64]]:
    """
    Naive American MC price (vectorised)
    """
    r: float = float(market.r)

    # Simulate all paths at once
    times, paths = simulate_gbm_paths_vector(
        market, trade, int(n_paths), int(n_steps), seed, antithetic
    )

    intrinsic = trade.payoff_vector(paths)   # payoff at every node, shape (N, M+1)
    disc = np.exp(-r * times)                # discount factors, shape (M+1,)
    intrinsic *= disc                        # discounted intrinsic at every node
    best_pv = intrinsic.max(axis=1)          # best discounted payoff per path, shape (N,)

    return float(best_pv.mean()), best_pv



# American Longstaff-Schwartz — Scalar
def _ls_scalar_backward(
    S: NDArray[np.float64],
    n_paths: int,
    n_steps: int,
    df: float,
    payoff_scalar: object,
    strike: float,
    basis: str,
    degree: int
) -> list[float]:
    """
    Scalar Longstaff-Schwartz backward induction.
    At each step: discount V, collect ITM paths in one pass,
    regress continuation values, apply exercise decision.
    """
    # Initialise V with terminal payoffs at step M
    V: list[float] = [payoff_scalar(float(S[i][n_steps])) for i in range(n_paths)]

    for j in range(n_steps - 1, 0, -1):

        # Discount all continuation values back one step
        for i in range(n_paths):
            V[i] *= df

        # Collect ITM paths in one pass — (idx, S_j, V_j, exercise) per ITM path
        itm: list[tuple[int, float, float, float]] = [
            (i, float(S[i][j]), V[i], payoff_scalar(float(S[i][j])))
            for i in range(n_paths) if payoff_scalar(float(S[i][j])) > 0.0
        ]
        if not itm:                 # no ITM paths — skip regression
            continue

        idx, Sj, Y, ex = zip(*itm)  # unpack four columns in one shot

        # Regress continuation values on ITM stock prices
        cont = _ls_regression_values(
            np.asarray(Sj, dtype=np.float64),
            np.asarray(Y, dtype=np.float64),
            strike, basis, degree,
        )

        # Exercise decision — override V where exercise >= fitted continuation
        for k, i in enumerate(idx):
            if ex[k] >= cont[k]:
                V[i] = ex[k]

    return V

def price_american_ls_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    basis: str = "laguerre",
    degree: int = 2
) -> tuple[float, NDArray[np.float64]]:
    """
    American option price via Longstaff-Schwartz scalar
    """
    r: float = float(market.r)
    T: float = float(trade.T)
    N: int = int(n_paths)
    M: int = int(n_steps)

    # Simulate scalar paths 
    _, S = simulate_gbm_paths_scalar(market, trade, N, M, seed, antithetic)

    # One-step discount factor used throughout backward induction
    df: float = math.exp(-r * (T / M))

    # Run scalar backward induction
    V = _ls_scalar_backward(S, N, M, df, trade.payoff_scalar, float(trade.strike), basis, int(degree))

    # Discount final cash flows and accumulate the mean
    discounted_cf = np.empty(N, dtype=np.float64)
    total: float = 0.0
    for i in range(N):
        Vi: float = V[i] * df      # one final discount step
        discounted_cf[i] = Vi
        total += Vi

    return total / N, discounted_cf


# American Longstaff-Schwartz — Vector
def _ls_vector_backward(
    S: NDArray[np.float64],
    n_steps: int,
    df: float,
    trade: OptionTrade,
    strike: float,
    basis: str,
    degree: int
) -> NDArray[np.float64]:
    """
    Vectorised Longstaff-Schwartz backward induction
    """
    # Initialise with terminal payoffs at step M — shape (N,)
    V: NDArray[np.float64] = np.array(trade.payoff_vector(S[:, -1]))

    for j in range(n_steps - 1, 0, -1):

        V *= df                             # discount all continuation values back one step

        Sj = S[:, j]                        # stock prices at step j, all paths — shape (N,)
        exercise = trade.payoff_vector(Sj)  # immediate exercise values — shape (N,)
        itm = exercise > 0.0                

        if not itm.any():                   # no ITM paths at this step — skip regression
            continue

        itm_idx = np.nonzero(itm)[0]       # integer indices of ITM paths

        # Regress discounted continuation on ITM stock prices
        cont = _ls_regression_values(Sj[itm], V[itm], strike, basis, degree)

        # Exercise where immediate payoff >= fitted continuation value
        mask_ex = exercise[itm] >= cont
        if mask_ex.any():
            V[itm_idx[mask_ex]] = exercise[itm_idx[mask_ex]]

    return V

def price_american_ls_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    basis: str = "laguerre",
    degree: int = 2
) -> tuple[float, NDArray[np.float64]]:
    """
    American option price via Longstaff-Schwartz (vectorised)
    """
    r: float = float(market.r)
    T: float = float(trade.T)
    M: int = int(n_steps)

    # Simulate all paths at once — shape (N, M+1)
    times, S = simulate_gbm_paths_vector(market, trade, int(n_paths), M, seed, antithetic)

    # Expose paths for diagnostics (same interface as price_european_naive_mc_vector)
    import model.path_simulator as path_sim
    path_sim.LAST_PATHS = S
    path_sim.LAST_TIMES = np.asarray(times, dtype=float)

    # One-step discount factor used throughout backward induction
    df: float = math.exp(-r * (T / M))

    # Run vectorised backward induction — returns continuation values at step 1
    V = _ls_vector_backward(S, M, df, trade, float(trade.strike), basis, int(degree))
    V *= df

    return float(V.mean()), V