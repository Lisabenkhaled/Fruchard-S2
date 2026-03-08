from __future__ import annotations
from typing import Optional, Tuple
import math
import numpy as np
from model.market import Market
from model.option import OptionTrade
from model.brownian import BrownianMotion

# Store last simulated paths for debugging or plotting
LAST_TIMES: Optional[np.ndarray] = None
LAST_PATHS: Optional[np.ndarray] = None

_MIN_PRICE = 1e-12


def _step_times(T: float, n_steps: int) -> np.ndarray:
    """Create the simulation time grid"""
    return np.linspace(0.0, T, n_steps + 1)


def _ex_div_index(trade: OptionTrade, dt: float, n_steps: int) -> Optional[int]:
    """Determine the simulation step corresponding to the ex-dividend date"""
    ex_time = trade.ex_div_time()
    if ex_time is None or float(trade.div_amount) == 0.0:
        return None
    j_div = int(math.ceil(float(ex_time) / dt))
    return max(0, min(n_steps, j_div))


def _apply_dividend_at_step(paths: np.ndarray, j_div: Optional[int], D: float) -> None:
    """Apply dividend adjustment to all paths at the dividend step"""
    if j_div is None or D == 0.0:
        return
    paths[:, int(j_div)] = np.maximum(paths[:, int(j_div)] - D, _MIN_PRICE)


def _market_trade_params(market: Market, trade: OptionTrade) -> Tuple[float, float, float, float, float]:
    """Extract numerical parameters from market and trade objects"""
    r = float(market.r)
    q = float(trade.q)
    sigma = float(market.sigma)
    S0 = float(market.S0)
    D = float(trade.div_amount)
    return r, q, sigma, S0, D



# Scalar implementation
def _paths_scalar(
    S0: float,
    dW: np.ndarray,
    n_steps: int,
    drift_dt: float,
    sigma: float,
    j_div: Optional[int],
    D: float
) -> np.ndarray:
    """
    Generate all paths using pure Python scalar loop
    """
    n_paths = int(dW.shape[0])
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    exp_ = math.exp
    sig = sigma
    mu = drift_dt
    div_step = j_div
    div_amount = D
    min_price = _MIN_PRICE

    for i in range(n_paths):                    # one path at a time
        row = paths[i]
        shocks = dW[i]

        S = S0
        for j in range(n_steps):                # one step at a time
            S = S * exp_(mu + sig * float(shocks[j]))
            if div_step is not None and (j + 1) == div_step:
                S = S - div_amount
                if S < min_price:
                    S = min_price
            row[j + 1] = S

    return paths


def _simulate_gbm_paths_scalar_from_dW(
    market: Market,
    trade: OptionTrade,
    dW: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Scalar Monte Carlo simulation from pre-generated Brownian increments"""
    T = float(trade.T)
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    times = _step_times(T, n_steps)

    r, q, sigma, S0, D = _market_trade_params(market, trade)
    drift_dt = (r - q - 0.5 * sigma * sigma) * dt
    j_div = _ex_div_index(trade, dt, n_steps)

    paths = _paths_scalar(S0, dW, n_steps, drift_dt, sigma, j_div, D)
    return times, paths


def simulate_gbm_paths_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    dW: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Scalar Monte Carlo simulation of GBM paths with optional dividend"""
    T = float(trade.T)
    dt = T / n_steps

    if dW is None:
        bm = BrownianMotion(seed)
        dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)
    else:
        if dW.shape != (n_paths, n_steps):
            raise ValueError(f"dW must have shape {(n_paths, n_steps)}, got {dW.shape}")

    return _simulate_gbm_paths_scalar_from_dW(market, trade, dW)


# Vector implementation
def _fill_paths_no_dividend(
    paths: np.ndarray,
    S0: float,
    dW: np.ndarray,
    drift_dt: float,
    sigma: float
) -> None:
    """
    Fill all paths without dividend using log-space cumulative sum
    """
    paths[:, 0] = S0
    if dW.shape[1] == 0:
        return

    log_inc = paths[:, 1:]                   
    np.multiply(sigma, dW, out=log_inc)      
    log_inc += drift_dt                      
    np.cumsum(log_inc, axis=1, out=log_inc)  
    log_inc += math.log(S0)                  
    np.exp(log_inc, out=log_inc)             


def _fill_paths_with_dividend(
    paths: np.ndarray,
    S0: float,
    dW: np.ndarray,
    drift_dt: float,
    sigma: float,
    j_div: int,
    D: float
) -> None:
    """
    Fill all paths with one discrete dividend at step j_div.
    Splits simulation into two vectorized segments around the dividend date
    """
    n_steps = dW.shape[1]
    paths[:, 0] = S0

    # Segment 1: simulate from S0 up to and including dividend step
    if j_div > 0:
        head = paths[:, 1:j_div + 1]
        np.multiply(sigma, dW[:, :j_div], out=head)
        head += drift_dt
        np.cumsum(head, axis=1, out=head)
        head += math.log(S0)
        np.exp(head, out=head)

    # Apply the discrete dividend at ex-div step
    _apply_dividend_at_step(paths, j_div, D)

    # Segment 2: continue from post-dividend price
    if j_div < n_steps:
        tail = paths[:, j_div + 1:]
        np.multiply(sigma, dW[:, j_div:], out=tail)
        tail += drift_dt
        np.cumsum(tail, axis=1, out=tail)
        np.exp(tail, out=tail)
        tail *= paths[:, [j_div]]            # scale by post-dividend starting price


def _paths_vector(
    S0: float,
    dW: np.ndarray,
    drift_dt: float,
    sigma: float,
    j_div: Optional[int],
    D: float
) -> np.ndarray:
    """Fully vectorized path generation"""
    n_paths, n_steps = int(dW.shape[0]), int(dW.shape[1])
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    if j_div is None or D == 0.0:
        _fill_paths_no_dividend(paths, S0, dW, drift_dt, sigma)
        return paths

    if j_div == 0:
        # Dividend before first step: adjust S0 once, then simulate normally
        S_init = max(S0 - D, _MIN_PRICE)
        _fill_paths_no_dividend(paths, S_init, dW, drift_dt, sigma)
        return paths

    _fill_paths_with_dividend(paths, S0, dW, drift_dt, sigma, int(j_div), D)
    return paths


def _simulate_gbm_paths_vector_from_dW(
    market: Market,
    trade: OptionTrade,
    dW: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Monte Carlo GBM simulation from pre-generated Brownian increments"""
    global LAST_TIMES, LAST_PATHS

    T = float(trade.T)
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    times = _step_times(T, n_steps)

    r, q, sigma, S0, D = _market_trade_params(market, trade)
    drift_dt = (r - q - 0.5 * sigma * sigma) * dt
    j_div = _ex_div_index(trade, dt, n_steps)

    paths = _paths_vector(S0, dW, drift_dt, sigma, j_div, D)

    LAST_TIMES, LAST_PATHS = times, paths
    return times, paths


def simulate_gbm_paths_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = False,
    dW: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Monte Carlo GBM simulation — all paths computed simultaneously"""
    T = float(trade.T)
    dt = T / n_steps

    if dW is None:
        bm = BrownianMotion(seed)
        dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)
    else:
        if dW.shape != (n_paths, n_steps):
            raise ValueError(f"dW must have shape {(n_paths, n_steps)}, got {dW.shape}")

    return _simulate_gbm_paths_vector_from_dW(market, trade, dW)