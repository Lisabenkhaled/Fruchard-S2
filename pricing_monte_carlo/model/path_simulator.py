from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from model.market import Market
from model.option import OptionTrade
from model.brownian import BrownianMotion

# Store last simulated paths for debugging or plotting
LAST_TIMES: Optional[np.ndarray] = None
LAST_PATHS: Optional[np.ndarray] = None

def _step_times(T: float, n_steps: int) -> np.ndarray:
    """Create the simulation time grid"""
    return np.linspace(0.0, T, n_steps + 1)

def _ex_div_index(trade: OptionTrade, dt: float, n_steps: int) -> Optional[int]:
    """
    Determine the simulation step corresponding to the ex-dividend date
    Returns None if no dividend exists
    """
    ex_time = trade.ex_div_time()
    # No dividend case
    if ex_time is None or float(trade.div_amount) == 0.0:
        return None
    # Map dividend time to closest future grid step
    j_div = int(np.ceil(float(ex_time) / dt))
    return max(0, min(n_steps, j_div))

def _apply_dividend_scalar(S: float, D: float) -> float:
    """Apply a discrete dividend to a single price"""
    return max(S - D, 1e-12) if D != 0.0 else S

def _apply_dividend_at_step(paths: np.ndarray, j_div: Optional[int], D: float) -> None:
    """Apply dividend adjustment to all paths at the dividend step"""
    if j_div is None or D == 0.0:
        return
    j = int(j_div)
    # Subtract dividend and prevent negative prices
    paths[:, j] = np.maximum(paths[:, j] - D, 1e-12)

def _gbm_returns(r: float, q: float, sigma: float, dt: float, dW: np.ndarray) -> np.ndarray:
    """
    Compute multiplicative returns for GBM
    """
    drift_dt = (r - q - 0.5 * sigma * sigma) * dt
    return np.exp(drift_dt + sigma * dW)

def _market_trade_params(market: Market, trade: OptionTrade) -> Tuple[float, float, float, float, float]:
    """Extract numerical parameters from market and trade objects"""
    r = float(market.r)
    q = float(trade.q)
    sigma = float(market.sigma)
    S0 = float(market.S0)
    D = float(trade.div_amount)
    return r, q, sigma, S0, D


def _simulate_one_path(S0: float, R_row: np.ndarray,n_steps: int,
                       j_div: Optional[int], D: float) -> np.ndarray:
    """Simulate a single GBM path using precomputed returns"""
    out = np.empty(n_steps + 1, dtype=float)

    # Adjust initial price if dividend occurs at t=0
    S = _apply_dividend_scalar(S0, D) if j_div == 0 else S0
    out[0] = S

    # Iterate through time steps
    for j in range(1, n_steps + 1):
        S *= float(R_row[j - 1])

        # Apply dividend exactly at the ex-div step
        if j_div is not None and j == j_div:
            S = _apply_dividend_scalar(S, D)
        out[j] = S
    return out

def _paths_scalar(S0: float, R: np.ndarray, n_steps: int, j_div: Optional[int], D: float) -> np.ndarray:
    """Generate all paths using a scalar loop (one path at a time)"""
    n_paths = int(R.shape[0])
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    for i in range(n_paths):
        paths[i, :] = _simulate_one_path(S0, R[i, :], n_steps, j_div, D)
    return paths

def simulate_gbm_paths_scalar(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
                              seed: int = 0, antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Scalar Monte Carlo simulation of GBM paths with optional dividend"""
    T = float(trade.T)
    dt = T / n_steps
    times = _step_times(T, n_steps)

    # Extract model parameters
    r, q, sigma, S0, D = _market_trade_params(market, trade)

    # Generate Brownian increments
    bm = BrownianMotion(seed)
    dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)

    # Compute multiplicative GBM returns
    R = _gbm_returns(r, q, sigma, dt, dW)

    # Determine dividend step
    j_div = _ex_div_index(trade, dt, n_steps)
    paths = _paths_scalar(S0, R, n_steps, j_div, D)
    return times, paths

def _fill_paths_no_dividend(paths: np.ndarray, S0: float, R: np.ndarray) -> None:
    """Fill paths when no dividend is present"""
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(R, axis=1)

def _fill_paths_with_dividend(paths: np.ndarray, S0: float, R: np.ndarray, j_div: int, D: float) -> None:
    """Fill paths when a dividend occurs during the simulation"""
    paths[:, 0] = S0

    # Simulate up to dividend step
    paths[:, 1:j_div + 1] = S0 * np.cumprod(R[:, :j_div], axis=1)

    # Apply dividend adjustment
    _apply_dividend_at_step(paths, j_div, D)

    # Continue evolution after dividend
    if j_div < R.shape[1]:
        tail = np.cumprod(R[:, j_div:], axis=1)
        paths[:, j_div + 1:] = paths[:, [j_div]] * tail

def _paths_vector(S0: float, R: np.ndarray, j_div: Optional[int], D: float) -> np.ndarray:
    """Vectorized path generation"""
    n_paths, n_steps = int(R.shape[0]), int(R.shape[1])
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    # Case 1: no dividend
    if j_div is None or D == 0.0:
        _fill_paths_no_dividend(paths, S0, R)
        return paths
    # Case 2: dividend at t=0
    if j_div == 0:
        _fill_paths_no_dividend(paths, max(S0 - D, 1e-12), R)
        return paths
    # Case 3: dividend during simulation
    _fill_paths_with_dividend(paths, S0, R, int(j_div), D)
    return paths

def simulate_gbm_paths_vector(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
                              seed: int = 0, antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Monte Carlo GBM simulation"""
    global LAST_TIMES, LAST_PATHS

    T = float(trade.T)
    dt = T / n_steps
    times = _step_times(T, n_steps)

    # Extract model parameters
    r, q, sigma, S0, D = _market_trade_params(market, trade)

    # Generate Brownian motion increments
    bm = BrownianMotion(seed)
    dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)

    # Compute GBM returns
    R = _gbm_returns(r, q, sigma, dt, dW)

    # Determine dividend step
    j_div = _ex_div_index(trade, dt, n_steps)

    # Generate paths
    paths = _paths_vector(S0, R, j_div, D)

    LAST_TIMES, LAST_PATHS = times, paths
    return times, paths