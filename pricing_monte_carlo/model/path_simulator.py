import numpy as np
from model.market import Market
from model.option import OptionTrade
from model.brownian import BrownianMotion

# Time grid helper
def _step_times(T: float, n_steps: int) -> np.ndarray:
    return np.linspace(0.0, T, n_steps + 1)

# Helper to find the index of the ex-dividend time in the simulation grid
def _ex_div_index(trade: OptionTrade, dt: float, n_steps: int) -> int | None:
    """
    Map the ex-dividend time to a time-step index in the simulation grid.
    """
    ex_time = trade.ex_div_time()
    if ex_time is None or trade.div_amount == 0.0:
        return None

    # Map ex-dividend date to next step index
    j_div = int(np.ceil(float(ex_time) / dt))
    return max(0, min(n_steps, j_div))

# Helper to apply the discrete dividend adjustment to the simulated paths at the ex-dividend step
def _apply_dividend_at_step(paths: np.ndarray, j_div: int | None, div_amount: float) -> np.ndarray:

    if j_div is None:
        return paths

    j_div = int(j_div)
    D = float(div_amount)
    if D == 0.0:
        return paths

    paths[:, j_div] = np.maximum(paths[:, j_div] - D, 1e-12)
    return paths


# Main function to simulate GBM paths with discrete dividends using a scalar loop approach
def simulate_gbm_paths_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    T = float(trade.T)
    if T <= 0.0:
        raise ValueError("Maturity must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")

    dt = T / n_steps
    times = _step_times(T, n_steps)

    r = float(market.r)
    q = float(trade.q)
    sigma = float(market.sigma)
    S0 = float(market.S0)

    bm = BrownianMotion(seed)
    dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)

    drift_dt = (r - q - 0.5 * sigma * sigma) * dt

    j_div = _ex_div_index(trade, dt, n_steps)
    D = float(trade.div_amount)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    for i in range(n_paths):
        S = S0

        # If dividend is at t=0 on the grid
        if j_div == 0 and D != 0.0:
            S = max(S - D, 1e-12)

        paths[i, 0] = S

        for j in range(1, n_steps + 1):
            # Exact GBM step
            S *= np.exp(drift_dt + sigma * dW[i, j - 1])

            # Apply dividend exactly at j_div grid point
            if j_div is not None and j == j_div and D != 0.0:
                S = max(S - D, 1e-12)

            paths[i, j] = S

    return times, paths

# Main function to simulate GBM paths with discrete dividends using a vectorized approach
def simulate_gbm_paths_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    T = float(trade.T)
    if T <= 0.0:
        raise ValueError("Maturity must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")

    dt = T / n_steps
    times = _step_times(T, n_steps)

    r = float(market.r)
    q = float(trade.q)
    sigma = float(market.sigma)
    S0 = float(market.S0)

    bm = BrownianMotion(seed)
    dW = bm.dW(n_paths, n_steps, dt, antithetic=antithetic)

    # Precompute the GBM returns for all paths and steps (without dividends)
    drift_dt = (r - q - 0.5 * sigma * sigma) * dt
    R = np.exp(drift_dt + sigma * dW)  # shape (n_paths, n_steps)

    j_div = _ex_div_index(trade, dt, n_steps)
    D = float(trade.div_amount)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    # No discrete dividend
    if j_div is None:
        paths[:, 1:] = S0 * np.cumprod(R, axis=1)
        return times, paths

    # Dividend at t=0 on grid: adjust S0 then evolve normally
    if j_div == 0:
        paths[:, 0] = np.maximum(S0 - D, 1e-12)
        paths[:, 1:] = paths[:, [0]] * np.cumprod(R, axis=1)
        return times, paths

    # With dividend at j_div > 0, we need to split the evolution into 2 parts
    # Part 1: evolve from t=0 to t_j_div
    paths[:, 1:j_div + 1] = S0 * np.cumprod(R[:, :j_div], axis=1)

    # Apply dividend exactly at j_div 
    _apply_dividend_at_step(paths, j_div, D)

    # Part 2: evolve from t_j_div to maturity using the adjusted value at j_div as the new S0 for the remaining steps
    if j_div < n_steps:
        tail_cum = np.cumprod(R[:, j_div:], axis=1)  # length n_steps - j_div
        paths[:, j_div + 1:] = paths[:, [j_div]] * tail_cum

    return times, paths