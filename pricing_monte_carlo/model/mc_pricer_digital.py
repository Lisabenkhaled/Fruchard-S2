from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import (
    simulate_gbm_paths_vector,
    simulate_gbm_paths_scalar)

def _digital_inputs(trade: OptionTrade, n_steps: int, digital_strike: float, payout: float) -> tuple[float, float, float]:
    """Prepare common scalar inputs"""
    T = float(trade.T)
    dt = T / int(n_steps)
    K = float(digital_strike)
    pay = float(payout)
    return T, dt, K, pay

def _discount_factor(r: float, hit_index: int, dt: float) -> float:
    """Discount factor from hit time back to today"""
    return float(np.exp(-float(r) * (float(hit_index) * dt)))

def price_american_digital_vector(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    digital_strike: float, payout: float = 1.0, seed: int = 0, antithetic: bool = False) -> Tuple[float, NDArray[np.float64]]:
    """
    Price an American digital option with vectorized path processing
    """
    _, dt, K, pay = _digital_inputs(trade, n_steps, digital_strike, payout)

    # Simulate all GBM paths at once
    _, paths = simulate_gbm_paths_vector(market, trade, n_paths, n_steps,
        seed=seed, antithetic=antithetic)

    itm: NDArray[np.bool_] = paths < K
    any_itm: NDArray[np.bool_] = np.any(itm, axis=1)
    first_idx: NDArray[np.int_] = np.argmax(itm, axis=1).astype(int)
    disc: NDArray[np.float64] = np.exp(-float(market.r) * (first_idx * dt))

    # If never hit, payoff is 0
    discounted_payoffs: NDArray[np.float64] = np.where(any_itm, pay * disc, 0.0)

    price = float(np.mean(discounted_payoffs))
    return price, discounted_payoffs


def _first_hit_index(path: NDArray[np.float64], n_steps: int, strike: float) -> int | None:
    """Return the first index j such that path[j] < strike."""
    for j in range(int(n_steps) + 1):
        if path[j] < strike:
            return j
    return None

def price_american_digital_scalar(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    digital_strike: float, payout: float = 1.0, seed: int = 0, antithetic: bool = False) -> Tuple[float, NDArray[np.float64]]:
    """
    Price an American digital option with scalar-style path inspection
    """
    _, dt, K, pay = _digital_inputs(trade, n_steps, digital_strike, payout)

    # Simulate paths
    _, paths = simulate_gbm_paths_scalar(market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic)

    # Store one discounted payoff per path
    discounted_payoffs: NDArray[np.float64] = np.zeros(int(n_paths), dtype=np.float64)

    r = float(market.r)

    # Loop path by path
    for i in range(int(n_paths)):
        # Find first time where the path goes below the digital strike
        first_hit = _first_hit_index(paths[i], n_steps, K)

        # If the event occurs, assign the discounted payout
        if first_hit is not None:
            discounted_payoffs[i] = pay * _discount_factor(r, first_hit, dt)

    # MC estimator = average discounted payoff
    price = float(np.mean(discounted_payoffs))
    return price, discounted_payoffs