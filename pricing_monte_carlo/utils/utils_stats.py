import numpy as np
from typing import Tuple

# Basic Statistics Functions
def sample_mean(x: np.ndarray) -> float:
    return float(np.mean(x))

def sample_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    return float(np.std(x, ddof=1))

def sample_std_anti(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n % 2 != 0:
        raise ValueError("Antithetic requires an even number of samples.")

    n_pairs = n // 2
    x_plus = x[:n_pairs]
    x_minus = x[n_pairs:]

    pair_avg = 0.5 * (x_plus + x_minus)

    if n_pairs < 2:
        return float("nan")

    return float(np.std(pair_avg, ddof=1))

# Monte Carlo Statistics

# Standard Error
def standard_error(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 2:
        return float("nan")

    s = np.std(x, ddof=1)
    return float(s / np.sqrt(n))

def standard_error_anti(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n % 2 != 0:
        raise ValueError("Antithetic requires an even number of samples.")

    n_pairs = n // 2
    x_plus = x[:n_pairs]
    x_minus = x[n_pairs:]

    pair_avg = 0.5 * (x_plus + x_minus)

    if n_pairs < 2:
        return float("nan")

    s = np.std(pair_avg, ddof=1)
    return float(s / np.sqrt(n_pairs))

# Confidence Interval
def confidence_interval(x: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float]:

    mean = sample_mean(x)
    se = standard_error(x)

    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)

    lower = mean - z * se
    upper = mean + z * se
    return float(lower), float(upper)

def z_score(mc_price: float, benchmark_price: float, se: float) -> float:
    return (mc_price - benchmark_price) / se if se else 0.0

