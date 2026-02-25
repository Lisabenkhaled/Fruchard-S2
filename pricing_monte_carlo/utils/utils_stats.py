import numpy as np
from typing import Tuple

# Basic Statistics Functions
def sample_mean(x: np.ndarray) -> float:
    return float(np.mean(x))

def sample_variance(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1))

def sample_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1))

# Monte Carlo Statistics

# Standard Error
def standard_error(x: np.ndarray) -> float:
    n = len(x)
    return float(np.std(x, ddof=1) / np.sqrt(n))

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

