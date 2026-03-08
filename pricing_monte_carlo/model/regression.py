import numpy as np
from typing import Literal

Basis = Literal["power", "laguerre"]


def _laguerre_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate Laguerre polynomials
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    n = z.shape[0]

    X = np.empty((n, degree + 1), dtype=float)
    X[:, 0] = 1.0

    if degree >= 1:
        X[:, 1] = 1.0 - z

    for k in range(1, degree):
        # recurrence applied element-wise across all n points simultaneously
        X[:, k + 1] = ((2.0 * k + 1.0 - z) * X[:, k] - k * X[:, k - 1]) / (k + 1.0)

    return X


def _power_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate the monomial basis
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    n = z.shape[0]

    X = np.empty((n, degree + 1), dtype=float)
    X[:, 0] = 1.0

    if degree >= 1:
        X[:, 1] = z
        for d in range(2, degree + 1):
            # each column is built from the previous — vectorized over n
            X[:, d] = X[:, d - 1] * z

    return X


def design_matrix(
    S: np.ndarray,
    degree: int = 2,
    basis: Basis = "power",
    scale: float | None = None
) -> np.ndarray:
    """
    Build a regression design matrix from stock prices S.
    scale  : if provided, S is normalized by scale (typically the strike K)
             before basis evaluation, improving numerical conditioning
    """
    S = np.asarray(S, dtype=float).reshape(-1)
    z = S / float(scale) if scale is not None else S

    if basis == "power":
        return _power_basis(z, degree)
    if basis == "laguerre":
        return _laguerre_basis(z, degree)
    raise ValueError(f"Unknown basis '{basis}'. Choose 'power' or 'laguerre'.")


def ols_fit_predict(X: np.ndarray, y: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """
    Fit OLS coefficients 
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return X_pred @ beta