import numpy as np
from typing import Literal

Basis = Literal["power", "laguerre"]

# Laguerre polynomials basis functions
def _laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]

    L = np.empty((n, degree + 1), dtype=float)
    L[:, 0] = 1.0

    if degree >= 1:
        L[:, 1] = 1.0 - x

    for k in range(1, degree):
        L[:, k + 1] = ((2.0 * k + 1.0 - x) * L[:, k] - k * L[:, k - 1]) / (k + 1.0)

    return L


# Design matrix for regression, supporting both power and Laguerre bases
def design_matrix(
    S: np.ndarray,
    degree: int = 2,
    basis: Basis = "power",
    scale: float | None = None,
) -> np.ndarray:
    S = np.asarray(S, dtype=float).reshape(-1)

    if degree < 0:
        raise ValueError("degree must be >= 0")
    
    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be positive if provided.")
        z = S / float(scale)
    else:
        z = S

    if basis == "power":
        cols = [np.ones_like(z)]
        for d in range(1, degree + 1):
            cols.append(z ** d)
        return np.column_stack(cols)

    if basis == "laguerre":
        if scale is None:
            raise ValueError("For Laguerre basis, provide scale (e.g., strike K) for z=S/K.")
        return _laguerre_basis(z, degree)

    raise ValueError(f"Unknown basis='{basis}'")

# Ordinary Least Squares regression fit and predict
def ols_fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return X_pred @ beta
