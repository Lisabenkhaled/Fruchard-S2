import numpy as np
from scipy.stats import norm


class BrownianMotion:

    # random number generator
    def __init__(self, seed: int = 0):
        self._gen = np.random.default_rng(seed) if seed != 0 else np.random.default_rng()

    # Generate Brownian increments dW and paths W
    # We decide to add antithetic variates at this level, so that we can reuse the same Brownian paths for both scalar and vector methods, ensuring a fair comparison
    def dW(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        antithetic: bool = False,
        eps: float = 1e-12, 
    ) -> np.ndarray:
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True")
            half = n_paths // 2

            U_half = self._gen.uniform(0.0, 1.0, size=(half, n_steps)) # uniform(0,1) for half paths for vectorized antithetic variates
            U_half = np.clip(U_half, eps, 1.0 - eps)                   # avoid 0 and 1 for numerical stability
            Z_half = norm.ppf(U_half)                                  # inverse CDF to get standard normal for half paths

            Z = np.vstack([Z_half, -Z_half])
        else:
            U = self._gen.uniform(0.0, 1.0, size=(n_paths, n_steps))
            U = np.clip(U, eps, 1.0 - eps)
            Z = norm.ppf(U)

        return np.sqrt(dt) * Z

    # Cumulative sum to get Brownian paths W from increments dW
    def W(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        antithetic: bool = False,
        eps: float = 1e-12,
    ) -> np.ndarray:
        dW = self.dW(n_paths, n_steps, dt, antithetic=antithetic, eps=eps)
        W = np.zeros((n_paths, n_steps + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        return W