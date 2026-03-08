import numpy as np


class BrownianMotion:
    def __init__(self, seed: int = 0):
        self._gen = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def dW(self, n_paths: int, n_steps: int, dt: float, antithetic: bool = False) -> np.ndarray:
        sqrt_dt = np.sqrt(dt)

        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True")
            half = n_paths // 2
            z_half = self._gen.standard_normal((half, n_steps))
            out = np.empty((n_paths, n_steps), dtype=np.float64)
            out[:half] = z_half
            out[half:] = -z_half
            out *= sqrt_dt
            return out

        return sqrt_dt * self._gen.standard_normal((n_paths, n_steps))

    def W(self, n_paths: int, n_steps: int, dt: float, antithetic: bool = False) -> np.ndarray:
        dW = self.dW(n_paths, n_steps, dt, antithetic=antithetic)
        W = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        W[:, 0] = 0.0
        np.cumsum(dW, axis=1, out=W[:, 1:])
        return W