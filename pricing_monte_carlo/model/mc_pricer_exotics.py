import numpy as np
from typing import Literal, Sequence

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector, simulate_gbm_paths_scalar
from model.regression import design_matrix, ols_fit_predict

Basis = Literal["power", "laguerre"]


def _vanilla_put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)


def _digital_down_payoff(S: np.ndarray, K: float, payout: float) -> np.ndarray:
    return (S < K).astype(float) * payout


def _discount_factor(r: float, dt: float) -> float:
    return float(np.exp(-r * dt))


def price_bermudan_put_ls_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    exercise_steps: Sequence[int],
    seed: int = 0,
    antithetic: bool = False,
    basis: Basis = "laguerre",
    degree: int = 2,
):

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    ex_set = set(int(i) for i in exercise_steps)
    if n_steps not in ex_set:
        ex_set.add(n_steps)

    _, S = simulate_gbm_paths_vector(
        market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic
    )
    if S.shape[0] != n_steps + 1:
        if S.shape[1] == n_steps + 1:
            S = S.T
        else:
            raise ValueError("Unexpected path shape from simulate_gbm_paths_vector")

    # Time step in years (assumes uniform grid between pricing and maturity)
    T = trade.T
    dt = T / n_steps
    disc = _discount_factor(market.r, dt)

    K = trade.strike

    # Cashflow at maturity
    V = _vanilla_put_payoff(S[n_steps], K)

    # Backward induction
    for t in range(n_steps - 1, -1, -1):
        V *= disc

        if t not in ex_set:
            continue 

        St = S[t]
        immediate = _vanilla_put_payoff(St, K)

        itm = immediate > 0.0
        if np.any(itm):
            X = design_matrix(St[itm], basis=basis, degree=degree, scale=K)
            cont = np.zeros_like(V)
            cont[itm] = ols_fit_predict(X, V[itm], X)
        else:
            cont = np.zeros_like(V)

        exercise = immediate >= cont
        V = np.where(exercise, immediate, V)

    price = float(np.mean(V))
    discounted_payoffs = V
    return price, discounted_payoffs


def price_american_digital_ls_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    digital_strike: float,
    payout: float = 1.0,
    seed: int = 0,
    antithetic: bool = False,
    basis: Basis = "laguerre",
    degree: int = 2,
):
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    _, S = simulate_gbm_paths_vector(
        market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic
    )
    if S.shape[0] != n_steps + 1:
        if S.shape[1] == n_steps + 1:
            S = S.T
        else:
            raise ValueError("Unexpected path shape from simulate_gbm_paths_vector")

    T = trade.T
    dt = T / n_steps
    disc = _discount_factor(market.r, dt)

    Kd = float(digital_strike)

    # maturity value
    V = _digital_down_payoff(S[n_steps], Kd, payout)

    for t in range(n_steps - 1, -1, -1):
        V *= disc
        St = S[t]
        immediate = _digital_down_payoff(St, Kd, payout)

        itm = immediate > 0.0
        if np.any(itm):
            X = design_matrix(St[itm], basis=basis, degree=degree, scale=Kd)
            cont = np.zeros_like(V)
            cont[itm] = ols_fit_predict(X, V[itm], X)
        else:
            cont = np.zeros_like(V)

        exercise = immediate >= cont
        V = np.where(exercise, immediate, V)

    price = float(np.mean(V))
    discounted_payoffs = V
    return price, discounted_payoffs

def price_bermudan_put_ls_scalar(*args, **kwargs):
    return price_bermudan_put_ls_vector(*args, **kwargs)

def price_american_digital_ls_scalar(*args, **kwargs):
    return price_american_digital_ls_vector(*args, **kwargs)