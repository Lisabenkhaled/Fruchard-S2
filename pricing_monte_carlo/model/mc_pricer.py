import numpy as np
from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector, simulate_gbm_paths_scalar
from model.regression import design_matrix, ols_fit_predict

# Monte Carlo pricers for European options scalar version
def price_european_naive_mc_scalar(
    market: Market, trade: OptionTrade,
    n_paths: int, n_steps: int,
    seed: int = 0, antithetic: bool = False
) -> float:
    
    _, paths = simulate_gbm_paths_scalar(market, trade, n_paths, n_steps, seed, antithetic)
    disc = np.exp(-market.r * trade.T)

    discounted_payoffs = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        discounted_payoffs[i] = disc * trade.payoff_scalar(paths[i, -1])

    price = float(discounted_payoffs.mean())
    return price, discounted_payoffs

# Monte Carlo pricers for European options vectorized version
def price_european_naive_mc_vector(
    market: Market, trade: OptionTrade,
    n_paths: int, n_steps: int,
    seed: int = 0, antithetic: bool = False
) -> float:
    
    _, paths = simulate_gbm_paths_vector(market, trade, n_paths, n_steps, seed, antithetic)
    ST = paths[:, -1]
    payoff = trade.payoff_vector(ST)
    disc = np.exp(-market.r * trade.T)
    discounted_payoffs = disc * payoff

    price = float(discounted_payoffs.mean())
    return price, discounted_payoffs


# Naive American MC scalar
def price_american_naive_mc_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
) -> float:
    
    if trade.exercise.lower() != "american":
        raise ValueError("Trade is not american")

    times, paths = simulate_gbm_paths_scalar(market, trade, n_paths, n_steps, seed, antithetic)
    disc = np.exp(-market.r * times)

    best_pv = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        best = -np.inf
        for j in range(paths.shape[1]):
            ex = trade.payoff_scalar(paths[i, j])
            pv = ex * disc[j]
            if pv > best:
                best = pv
        best_pv[i] = best

    price = float(best_pv.mean())
    return price, best_pv

# Naive American MC vectorized
def price_american_naive_mc_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
) -> float:
    
    if trade.exercise.lower() != "american":
        raise ValueError("Trade is not american")

    times, paths = simulate_gbm_paths_vector(market, trade, n_paths, n_steps, seed, antithetic)
    disc = np.exp(-market.r * times)  

    # intrinsic values along the grid for each path and step
    try:
        intrinsic = trade.payoff_vector(paths)
    except Exception:
        intrinsic = np.empty_like(paths)
        for j in range(paths.shape[1]):
            intrinsic[:, j] = trade.payoff_vector(paths[:, j])

    pv = intrinsic * disc[None, :]
    best_pv = np.max(pv, axis=1)

    price = float(best_pv.mean())
    return price, best_pv


# Longstaff–Schwartz American option pricer scalar version
def price_american_ls_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
    basis: str = "laguerre",
    degree: int = 2,
) -> float:

    _, S = simulate_gbm_paths_scalar(
        market=market,
        trade=trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )

    r = float(market.r)
    dt = float(trade.T) / n_steps
    df = np.exp(-r * dt)


    V = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        V[i] = trade.payoff_scalar(S[i, -1])

    # Backward induction
    for j in range(n_steps - 1, 0, -1):
        for i in range(n_paths):
            V[i] = df * V[i]

        # Compute intrinsic values at time j
        exercise = np.empty(n_paths, dtype=float)
        for i in range(n_paths):
            exercise[i] = trade.payoff_scalar(S[i, j])

        # ITM indices
        itm_idx = np.where(exercise > 0.0)[0]
        if itm_idx.size == 0:
            continue

        Sj_itm = S[itm_idx, j]
        Y = V[itm_idx]

        # Regression on ITM only
        X = design_matrix(
            Sj_itm,
            degree=degree,
            basis=basis,
            scale=trade.strike,
        )

        cont = ols_fit_predict(X, Y, X)

        # Exercise decision
        for k, idx in enumerate(itm_idx):
            if exercise[idx] >= cont[k]:
                V[idx] = exercise[idx]

    discounted_cf = df * V
    price = float(discounted_cf.mean())

    return price, discounted_cf


# Longstaff–Schwartz American option pricer vectorized version
def price_american_ls_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = False,
    basis: str = "laguerre",
    degree: int = 2,
) -> float:

    _, S = simulate_gbm_paths_vector(
        market=market,
        trade=trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )

    r = float(market.r)
    dt = float(trade.T) / n_steps
    df = np.exp(-r * dt)

    V = trade.payoff_vector(S[:, -1])

    for j in range(n_steps - 1, 0, -1):
        Sj = S[:, j]
        exercise = trade.payoff_vector(Sj)

        itm = exercise > 0.0

        V = df * V

        if np.any(itm):
            Y = V[itm]

            # Build regressors on ITM paths only
            X = design_matrix(
                Sj[itm],
                degree=degree,
                basis=basis,
                scale=trade.strike,
            )

            # Continuation estimate on ITM
            cont = ols_fit_predict(X, Y, X)
            ex_now = exercise[itm] >= cont

            V_itm = V[itm]
            V_itm = np.where(ex_now, exercise[itm], V_itm)
            V[itm] = V_itm

    discounted_cf = df * V
    price = float(discounted_cf.mean())
    
    return price, discounted_cf