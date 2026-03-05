import numpy as np

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector, simulate_gbm_paths_scalar

# Option C and D in QCM quantitative
def price_american_digital_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    digital_strike: float,
    payout: float = 1.0,
    seed: int = 0,
    antithetic: bool = False,
) -> tuple[float, np.ndarray]:

    T = float(trade.T)
    dt = T / int(n_steps)

    _, paths = simulate_gbm_paths_vector(
        market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic
    )

    K = float(digital_strike)
    payout = float(payout)

    itm = (paths < K)
    any_itm = np.any(itm, axis=1)
    first_idx = np.argmax(itm, axis=1).astype(int)

    disc = np.exp(-float(market.r) * (first_idx * dt))
    discounted_payoffs = np.where(any_itm, payout * disc, 0.0)

    return float(np.mean(discounted_payoffs)), discounted_payoffs


def price_american_digital_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    digital_strike: float,
    payout: float = 1.0,
    seed: int = 0,
    antithetic: bool = False,
) -> tuple[float, np.ndarray]:
    T = float(trade.T)
    dt = T / int(n_steps)

    _, paths = simulate_gbm_paths_scalar(
        market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic
    )  # (n_paths, n_steps+1)

    K = float(digital_strike)
    payout = float(payout)

    # for each path i, find first j such that paths[i, j] < K
    discounted_payoffs = np.zeros(n_paths, dtype=float)

    r = float(market.r)

    for i in range(int(n_paths)):
        first_hit = None
        for j in range(int(n_steps) + 1):
            if paths[i, j] < K:
                first_hit = j
                break

        if first_hit is not None:
            discounted_payoffs[i] = payout * np.exp(-r * (first_hit * dt))
        # else remains 0.0

    price = float(np.mean(discounted_payoffs))
    return price, discounted_payoffs