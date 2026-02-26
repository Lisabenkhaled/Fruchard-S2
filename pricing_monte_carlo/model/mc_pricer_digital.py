import numpy as np

from model.market import Market
from model.option import OptionTrade


def _year_frac(pricing_date, d):
    # simple ACT/365
    return (d - pricing_date).days / 365.0


def _ex_div_step(trade: OptionTrade, n_steps: int) -> int | None:
    """
    Map ex-div date to a step index in [0..n_steps].
    Returns None if no discrete dividend.
    """
    if trade.ex_div_date is None or trade.div_amount == 0.0:
        return None

    T = trade.T
    if T <= 0:
        return None

    t_div = _year_frac(trade.pricing_date, trade.ex_div_date)
    if t_div <= 0:
        return 0
    if t_div >= T:
        return n_steps

    j = int(round((t_div / T) * n_steps))
    return max(0, min(n_steps, j))


def _simulate_gbm_paths_matrix(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    seed: int = 0,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simulate GBM paths: returns S with shape (n_steps+1, n_paths).
    Includes:
      - continuous dividend yield q in drift (r - q)
      - one discrete dividend (subtract div_amount at ex-div step)
    """
    if antithetic and (n_paths % 2 != 0):
        raise ValueError("n_paths must be even when antithetic=True")

    rng = np.random.default_rng(seed)

    T = trade.T
    dt = T / n_steps
    mu = (market.r - trade.q - 0.5 * market.sigma**2) * dt
    vol = market.sigma * np.sqrt(dt)

    # normals
    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(size=(n_steps, half))
        Z = np.concatenate([Z_half, -Z_half], axis=1)
    else:
        Z = rng.standard_normal(size=(n_steps, n_paths))

    S = np.empty((n_steps + 1, n_paths), dtype=float)
    S[0, :] = market.S0

    j_div = _ex_div_step(trade, n_steps)
    div = float(trade.div_amount)

    # step-by-step to apply discrete dividend at the right time
    for t in range(n_steps):
        S[t + 1, :] = S[t, :] * np.exp(mu + vol * Z[t, :])

        # apply discrete dividend at ex-div step (after evolving to that time)
        if j_div is not None and (t + 1) == j_div and div != 0.0:
            S[t + 1, :] = np.maximum(S[t + 1, :] - div, 0.0)

    return S


def price_american_digital_first_hit_vector(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    digital_strike: float,
    payout: float = 1.0,
    seed: int = 0,
    antithetic: bool = False,
):
    """
    American DIGITAL (cash-or-nothing, down):
        if exercise at time t and S_t < K => receive payout, else 0.

    With r >= 0, optimal policy on discrete grid:
        exercise at FIRST time S_t < K.

    Returns (price, discounted_payoffs) where discounted_payoffs are PV at t=0.
    """
    S = _simulate_gbm_paths_matrix(
        market, trade, n_paths, n_steps, seed=seed, antithetic=antithetic
    )

    K = float(digital_strike)
    payout = float(payout)

    dt = trade.T / n_steps

    itm = (S < K)                      # (n_steps+1, n_paths)
    any_itm = np.any(itm, axis=0)      # (n_paths,)
    first_idx = np.argmax(itm, axis=0) # if never ITM -> 0, fixed by any_itm mask

    disc = np.exp(-market.r * (first_idx * dt))
    discounted_payoffs = np.where(any_itm, payout * disc, 0.0)

    price = float(np.mean(discounted_payoffs))
    return price, discounted_payoffs


# if you still want "scalar" API compatibility:
def price_american_digital_first_hit_scalar(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    *,
    digital_strike: float,
    payout: float = 1.0,
    seed: int = 0,
    antithetic: bool = False,
):
    # fallback: just call vector version (still correct)
    return price_american_digital_first_hit_vector(
        market, trade, n_paths, n_steps,
        digital_strike=digital_strike, payout=payout,
        seed=seed, antithetic=antithetic
    )