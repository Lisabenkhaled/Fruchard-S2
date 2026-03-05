import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price


def scaled_error_histogram(
    market: Market,
    trade: OptionTrade,
    N: int,
    n_steps: int,
    seeds: list[int],
    antithetic: bool = True,
):

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    # Black–Scholes reference
    bs = bs_price(
        S=market.S0,
        K=trade.strike,
        r=market.r,
        sigma=market.sigma,
        T=trade.T,
        is_call=trade.is_call,
    )

    scaled_errors = []

    for sd in seeds:
        mc, discounted_payoffs = price_european_naive_mc_vector(
            market=market,
            trade=trade,
            n_paths=N,
            n_steps=n_steps,
            seed=sd,
            antithetic=antithetic,
        )
        scaled_errors.append((mc - bs) * np.sqrt(N))

    scaled_errors = np.asarray(scaled_errors, dtype=float)

    mean = scaled_errors.mean()
    std = scaled_errors.std(ddof=1)

    print("\n==============================")
    print(f"N = {N}")
    print(f"Mean scaled error = {mean:.6f}")
    print(f"Std  scaled error = {std:.6f}")
    print("==============================")

    # Histogram
    plt.figure(figsize=(9, 5.8))
    count, bins, _ = plt.hist(
        scaled_errors,
        bins=40,
        density=True,
        alpha=0.6,
        label="Scaled errors"
    )

    # Normal fit overlay
    x = np.linspace(bins.min(), bins.max(), 500)
    plt.plot(
        x,
        norm.pdf(x, mean, std),
        linewidth=2,
        label=f"Normal fit (μ={mean:.2f}, σ={std:.2f})"
    )

    plt.title(r"Histogram of $\sqrt{N}(MC - BS)$")
    plt.xlabel(r"$\sqrt{N}(MC - BS)$")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return scaled_errors


if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 2, 18)
    maturity_date = dt.date(2027, 2, 18)

    market = Market(S0=100.0, r=0.05, sigma=0.30)

    trade = OptionTrade(
        strike=102.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0,
    )

    N = 10000
    seeds = list(range(1, 1001))

    scaled_error_histogram(
        market=market,
        trade=trade,
        N=N,
        n_steps=300,
        seeds=seeds,
        antithetic=True,
    )