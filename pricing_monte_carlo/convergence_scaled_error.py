import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price


def _plot_histogram(scaled_errors: np.ndarray, mean: float, std: float) -> None:
    # histogram
    plt.figure(figsize=(9, 5.8))  # fig
    count, bins, _ = plt.hist(
        scaled_errors,
        bins=40,
        density=True,
        alpha=0.6,
        label="Scaled errors",
    )  # hist

    # normal fit
    x = np.linspace(bins.min(), bins.max(), 500)  # grid
    plt.plot(x, norm.pdf(x, mean, std), linewidth=2, label=f"Normal fit (μ={mean:.2f}, σ={std:.2f})")  # fit

    # style
    plt.title(r"Histogram of $\sqrt{N}(MC - BS)$")  # title
    plt.xlabel(r"$\sqrt{N}(MC - BS)$")  # x
    plt.ylabel("Density")  # y
    plt.grid(alpha=0.3)  # grid
    plt.legend()  # legend
    plt.tight_layout()  # layout
    plt.show()  # show


def scaled_error_histogram(market: Market, trade: OptionTrade, N: int,
    n_steps: int, seeds: List[int], antithetic: bool = True) -> np.ndarray:
    # check
    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")  # eu only

    # BS ref
    bs = bs_price(
        S=market.S0,  # spot
        K=trade.strike,  # strike
        r=market.r,  # rate
        sigma=market.sigma,  # vol
        T=trade.T,  # maturity
        is_call=trade.is_call,  # call/put
    )

    # compute errors
    scaled_errors: List[float] = []  # store
    for sd in seeds:
        mc, discounted_payoffs = price_european_naive_mc_vector(
            market=market,  # market
            trade=trade,  # trade
            n_paths=N,  # paths
            n_steps=n_steps,  # steps
            seed=sd,  # seed
            antithetic=antithetic,  # anti
        )
        scaled_errors.append((float(mc) - float(bs)) * float(np.sqrt(N)))  # error

    # stats
    arr = np.asarray(scaled_errors, dtype=float)  # array
    mean = float(arr.mean())  # mean
    std = float(arr.std(ddof=1))  # std

    # print
    print("\n==============================")  # line
    print(f"N = {N}")  # N
    print(f"Mean scaled error = {mean:.6f}")  # mean
    print(f"Std  scaled error = {std:.6f}")  # std
    print("==============================")  # line

    # plot
    _plot_histogram(arr, mean, std)  # plot

    return arr


if __name__ == "__main__":
    import datetime as dt  # dates

    # dates
    pricing_date = dt.date(2026, 2, 18)  # t0
    maturity_date = dt.date(2027, 2, 18)  # T

    # market
    market = Market(S0=100.0, r=0.05, sigma=0.30)  # market

    # trade
    trade = OptionTrade(
        strike=102.0,  # strike
        is_call=True,  # call
        exercise="european",  # eu
        pricing_date=pricing_date,  # t0
        maturity_date=maturity_date,  # T
        q=0.0,  # q
        ex_div_date=None,  # div date
        div_amount=0.0,  # div
    )

    # params
    N = 10_000  # paths
    seeds = list(range(1, 1001))  # seeds

    # run
    scaled_error_histogram(
        market=market,  # market
        trade=trade,  # trade
        N=N,  # N
        n_steps=300,  # steps
        seeds=seeds,  # seeds
        antithetic=True,  # anti
    )