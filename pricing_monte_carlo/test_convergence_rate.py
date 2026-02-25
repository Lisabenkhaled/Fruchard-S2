import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price


def convergence_rate_plot(
    market: Market,
    trade: OptionTrade,
    N_list: list[int],
    n_steps: int,
    seeds: list[int],
    antithetic: bool = True,
):

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    bs = bs_price(
        S=market.S0,
        K=trade.strike,
        r=market.r,
        sigma=market.sigma,
        T=trade.T,
        is_call=trade.is_call,
    )

    x_vals, y_vals = [], []

    for N in N_list:
        errs = []

        for sd in seeds:
            mc, discounted_payoffs = price_european_naive_mc_vector(
                market=market,
                trade=trade,
                n_paths=N,
                n_steps=n_steps,
                seed=sd,
                antithetic=antithetic,
            )
            errs.append(abs(mc - bs))

        errs = np.asarray(errs, dtype=float)

        x_vals.append(1.0 / np.sqrt(N))
        y_vals.append(errs.mean())

        print(f"N={N:<7d}  mean|MC-BS|={errs.mean():.6f}")

    x = np.asarray(x_vals)
    y = np.asarray(y_vals)

    # Linear fit
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    y_hat = a * x + b

    plt.figure(figsize=(9, 5.8))
    plt.plot(x, y, marker="o", linewidth=2, label="Mean |MC - BS|")
    plt.plot(x, y_hat, linestyle="--", linewidth=2,
             label=f"Fit: y={a:.3f}x+{b:.3f}")
    plt.title("MC Convergence Rate")
    plt.xlabel("1 / sqrt(N)")
    plt.ylabel("Mean |MC - BS|")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    N_list = [1000, 2000, 5000]
    seeds = list(range(1, 1001))

    convergence_rate_plot(
        market,
        trade,
        N_list=N_list,
        n_steps=100,
        seeds=seeds,
        antithetic=True,
    )