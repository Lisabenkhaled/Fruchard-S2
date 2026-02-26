import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price
from utils.utils_stats import (
    sample_mean,
    sample_std,
    standard_error,
    sample_std_anti,
    standard_error_anti,
)


def study_antithetic_effect_on_std_eu(
    market: Market,
    trade: OptionTrade,
    N_list: list[int],
    n_steps: int,
    seeds: list[int],
):
    if trade.exercise.lower() != "european":
        raise ValueError("This study is for EUROPEAN options only.")

    bs = bs_price(
        S=market.S0,
        K=trade.strike,
        r=market.r,
        sigma=market.sigma,
        T=trade.T,
        is_call=trade.is_call
    )

    # 1) Across-seeds price dispersion (what you already had)
    price_std_no = []
    price_std_anti = []
    price_mean_no = []
    price_mean_anti = []

    # 2) Proper MC payoff std & SE (computed per seed, then aggregated)
    payoff_std_no_mean = []
    payoff_std_anti_mean = []
    se_no_mean = []
    se_anti_mean = []

    for N in N_list:
        prices_no = []
        prices_anti = []

        payoff_std_no_seed = []
        payoff_std_anti_seed = []
        se_no_seed = []
        se_anti_seed = []

        for sd in seeds:
            # no antithetic
            p0, pay0 = price_european_naive_mc_vector(
                market=market, trade=trade,
                n_paths=N, n_steps=n_steps,
                seed=sd, antithetic=False
            )

            # antithetic
            p1, pay1 = price_european_naive_mc_vector(
                market=market, trade=trade,
                n_paths=N, n_steps=n_steps,
                seed=sd, antithetic=True
            )

            prices_no.append(p0)
            prices_anti.append(p1)

            # non anti
            payoff_std_no_seed.append(sample_std(pay0))
            se_no_seed.append(standard_error(pay0))

            # anti
            payoff_std_anti_seed.append(sample_std_anti(pay1))
            se_anti_seed.append(standard_error_anti(pay1))

        prices_no = np.asarray(prices_no, dtype=float)
        prices_anti = np.asarray(prices_anti, dtype=float)

        payoff_std_no_seed = np.asarray(payoff_std_no_seed, dtype=float)
        payoff_std_anti_seed = np.asarray(payoff_std_anti_seed, dtype=float)
        se_no_seed = np.asarray(se_no_seed, dtype=float)
        se_anti_seed = np.asarray(se_anti_seed, dtype=float)

        # across seeds (price)
        price_mean_no.append(sample_mean(prices_no))
        price_mean_anti.append(sample_mean(prices_anti))
        price_std_no.append(sample_std(prices_no))
        price_std_anti.append(sample_std(prices_anti))

        # across seeds (payoff std + SE)
        payoff_std_no_mean.append(float(np.nanmean(payoff_std_no_seed)))
        payoff_std_anti_mean.append(float(np.nanmean(payoff_std_anti_seed)))
        se_no_mean.append(float(np.nanmean(se_no_seed)))
        se_anti_mean.append(float(np.nanmean(se_anti_seed)))

        # quick print
        print(
            f"N={N:<7d} | "
            f"price_std(no)={price_std_no[-1]:.6f} | price_std(anti)={price_std_anti[-1]:.6f} | "
            f"price_mean(no)={price_mean_no[-1]:.6f} | price_mean(anti)={price_mean_anti[-1]:.6f} | "
            f"meanSE(no)={se_no_mean[-1]:.6f} | meanSE(anti)={se_anti_mean[-1]:.6f 
                                                }"
        )

    N_arr = np.asarray(N_list, dtype=float)

    price_std_no = np.asarray(price_std_no, dtype=float)
    price_std_anti = np.asarray(price_std_anti, dtype=float)
    price_mean_no = np.asarray(price_mean_no, dtype=float)
    price_mean_anti = np.asarray(price_mean_anti, dtype=float)

    payoff_std_no_mean = np.asarray(payoff_std_no_mean, dtype=float)
    payoff_std_anti_mean = np.asarray(payoff_std_anti_mean, dtype=float)
    se_no_mean = np.asarray(se_no_mean, dtype=float)
    se_anti_mean = np.asarray(se_anti_mean, dtype=float)

    # =========================
    # Plot 1: std(price) across seeds (your original idea)
    # =========================
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.plot(N_arr, price_std_no, marker="o", linewidth=2, label="Std(price) across seeds (no anti)")
    ax.plot(N_arr, price_std_anti, marker="s", linewidth=2, label="Std(price) across seeds (anti)")
    ax.set_xscale("log")
    ax.set_title("EU MC: effect of antithetic on dispersion of price across seeds")
    ax.set_xlabel("N (number of paths) [log scale]")
    ax.set_ylabel("Std of MC price across seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # Plot 2: mean standard error (computed correctly from payoffs)
    # =========================
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.plot(N_arr, se_no_mean, marker="o", linewidth=2, label="Mean SE (no anti)")
    ax.plot(N_arr, se_anti_mean, marker="s", linewidth=2, label="Mean SE (anti)")
    ax.set_xscale("log")
    ax.set_title("EU MC: effect of antithetic on standard error (from payoffs)")
    ax.set_xlabel("N (number of paths) [log scale]")
    ax.set_ylabel("Mean standard error across seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n===============================")
    print("Summary (EU only)")
    print("===============================")
    print(f"BS reference = {bs:.6f}")
    for i, N in enumerate(N_list):
        print(
            f"N={N:<7d} | "
            f"mean(no)={price_mean_no[i]:.6f} | mean(anti)={price_mean_anti[i]:.6f} | "
            f"std_price(no)={price_std_no[i]:.6f} | std_price(anti)={price_std_anti[i]:.6f} | "
            f"meanSE(no)={se_no_mean[i]:.6f} | meanSE(anti)={se_anti_mean[i]:.6f}"
        )

    return {
        "N": N_arr,
        "bs": bs,
        "price_mean_no": price_mean_no,
        "price_mean_anti": price_mean_anti,
        "price_std_no": price_std_no,
        "price_std_anti": price_std_anti,
        "payoff_std_no_mean": payoff_std_no_mean,
        "payoff_std_anti_mean": payoff_std_anti_mean,
        "se_no_mean": se_no_mean,
        "se_anti_mean": se_anti_mean,
    }


if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    market = Market(S0=100.0, r=0.05, sigma=0.20)

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0
    )

    N_list = [1000, 2000, 5000]
    seeds = list(range(1, 501))

    study_antithetic_effect_on_std_eu(
        market=market,
        trade=trade,
        N_list=N_list,
        n_steps=100,
        seeds=seeds
    )