import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price
from utils.utils_stats import sample_mean, sample_std, sample_variance, standard_error 


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

    std_no = []
    std_anti = []
    mean_no = []
    mean_anti = []
    se_no = []
    se_anti = []

    for N in N_list:
        prices_no = []
        prices_anti = []

        for sd in seeds:
            p0, _ = price_european_naive_mc_vector(
                market=market, trade=trade,
                n_paths=N, n_steps=n_steps,
                seed=sd, antithetic=False
            )
            p1, _ = price_european_naive_mc_vector(
                market=market, trade=trade,
                n_paths=N, n_steps=n_steps,
                seed=sd, antithetic=True
            )
            prices_no.append(p0)
            prices_anti.append(p1)

        prices_no = np.asarray(prices_no, dtype=float)
        prices_anti = np.asarray(prices_anti, dtype=float)

        mean_no.append(sample_mean(prices_no))
        mean_anti.append(sample_mean(prices_anti))

        std_no.append(sample_std(prices_no))
        std_anti.append(sample_std(prices_anti))

        # std error
        se_no = standard_error(prices_no)
        se_anti = standard_error(prices_anti)

        print(f"N={N:<7d} | "
              f"std(no)={std_no[-1]:.6f} | std(anti)={std_anti[-1]:.6f} | "
              f"mean(no)={mean_no[-1]:.6f} | mean(anti)={mean_anti[-1]:.6f} | "
              f"se(no)={se_no:.6f} | se(anti)={se_anti:.6f}")

    std_no = np.asarray(std_no)
    std_anti = np.asarray(std_anti)
    mean_no = np.asarray(mean_no)
    mean_anti = np.asarray(mean_anti)
    N_arr = np.asarray(N_list, dtype=float)

    # Plotting the standard deviations
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.plot(N_arr, std_no, marker="o", linewidth=2, label="Std (no antithetic)")
    ax.plot(N_arr, std_anti, marker="s", linewidth=2, label="Std (antithetic)")
    ax.set_xscale("log")
    ax.set_title("European MC: effect of antithetic on std(price)")
    ax.set_xlabel("N (number of paths) [log scale]")
    ax.set_ylabel("Std of MC price (across seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


    print("\n===============================")
    print("Summary (EU only)")
    print("===============================")
    print(f"BS reference = {bs:.6f}")
    for i, N in enumerate(N_list):
        print(f"N={N:<7d} | mean(no)={mean_no[i]:.6f} | mean(anti)={mean_anti[i]:.6f} | "
              f"std(no)={std_no[i]:.6f} | std(anti)={std_anti[i]:.6f}")

    return {
        "N": N_arr,
        "std_no": std_no,
        "std_anti": std_anti,
        "se_no": se_no,
        "se_anti": se_anti,
        "mean_no": mean_no,
        "mean_anti": mean_anti,
        "bs": bs
    }


if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 2, 18)
    maturity_date = dt.date(2027, 2, 18)

    market = Market(S0=100.0, r=0.03, sigma=0.25)

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0
    )

    N_list = [1000, 2000, 5000, 10000, 20000, 50000]
    seeds = list(range(1, 501))

    study_antithetic_effect_on_std_eu(
        market=market,
        trade=trade,
        N_list=N_list,
        n_steps=100,
        seeds=seeds
    )