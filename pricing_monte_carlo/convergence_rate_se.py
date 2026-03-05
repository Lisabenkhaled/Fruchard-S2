import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector

from utils.utils_stats import (
    standard_error,
    standard_error_anti,
)

from utils.utils_bs import bs_price


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def test_se_linear_in_1_over_sqrtN(
    market: Market,
    trade: OptionTrade,
    N_list: list[int],
    n_steps: int,
    seed: int = 42,
    antithetic: bool = True,
    plot: bool = True,
):
    """
    TEST: SE ∝ 1/sqrt(N)

    - Fixed seed (clean test)
    - Computes SE using your utils:
        * non-anti : standard_error
        * anti     : standard_error_anti

    Two regressions (both shown):
      1) Linear:     SE = a*(1/sqrt(N)) + b   (b should be ~ 0)
      2) Log–log:    log(SE) = alpha + beta*log(N)  (beta should be ~ -0.5)

    Also (optional) prints price error vs BS (only if no dividends are present).
    """

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    N = np.asarray(list(map(int, N_list)), dtype=int)
    if antithetic:
        N = np.where(N % 2 == 0, N, N + 1)
        N = np.unique(N)

    # BS reference is only clean without dividends (q handled, discrete dividends not in BS)
    use_bs = (getattr(trade, "div_amount", 0.0) == 0.0) and (getattr(trade, "ex_div_date", None) is None)
    bs = None
    if use_bs:
        bs = bs_price(
            S=market.S0,
            K=trade.strike,
            r=market.r,
            sigma=market.sigma,
            T=trade.T,
            is_call=trade.is_call,
        )

    x = 1.0 / np.sqrt(N.astype(float))

    se_vals = []
    abs_err_vals = []

    for n_paths in N:
        mc, disc_payoff = price_european_naive_mc_vector(
            market=market,
            trade=trade,
            n_paths=int(n_paths),
            n_steps=n_steps,
            seed=seed,
            antithetic=bool(antithetic),
        )

        if antithetic:
            se = float(standard_error_anti(disc_payoff))
        else:
            se = float(standard_error(disc_payoff))

        se_vals.append(se)

        if use_bs and bs is not None:
            abs_err_vals.append(abs(float(mc) - float(bs)))

        if use_bs and bs is not None:
            print(f"N={int(n_paths):<7d}  SE={se:.6e}  |MC-BS|={abs_err_vals[-1]:.6e}")
        else:
            print(f"N={int(n_paths):<7d}  SE={se:.6e}")

    y = np.asarray(se_vals, dtype=float)

    # -------------------------
    # Regression 1: SE = a*x + b
    # -------------------------
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a * x + b
    r2_lin = _r2(y, y_hat)

    # -------------------------
    # Regression 2: log(SE) = alpha + beta log(N)
    # -------------------------
    logN = np.log(N.astype(float))
    logSE = np.log(y)
    A2 = np.column_stack([np.ones_like(logN), logN])
    alpha, beta = np.linalg.lstsq(A2, logSE, rcond=None)[0]
    logSE_hat = alpha + beta * logN
    r2_log = _r2(logSE, logSE_hat)

    print("\n" + "=" * 95)
    print("SE scaling test")
    print("=" * 95)
    print(f"antithetic={antithetic} | seed={seed} | n_steps={n_steps}")
    print("\n[Linear fit] SE = a*(1/sqrt(N)) + b   (expect b ~ 0)")
    print(f"a={a:.6e}, b={b:.6e}, R2={r2_lin:.6f}")
    print("\n[Log–log fit] log(SE) = alpha + beta*log(N)   (expect beta ~ -0.5)")
    print(f"alpha={alpha:.6f}, beta={beta:.6f}, R2={r2_log:.6f}")

    # -------------------------
    # Plots
    # -------------------------
    if plot:
        # Plot SE vs 1/sqrt(N) + linear fit
        plt.figure(figsize=(9, 5.8))
        plt.plot(x, y, marker="o", linewidth=2, label="SE")
        plt.plot(x, y_hat, linestyle="--", linewidth=2, label=f"Fit: SE={a:.3e}x + {b:.3e} (R2={r2_lin:.3f})")
        plt.title("SE linearity test: SE vs 1/sqrt(N)")
        plt.xlabel("1 / sqrt(N)")
        plt.ylabel("Standard Error (SE)")
        plt.grid(alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot log(SE) vs log(N) + fit
        plt.figure(figsize=(9, 5.8))
        plt.plot(logN, logSE, marker="o", linestyle="", label="log(SE)")
        plt.plot(logN, logSE_hat, linestyle="--", linewidth=2,
                 label=f"Fit slope beta={beta:.3f} (R2={r2_log:.3f})")
        plt.title("Log–log test: slope should be ≈ -0.5")
        plt.xlabel("log(N)")
        plt.ylabel("log(SE)")
        plt.grid(alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Optional: show |MC-BS| vs 1/sqrt(N) if BS is valid
        if use_bs and len(abs_err_vals) == len(N):
            abs_err = np.asarray(abs_err_vals, dtype=float)
            plt.figure(figsize=(9, 5.8))
            plt.plot(x, abs_err, marker="o", linewidth=2, label="|MC - BS| (fixed seed)")
            plt.title("Price error vs 1/sqrt(N) (BS reference)")
            plt.xlabel("1 / sqrt(N)")
            plt.ylabel("|MC - BS|")
            plt.grid(alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return {
        "N": N.astype(int),
        "x_1_over_sqrtN": x,
        "se": y,
        "lin_fit": {"a": float(a), "b": float(b), "r2": float(r2_lin)},
        "loglog_fit": {"alpha": float(alpha), "beta": float(beta), "r2": float(r2_log)},
        "abs_err_vs_bs": np.asarray(abs_err_vals, dtype=float) if use_bs else None,
    }


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

    N_list = [500, 1000, 2000, 5000, 10000, 15000, 20000]

    # Choose ONE of the two tests:
    # 1) non-antithetic:
    test_se_linear_in_1_over_sqrtN(
        market,
        trade,
        N_list=N_list,
        n_steps=100,
        seed=42,
        antithetic=False,
        plot=True,
    )

    # 2) antithetic:
    test_se_linear_in_1_over_sqrtN(
        market,
        trade,
        N_list=N_list,
        n_steps=100,
        seed=42,
        antithetic=True,
        plot=True,
    )