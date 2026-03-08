from __future__ import annotations

import datetime as dt
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_stats import standard_error, standard_error_anti
from utils.utils_bs import bs_price


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Compute the coefficient of determination"""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan


def _prepare_n_values(N_list: list[int], antithetic: bool) -> np.ndarray:
    """Build the array of path counts used in the test"""
    n_values = np.asarray(list(map(int, N_list)), dtype=int)

    # Antithetic sampling requires an even number of paths.
    if antithetic:
        n_values = np.where(n_values % 2 == 0, n_values, n_values + 1)
        n_values = np.unique(n_values)

    return n_values


def _bs_reference_if_valid(market: Market, trade: OptionTrade) -> float | None:
    """Return the Black-Scholes price when no discrete dividends are present"""
    has_dividend = getattr(trade, "div_amount", 0.0) != 0.0
    has_ex_div_date = getattr(trade, "ex_div_date", None) is not None

    # BS is used only in the clean no-discrete-dividend case.
    if has_dividend or has_ex_div_date:
        return None

    return float(
        bs_price(
            S=market.S0,
            K=trade.strike,
            r=market.r,
            sigma=market.sigma,
            T=trade.T,
            is_call=trade.is_call,
        )
    )


def _compute_se(antithetic: bool, disc_payoff: np.ndarray) -> float:
    """Compute the Monte Carlo standard error with the chosen estimator"""
    if antithetic:
        return float(standard_error_anti(disc_payoff))
    return float(standard_error(disc_payoff))


def _simulate_one_n(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
                    seed: int, antithetic: bool) -> tuple[float, np.ndarray]:
    """Run one MC pricing experiment for a given number of paths"""
    mc, disc_payoff = price_european_naive_mc_vector(
        market=market,
        trade=trade,
        n_paths=int(n_paths),
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )
    return float(mc), np.asarray(disc_payoff, dtype=float)


def _print_one_result(n_paths: int, se: float, mc: float, bs_ref: float | None) -> float | None:
    """Print one result line and return the absolute BS pricing error if available"""
    if bs_ref is None:
        print(f"N={int(n_paths):<7d}  SE={se:.6e}")
        return None

    abs_err = abs(mc - bs_ref)
    print(f"N={int(n_paths):<7d}  SE={se:.6e}  |MC-BS|={abs_err:.6e}")
    return float(abs_err)


def _collect_statistics(market: Market, trade: OptionTrade, n_values: np.ndarray, n_steps: int,
                        seed: int, antithetic: bool, bs_ref: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute SE values and optional BS absolute errors for all path counts"""
    se_vals: list[float] = []
    abs_err_vals: list[float] = []

    # Loop over the different Monte Carlo sample sizes.
    for n_paths in n_values:
        mc, disc_payoff = _simulate_one_n(
            market,
            trade,
            int(n_paths),
            n_steps,
            seed,
            antithetic,
        )
        se = _compute_se(antithetic, disc_payoff)
        abs_err = _print_one_result(int(n_paths), se, mc, bs_ref)

        se_vals.append(se)
        if abs_err is not None:
            abs_err_vals.append(abs_err)

    return np.asarray(se_vals, dtype=float), np.asarray(abs_err_vals, dtype=float)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """Fit SE = ax + b"""
    a_mat = np.column_stack([x, np.ones_like(x)])
    a_coef, b_coef = np.linalg.lstsq(a_mat, y, rcond=None)[0]
    y_hat = a_coef * x + b_coef
    r2_lin = _r2(y, y_hat)
    return float(a_coef), float(b_coef), y_hat, float(r2_lin)


def _loglog_fit(n_values: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Fit log(SE) = alpha + beta*log(N)"""
    log_n = np.log(n_values.astype(float))
    log_se = np.log(y)
    a_mat = np.column_stack([np.ones_like(log_n), log_n])
    alpha, beta = np.linalg.lstsq(a_mat, log_se, rcond=None)[0]
    log_se_hat = alpha + beta * log_n
    r2_log = _r2(log_se, log_se_hat)
    return float(alpha), float(beta), log_n, log_se_hat, float(r2_log)


def _print_summary(antithetic: bool, seed: int, n_steps: int, 
                   a_coef: float, b_coef: float, 
                   r2_lin: float, alpha: float,
                   beta: float, r2_log: float) -> None:
    """Print the regression summary for the SE scaling test"""
    print("\n" + "=" * 95)
    print("SE scaling test")
    print("=" * 95)
    print(f"antithetic={antithetic} | seed={seed} | n_steps={n_steps}")

    # First regression: linear relation in 1/sqrt(N).
    print("\n[Linear fit] SE = a*(1/sqrt(N)) + b   (expect b ~ 0)")
    print(f"a={a_coef:.6e}, b={b_coef:.6e}, R2={r2_lin:.6f}")

    # Second regression
    print("\n[Log-log fit] log(SE) = alpha + beta*log(N)   (expect beta ~ -0.5)")
    print(f"alpha={alpha:.6f}, beta={beta:.6f}, R2={r2_log:.6f}")


def _plot_linear_fit(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
                     a_coef: float, b_coef: float, r2_lin: float) -> None:
    """Plot SE against 1/sqrt(N) with the fitted straight line"""
    plt.figure(figsize=(9, 5.8))
    plt.plot(x, y, marker="o", linewidth=2, label="SE")
    plt.plot(
        x,
        y_hat,
        linestyle="--",
        linewidth=2,
        label=f"Fit: SE={a_coef:.3f}x + {b_coef:.3f} (R2={r2_lin:.3f})",
    )
    plt.title("SE vs 1/sqrt(N)")
    plt.xlabel("1 / sqrt(N)")
    plt.ylabel("Standard Error")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_loglog_fit(log_n: np.ndarray, log_se: np.ndarray, 
                     log_se_hat: np.ndarray, beta: float,
                     r2_log: float) -> None:
    """Plot log(SE) against log(N) with the fitted line"""
    plt.figure(figsize=(9, 5.8))
    plt.plot(log_n, log_se, marker="o", linestyle="", label="log(SE)")
    plt.plot(
        log_n,
        log_se_hat,
        linestyle="--",
        linewidth=2,
        label=f"Fit slope beta={beta:.3f} (R2={r2_log:.3f})",
    )
    plt.title("Log-log test: slope should be ≈ -0.5")
    plt.xlabel("log(N)")
    plt.ylabel("log(SE)")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_abs_error(x: np.ndarray, abs_err: np.ndarray) -> None:
    """Plot the absolute BS pricing error against 1/sqrt(N)"""
    plt.figure(figsize=(9, 5.8))
    plt.plot(x, abs_err, marker="o", linewidth=2, label="|MC - BS| (fixed seed)")
    plt.title("Price error vs 1/sqrt(N) (BS reference)")
    plt.xlabel("1 / sqrt(N)")
    plt.ylabel("|MC - BS|")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_results(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
                  a_coef: float, b_coef: float, r2_lin: float,
                  log_n: np.ndarray, log_se: np.ndarray, log_se_hat: np.ndarray,
                  beta: float, r2_log: float,
                  abs_err_vals: np.ndarray, bs_ref: float | None) -> None:
    """Display the figures used in the convergence analysis."""
    # Plot the linear relationship in 1/sqrt(N)
    _plot_linear_fit(x, y, y_hat, a_coef, b_coef, r2_lin)

    # Plot the log-log relationship
    _plot_loglog_fit(log_n, log_se, log_se_hat, beta, r2_log)

    # Plot BS pricing error only when a BS reference exists
    if bs_ref is not None and len(abs_err_vals) > 0:
        _plot_abs_error(x, abs_err_vals)


def test_se_linear_in_1_over_sqrtN(market: Market, trade: OptionTrade, N_list: list[int],
                                   n_steps: int, seed: int = 42, antithetic: bool = True) -> dict[str, Any]:
    """
    Test the theoretical scaling SE - 1/sqrt(N) for Monte Carlo pricing
    """
    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    # Prepare the tested path counts.
    n_values = _prepare_n_values(N_list, antithetic)
    x = 1.0 / np.sqrt(n_values.astype(float))

    # Use BS only in the no-discrete-dividend case.
    bs_ref = _bs_reference_if_valid(market, trade)

    # Run the MC experiments for all N values.
    y, abs_err_vals = _collect_statistics(
        market,
        trade,
        n_values,
        n_steps,
        seed,
        antithetic,
        bs_ref,
    )

    # Fit the linear model SE = a*(1/sqrt(N)) + b.
    a_coef, b_coef, y_hat, r2_lin = _linear_fit(x, y)

    # Fit the log-log model log(SE) = alpha + beta*log(N).
    alpha, beta, log_n, log_se_hat, r2_log = _loglog_fit(n_values, y)
    log_se = np.log(y)

    # Print the summary table.
    _print_summary(antithetic, seed, n_steps,
                   a_coef, b_coef, r2_lin,
                   alpha, beta, r2_log
    )

    # Display plots
    _plot_results(x, y, y_hat,
                  a_coef, b_coef, r2_lin,
                  log_n, log_se, log_se_hat,
                  beta, r2_log, abs_err_vals, bs_ref
    )

    return {
        "N": n_values.astype(int),
        "x_1_over_sqrtN": x,
        "se": y,
        "lin_fit": {
            "a": float(a_coef),
            "b": float(b_coef),
            "r2": float(r2_lin),
        },
        "loglog_fit": {
            "alpha": float(alpha),
            "beta": float(beta),
            "r2": float(r2_log),
        },
        "abs_err_vs_bs": abs_err_vals if bs_ref is not None else None,
    }


def _build_market() -> Market:
    """Create the market object used in the example"""
    return Market(S0=100.0, r=0.10, sigma=0.20)


def _build_trade() -> OptionTrade:
    """Create the European option used in the example"""
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    return OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=3.0,
    )


def main() -> None:
    """Run the SE scaling tests"""
    market = _build_market()
    trade = _build_trade()
    n_list = [500, 1000, 2000, 5000, 10000, 12000, 15000, 20000, 50000, 100000]

    test_se_linear_in_1_over_sqrtN(
        market,
        trade,
        N_list=n_list,
        n_steps=100,
        seed=1,
        antithetic=True
    )

if __name__ == "__main__":
    main()