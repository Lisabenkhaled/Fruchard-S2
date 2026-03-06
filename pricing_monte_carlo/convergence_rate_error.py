import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector
from utils.utils_bs import bs_price


def _bs_ref_price(market: Market, trade: OptionTrade) -> float:
    """Black–Scholes reference price (European only)."""
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


def _mc_price(market: Market, trade: OptionTrade, n_paths: int, n_steps: int,
    seed: int, antithetic: bool) -> float:
    """One MC run (one seed) -> one price."""
    mc, _ = price_european_naive_mc_vector(
        market=market,
        trade=trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )
    return float(mc)


def _mean_abs_error_for_n(market: Market, trade: OptionTrade, ref_price: float, n_paths: int, n_steps: int,
    seeds: list[int], antithetic: bool) -> float:
    """Mean |MC - ref| across seeds for a fixed N."""
    errs: list[float] = []

    # Loop over seeds to average out randomness
    for sd in seeds:
        mc = _mc_price(
            market=market,
            trade=trade,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=sd,
            antithetic=antithetic,
        )
        errs.append(abs(mc - ref_price))

    # Return average absolute error across seeds
    return float(np.mean(errs))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit y ≈ a x + b."""
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    y_hat = a * x + b
    return float(a), float(b), y_hat


def _collect_xy(market: Market, trade: OptionTrade, n_list: list[int], n_steps: int,
    seeds: list[int], antithetic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Build x=1/sqrt(N) and y=mean|MC-BS| arrays."""
    ref = _bs_ref_price(market, trade)

    x_vals: list[float] = []
    y_vals: list[float] = []

    # For each N, compute the mean error across seeds
    for n_paths in n_list:
        mean_err = _mean_abs_error_for_n(
            market=market,
            trade=trade,
            ref_price=ref,
            n_paths=n_paths,
            n_steps=n_steps,
            seeds=seeds,
            antithetic=antithetic,
        )

        # Theory suggests error scales like 1/sqrt(N)
        x_vals.append(1.0 / np.sqrt(float(n_paths)))
        y_vals.append(mean_err)

        # Print to quickly check monotonicity / noise
        print(f"N={n_paths:<7d}  mean|MC-BS|={mean_err:.6f}")

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def _plot_xy(x: np.ndarray, y: np.ndarray) -> None:
    """Plot points + fitted line."""
    a, b, y_hat = _linear_fit(x, y)

    # Points: observed mean absolute error
    plt.figure(figsize=(9, 5.8))
    plt.plot(x, y, marker="o", linewidth=2, label="Mean |MC - BS|")

    # Dashed line: best linear fit
    plt.plot(x, y_hat, linestyle="--", linewidth=2, label=f"Fit: y={a:.3f}x+{b:.3f}")
    plt.title("MC Convergence Rate")
    plt.xlabel("1 / sqrt(N)")
    plt.ylabel("Mean |MC - BS|")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def convergence_rate_plot(market: Market, trade: OptionTrade, n_list: list[int], n_steps: int,
    seeds: list[int], antithetic: bool = True) -> None:
    """Main function: compute points and plot them"""

    if trade.exercise.lower() != "european":
        raise ValueError("EUROPEAN options only.")

    # Compute x/y then plot
    x, y = _collect_xy(
        market=market,
        trade=trade,
        n_list=n_list,
        n_steps=n_steps,
        seeds=seeds,
        antithetic=antithetic,
    )
    _plot_xy(x, y)


def main() -> None:
    """Example usage"""
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

    n_list = [500, 1000, 2000, 5000, 10000, 15000, 20000]

    # Many seeds for stable mean error estimates
    seeds = list(range(1, 1001))

    convergence_rate_plot(
        market=market,
        trade=trade,
        n_list=n_list,
        n_steps=100,
        seeds=seeds,
        antithetic=True,
    )


if __name__ == "__main__":
    main()