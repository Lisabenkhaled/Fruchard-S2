import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from model.path_simulator import simulate_gbm_paths_vector
from model.regression import design_matrix
from model.regression import ols_fit_predict


def lsm_price_from_paths(market, trade, paths, n_steps, degree, basis):
    r = market.r
    dt = trade.T / n_steps
    df = np.exp(-r * dt)

    V = trade.payoff_vector(paths[:, -1])

    for t in range(n_steps - 1, 0, -1):
        St = paths[:, t]
        exercise = trade.payoff_vector(St)
        itm = exercise > 0.0

        if np.any(itm):
            Y = df * V[itm]

            if basis == "laguerre":
                X = design_matrix(
                    St[itm],
                    degree=degree,
                    basis="laguerre",
                    scale=trade.strike
                )
            else:
                X = design_matrix(
                    St[itm],
                    degree=degree,
                    basis="power"
                )

            cont = ols_fit_predict(X, Y, X)

            ex_now = exercise[itm] > cont
            V_itm = V[itm]
            V_itm = np.where(ex_now, exercise[itm], df * V_itm)
            V[itm] = V_itm

        V[~itm] = df * V[~itm]

    return float(df * V.mean())


def compare_bases_one_seed(
    market,
    trade,
    degrees,
    seed,
    n_paths,
    n_steps,
    antithetic=True,
    plot=True
):
    """
    Compare Power vs Laguerre basis for LSM for a SINGLE seed.
    Outputs direct prices (no averaging across seeds).
    Uses the SAME simulated paths for both bases.
    """
    if trade.exercise.lower() != "american":
        raise ValueError("This test is for American options (LSM).")

    # Simulate ONCE (same paths for both bases and all degrees)
    _, paths = simulate_gbm_paths_vector(
        market,
        trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic
    )

    results = {"power": [], "laguerre": []}

    for deg in degrees:
        results["power"].append(
            lsm_price_from_paths(market, trade, paths, n_steps, deg, "power")
        )
        results["laguerre"].append(
            lsm_price_from_paths(market, trade, paths, n_steps, deg, "laguerre")
        )

    results["power"] = np.array(results["power"], dtype=float)
    results["laguerre"] = np.array(results["laguerre"], dtype=float)

    print("\n============================================")
    print("LSM Regression Degree Comparison (1 seed)")
    print("============================================")
    print(f"seed={seed}, n_paths={n_paths}, n_steps={n_steps}, antithetic={antithetic}\n")
    print("Degree | Power Price | Laguerre Price | Diff (Lag - Pow)")
    print("---------------------------------------------------------")
    for i, deg in enumerate(degrees):
        diff = results["laguerre"][i] - results["power"][i]
        print(f"{deg:>6d} | {results['power'][i]:>10.6f} | {results['laguerre'][i]:>13.6f} | {diff:>13.6f}")

    if plot:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        ax.plot(degrees, results["power"], marker="o", linewidth=2, label="Power basis")
        ax.plot(degrees, results["laguerre"], marker="s", linewidth=2, label="Laguerre basis")

        ax.set_title(f"LSM Price vs Regression Degree (Single Seed = {seed})")
        ax.set_xlabel("Regression Degree")
        ax.set_ylabel("Option Price")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 26)

    market = Market(S0=100.0, r=0.10, sigma=0.20)

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0
    )

    degrees = [1, 2, 3, 4, 5, 6, 7]
    seed = 42

    compare_bases_one_seed(
        market=market,
        trade=trade,
        degrees=degrees,
        seed=seed,
        n_paths=100000,
        n_steps=100,
        antithetic=True,
        plot=True
    )