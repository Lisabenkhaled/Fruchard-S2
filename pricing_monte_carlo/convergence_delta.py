import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import datetime as dt
from dataclasses import replace
from typing import Callable, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price
from pricing_tree.greek_adaptateur import tree_greeks_from_mc


# =========================
# Pricing wrapper
# =========================
PriceFn = Callable[[Market, OptionTrade, CorePricingParams], float]

def price_from_core(market: Market, trade: OptionTrade, params: CorePricingParams) -> float:
    price, _, _, _ = core_price(market, trade, params)
    return float(price)


# =========================
# Fast delta FD (2 prices only)
# =========================
def delta_fd(price_fn: PriceFn, market: Market, trade: OptionTrade, params: CorePricingParams, eps_spot: float) -> float:
    m_up = Market(S0=market.S0 + eps_spot, r=market.r, sigma=market.sigma)
    m_dn = Market(S0=market.S0 - eps_spot, r=market.r, sigma=market.sigma)
    p_up = price_fn(m_up, trade, params)
    p_dn = price_fn(m_dn, trade, params)
    return (p_up - p_dn) / (2.0 * eps_spot)


# =========================
# BS delta reference (EU no dividend)
# =========================
def _norm_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta_ref(market: Market, trade: OptionTrade) -> float:
    if trade.exercise != "european":
        raise ValueError("BS ref only for european.")
    if getattr(trade, "div_amount", 0.0) not in (0.0, None):
        raise ValueError("BS ref assumes no dividend.")

    import math
    S = float(market.S0)
    K = float(trade.strike)
    r = float(market.r)
    sig = float(market.sigma)

    T = max((trade.maturity_date - trade.pricing_date).days / 365.0, 1e-12)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return _norm_cdf(d1) if trade.is_call else (_norm_cdf(d1) - 1.0)


# =========================
# Auto reference: BS if EU no div, else Tree
# =========================
def detect_reference_delta(market: Market, trade: OptionTrade, tree_N: int = 2000):
    if trade.exercise == "european" and getattr(trade, "div_amount", 0.0) in (0.0, None):
        return float(bs_delta_ref(market, trade)), "bs"

    tree = tree_greeks_from_mc(mc_market=market, mc_trade=trade, N=tree_N)
    # robust key (Delta/delta/DELTA)
    k = next((kk for kk in tree.keys() if str(kk).strip().lower() == "delta"), None)
    if k is None:
        raise KeyError(f"Tree has no delta key. Keys={list(tree.keys())}")
    return float(tree[k]), "tree"


# =========================
# Convergence study (meaningful metrics)
# =========================
def delta_convergence(
    price_fn: PriceFn,
    market: Market,
    trade: OptionTrade,
    base_params: CorePricingParams,
    n_paths_list=(500, 1_000, 2_000, 5_000, 10_000, 20_000),
    seeds=range(30),
    eps_spot: float | None = None,
    tree_N: int = 2000,
) -> Dict[str, Any]:

    # better default bump
    if eps_spot is None:
        eps_spot = 0.001 * market.S0  # 0.1%

    delta_ref, ref_type = detect_reference_delta(market, trade, tree_N=tree_N)

    rows = []
    for n_paths in n_paths_list:
        deltas = []

        for s in seeds:
            params = replace(base_params, n_paths=int(n_paths), seed=int(s))
            deltas.append(delta_fd(price_fn, market, trade, params, eps_spot))

        deltas = np.asarray(deltas, dtype=float)

        mean = float(deltas.mean())
        std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        se = float(std / np.sqrt(len(deltas))) if len(deltas) > 0 else np.nan

        # meaningful "error": RMSE across seeds vs reference
        rmse = float(np.sqrt(np.mean((deltas - delta_ref) ** 2)))

        rows.append({
            "n_paths": int(n_paths),
            "mean_delta": mean,
            "std_across_seeds": std,
            "se_of_mean": se,
            "delta_ref": delta_ref,
            "rmse_vs_ref": rmse,
            "inv_sqrt_n": 1.0 / np.sqrt(int(n_paths)),
        })

    return {"reference_type": ref_type, "delta_ref": delta_ref, "rows": rows}


# =========================
# Plots:
#   (1) mean delta + 95% CI + reference
#   (2) RMSE vs 1/sqrt(N) + fitted line
# =========================
def plot_results(study: Dict[str, Any], title_prefix: str = ""):
    rows = study["rows"]
    delta_ref = float(study["delta_ref"])

    N = np.array([r["n_paths"] for r in rows], dtype=float)
    meanD = np.array([r["mean_delta"] for r in rows], dtype=float)
    se = np.array([r["se_of_mean"] for r in rows], dtype=float)
    x = np.array([r["inv_sqrt_n"] for r in rows], dtype=float)
    rmse = np.array([r["rmse_vs_ref"] for r in rows], dtype=float)

    # Plot 1
    plt.figure()
    plt.plot(N, meanD, marker="o", label="Mean Δ across seeds")
    plt.fill_between(N, meanD - 1.96 * se, meanD + 1.96 * se, alpha=0.2, label="≈95% CI (mean)")
    plt.axhline(delta_ref, linestyle="--", label=f"Reference Δ = {delta_ref:.6f}")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("n_paths (log)")
    plt.ylabel("Delta")
    plt.title(f"{title_prefix} Mean Delta with CI")
    plt.legend()
    plt.show()

    # Plot 2: RMSE vs 1/sqrt(N) with fit through origin (RMSE ≈ c * 1/sqrt(N))
    c = float(np.dot(x, rmse) / np.dot(x, x))  # least squares slope through origin
    plt.figure()
    plt.plot(x, rmse, marker="o", label="RMSE vs reference")
    plt.plot(x, c * x, linestyle="--", label=f"Fit: RMSE ≈ {c:.4f}·(1/√N)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("1 / sqrt(n_paths)")
    plt.ylabel("RMSE(Δ) vs reference")
    plt.title(f"{title_prefix} RMSE scaling check")
    plt.legend()
    plt.show()


# =========================
# Example run
# =========================
if __name__ == "__main__":
    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    market = Market(S0=100, r=0.05, sigma=0.2)

    trade = OptionTrade(
        strike=100,
        is_call=True,
        exercise="european",     # -> TREE reference auto
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=0.0
    )

    base_params = CorePricingParams(
        n_paths=10_000,  # overwritten
        n_steps=300,
        seed=0,          # overwritten
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2
    )

    study = delta_convergence(
        price_fn=price_from_core,
        market=market,
        trade=trade,
        base_params=base_params,
        n_paths_list=(1_000, 2_000, 5_000, 10_000, 12_000, 15_000, 20_000, 50_000),
        seeds=range(30),
        eps_spot=0.001 * market.S0,     # default = 0.1% * S0
        tree_N=300
    )

    print(f"\nReference ({study['reference_type']}): delta_ref = {study['delta_ref']:.6f}\n")
    for r in study["rows"]:
        print(
            f"N={r['n_paths']:>6d} | meanΔ={r['mean_delta']:+.6f} "
            f"| std={r['std_across_seeds']:.6f} | se={r['se_of_mean']:.6f} "
            f"| RMSE={r['rmse_vs_ref']:.6f}"
        )

    plot_results(study, title_prefix=f"Ref={study['reference_type'].upper()}")