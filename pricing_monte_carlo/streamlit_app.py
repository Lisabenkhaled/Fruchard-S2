import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import math
import sys

# Ensure repository root is importable when app is launched from subfolders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pricing_monte_carlo.core_pricer import CorePricingParams, core_price
from pricing_monte_carlo.core_greeks import core_greeks
from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.model.path_simulator import (
    simulate_gbm_paths_scalar,
    simulate_gbm_paths_vector,
)
from pricing_monte_carlo.utils.utils_bs import bs_price
from pricing_tree.adaptateur import tree_price_from_mc

def normalize_n_paths(n_paths: int, antithetic: bool) -> int:
    n = max(int(n_paths), 1)
    if antithetic and n % 2 == 1:
        return n + 1
    return n  


def build_trade(
    exercise: str,
    strike: float,
    is_call: bool,
    pricing_date: dt.date,
    maturity_date: dt.date,
    q: float,
    ex_div_date: dt.date | None,
    div_amount: float,
) -> OptionTrade:
    return OptionTrade(
        strike=float(strike),
        is_call=bool(is_call),
        exercise=exercise,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=float(q),
        ex_div_date=ex_div_date,
        div_amount=float(div_amount),
    )


def build_params(
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool,
    method: str | None = None,
    american_algo: str = "ls",
    basis: str = "laguerre",
    degree: int = 2,
    pricing_method: str | None = None,
    **_: object,
) -> CorePricingParams:
    picked_method = pricing_method or method or "vector"
    return CorePricingParams(
        n_paths=normalize_n_paths(int(n_paths), bool(antithetic)),
        n_steps=int(n_steps),
        seed=int(seed),
        antithetic=bool(antithetic),
        method=picked_method,
        american_algo=american_algo,
        basis=basis,
        degree=int(degree),
        payoff="vanilla",
    )


def benchmark_price(market: Market, trade: OptionTrade) -> tuple[float, str]:
    no_discrete_div = (trade.ex_div_date is None) or (float(trade.div_amount) == 0.0)
    if trade.exercise.lower() == "european" and no_discrete_div and float(trade.q) == 0.0:
        return float(
            bs_price(
                S=float(market.S0),
                K=float(trade.strike),
                r=float(market.r),
                sigma=float(market.sigma),
                T=float(trade.T),
                is_call=bool(trade.is_call),
            )
        ), "Black-Scholes"

    out = tree_price_from_mc(mc_market=market, mc_trade=trade, N=500, optimize=False, threshold=0.0, return_tree=False)
    return float(out["tree_price"]), "Tree"


def one_run(market: Market, trade: OptionTrade, p: CorePricingParams) -> dict:
    price, std, se, elapsed = core_price(market, trade, p)
    return {
        "method": p.method,
        "antithetic": p.antithetic,
        "n_paths": p.n_paths,
        "price": price,
        "std": std,
        "se": se,
        "ci_low_95": price - 1.96 * se,
        "ci_high_95": price + 1.96 * se,
        "time_s": elapsed,
    }


def discounted_option_profile(market: Market, trade: OptionTrade, p: CorePricingParams) -> pd.DataFrame:
    if p.method == "vector":
        times, paths = simulate_gbm_paths_vector(market, trade, p.n_paths, p.n_steps, seed=p.seed, antithetic=p.antithetic)
    else:
        times, paths = simulate_gbm_paths_scalar(market, trade, p.n_paths, p.n_steps, seed=p.seed, antithetic=p.antithetic)

    intrinsic = trade.payoff_vector(paths)
    discounts = np.exp(-float(market.r) * np.asarray(times, dtype=float))
    discounted = intrinsic * discounts[None, :]

    return pd.DataFrame(
        {
            "step": np.arange(len(times)),
            "time_years": np.asarray(times, dtype=float),
            "mean_discounted_option_value": discounted.mean(axis=0),
            "std_discounted_option_value": discounted.std(axis=0, ddof=1),
        }
    )


def convergence_grid(min_n: int, max_n: int, n_points: int) -> list[int]:
    min_safe = max(int(min_n), 100)
    max_safe = max(int(max_n), min_safe + 1)
    pts = max(int(n_points), 2)
    grid = np.unique(np.round(np.logspace(np.log10(min_safe), np.log10(max_safe), pts)).astype(int))
    return list(map(int, grid))


def convergence_multi_tests(
    market: Market,
    trade: OptionTrade,
    test_cfg: dict[str, dict[str, int | bool | str]],
    basis: str,
    degree: int,
) -> pd.DataFrame:
    rows = []
    ref, ref_name = benchmark_price(market, trade)

    tests = [
        ("EU naive", "european", "naive"),
        ("AM naive", "american", "naive"),
        ("AM LS", "american", "ls"),
    ]

    for label, ex_style, algo in tests:
        cfg = test_cfg.get(label, {})
        grid = convergence_grid(
            int(cfg.get("min_paths", 1000)),
            int(cfg.get("max_paths", 30000)),
            int(cfg.get("n_points", 6)),
        )
        trade_local = OptionTrade(
            strike=float(trade.strike),
            is_call=bool(trade.is_call),
            exercise=ex_style,
            pricing_date=trade.pricing_date,
            maturity_date=trade.maturity_date,
            q=float(trade.q),
            ex_div_date=trade.ex_div_date,
            div_amount=float(trade.div_amount),
        )

        for n in grid:
            p_local = build_params(
                n_paths=int(n),
                n_steps=int(cfg.get("n_steps", 120)),
                seed=int(cfg.get("seed", 127)),
                antithetic=bool(cfg.get("antithetic", True)),
                method=str(cfg.get("method", "vector")),
                american_algo=algo,
                basis=basis,
                degree=int(degree),
            )
            out = one_run(market, trade_local, p_local)
            rows.append({
                "test": label,
                "benchmark": ref_name,
                "n_paths": int(n),
                "price": out["price"],
                "se": out["se"],
                "error_vs_ref": abs(out["price"] - ref),
                "time_s": out["time_s"],
            })

    return pd.DataFrame(rows)


st.set_page_config(page_title="Dashboard Pricing MC", layout="wide")
st.title("Dashboard Monte Carlo — Pricing, Greeks, Convergence")
st.caption("Interface de calcul et de comparaisons")

with st.sidebar:
    st.header("Marché")
    s0 = st.number_input("Spot S0", min_value=0.01, value=100.0, step=1.0)
    r = st.number_input("Taux r", value=0.04, step=0.005, format="%.4f")
    sigma = st.number_input("Volatilité sigma", min_value=0.0001, value=0.25, step=0.01, format="%.4f")

    st.header("Option")
    exercise = st.selectbox("Exercice", options=["european", "american"], index=1)
    is_call = st.toggle("Call (sinon Put)", value=False)
    strike = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    pricing_date = st.date_input("Date de pricing", value=dt.date(2026, 2, 26))
    maturity_date = st.date_input("Maturité", value=dt.date(2027, 4, 26))
    q = st.number_input("Dividende continu q", value=0.0, step=0.005, format="%.4f")
    has_div = st.toggle("Dividende discret", value=True)
    ex_div_date = st.date_input("Ex-div date", value=dt.date(2026, 6, 21), disabled=not has_div)
    div_amount = st.number_input("Montant dividende", min_value=0.0, value=3.0, step=0.5, disabled=not has_div)
    
    st.header("Réglages MC")
    n_paths = st.number_input("Nombre de chemins", min_value=100, value=30000, step=1000)
    n_steps = st.number_input("Nombre de pas", min_value=5, value=120, step=5)
    seed = st.number_input("Seed", min_value=0, value=127, step=1)
    
    method = st.selectbox("Méthode", options=["vector", "scalar"], index=0)
    antithetic = st.toggle("Antithetic", value=True)
    american_algo = st.selectbox("Algo américain", options=["ls", "naive"], index=0)
    basis = st.selectbox("Base LS", options=["laguerre", "power"], index=0)
    degree = st.slider("Degré LS", min_value=1, max_value=6, value=2)

    st.subheader("Greeks (epsilons)")
    greek_shift_spot = st.number_input("Epsilon Spot (ΔS)", min_value=0.0001, value=0.1, step=0.1, format="%.4f")
    greek_shift_vol = st.number_input("Epsilon Vol (Δσ)", min_value=0.0001, value=0.01, step=0.005, format="%.4f")
    greek_tree_n = st.number_input("Tree N (benchmark greeks)", min_value=50, value=400, step=50)

    st.subheader("Convergence (paramètres globaux)")
    conv_min_paths = st.number_input("Convergence min paths", min_value=100, value=1000, step=100)
    conv_max_paths = st.number_input("Convergence max paths", min_value=500, value=30000, step=500)
    conv_n_points = st.slider("Convergence nombre de points", min_value=3, max_value=12, value=6)
    conv_n_steps = st.number_input("Convergence pas MC", min_value=5, value=int(n_steps), step=5)
    conv_seed = st.number_input("Convergence seed", min_value=0, value=int(seed), step=1)
    conv_method = st.selectbox("Convergence méthode", options=["vector", "scalar"], index=0)
    conv_antithetic = st.toggle("Convergence antithetic", value=True)

    st.subheader("Convergence Orix (paramètres par test)")
    st.caption("Tu peux fixer des paramètres différents pour EU naive, AM naive et AM LS.")
    eu_min_paths = st.number_input("EU naive min paths", min_value=100, value=1000, step=100)
    eu_max_paths = st.number_input("EU naive max paths", min_value=500, value=20000, step=500)
    eu_points = st.slider("EU naive points", min_value=3, max_value=12, value=6)
    eu_steps = st.number_input("EU naive pas MC", min_value=5, value=int(n_steps), step=5)
    eu_seed = st.number_input("EU naive seed", min_value=0, value=int(seed), step=1)
    eu_method = st.selectbox("EU naive méthode", options=["vector", "scalar"], index=0)
    eu_anti = st.toggle("EU naive antithetic", value=True)

    amn_min_paths = st.number_input("AM naive min paths", min_value=100, value=1000, step=100)
    amn_max_paths = st.number_input("AM naive max paths", min_value=500, value=20000, step=500)
    amn_points = st.slider("AM naive points", min_value=3, max_value=12, value=6)
    amn_steps = st.number_input("AM naive pas MC", min_value=5, value=int(n_steps), step=5)
    amn_seed = st.number_input("AM naive seed", min_value=0, value=int(seed), step=1)
    amn_method = st.selectbox("AM naive méthode", options=["vector", "scalar"], index=0)
    amn_anti = st.toggle("AM naive antithetic", value=True)

    amls_min_paths = st.number_input("AM LS min paths", min_value=100, value=1000, step=100)
    amls_max_paths = st.number_input("AM LS max paths", min_value=500, value=20000, step=500)
    amls_points = st.slider("AM LS points", min_value=3, max_value=12, value=6)
    amls_steps = st.number_input("AM LS pas MC", min_value=5, value=int(n_steps), step=5)
    amls_seed = st.number_input("AM LS seed", min_value=0, value=int(seed), step=1)
    amls_method = st.selectbox("AM LS méthode", options=["vector", "scalar"], index=0)
    amls_anti = st.toggle("AM LS antithetic", value=True)

    st.header("Analyses à lancer")
    selected_panels = st.multiselect(
        "Tu peux lancer seulement ce qui t'intéresse",
        options=[
            "Prix principal",
            "Greeks (via core_greeks)",
            "Convergence Prix / SE / Erreur",
            "Convergence Delta",
            "Convergence Orix (EU naive / AM naive / AM LS)",
            "Convergence SE / Error (3 tests)",
            "Comparaisons performance",
            "Effet degré régression LS",
            "Valeur moyenne actualisée par date",
        ],
        default=["Prix principal"],
    )
    run_btn = st.button("Lancer", type="primary")
if maturity_date <= pricing_date:
    st.error("La maturité doit être > date de pricing.")
    st.stop()

market = Market(S0=float(s0), r=float(r), sigma=float(sigma))
trade = build_trade(
    exercise=exercise,
    strike=float(strike),
    is_call=bool(is_call),
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    q=float(q),
    ex_div_date=ex_div_date if has_div else None,
    div_amount=float(div_amount) if has_div else 0.0,
)
params = build_params(
    n_paths=int(n_paths),
    n_steps=int(n_steps),
    seed=int(seed),
    antithetic=bool(antithetic),
    method=method,
    american_algo=american_algo,
    basis=basis,
    degree=int(degree),
)

orix_test_cfg = {
    "EU naive": {
        "min_paths": int(eu_min_paths),
        "max_paths": int(eu_max_paths),
        "n_points": int(eu_points),
        "n_steps": int(eu_steps),
        "seed": int(eu_seed),
        "method": eu_method,
        "antithetic": bool(eu_anti),
    },
    "AM naive": {
        "min_paths": int(amn_min_paths),
        "max_paths": int(amn_max_paths),
        "n_points": int(amn_points),
        "n_steps": int(amn_steps),
        "seed": int(amn_seed),
        "method": amn_method,
        "antithetic": bool(amn_anti),
    },
    "AM LS": {
        "min_paths": int(amls_min_paths),
        "max_paths": int(amls_max_paths),
        "n_points": int(amls_points),
        "n_steps": int(amls_steps),
        "seed": int(amls_seed),
        "method": amls_method,
        "antithetic": bool(amls_anti),
    },
}

if not selected_panels:
    st.warning("Sélectionne au moins une analyse dans la barre latérale.")
    st.stop()

panels = st.tabs(selected_panels)
tab_map = {name: tab for name, tab in zip(selected_panels, panels)}

if run_btn:
    if "Prix principal" in tab_map:
        with tab_map["Prix principal"]:
            row = one_run(market, trade, params)
            c1, c2, c3, c4 = st.columns(4)
            
            
            c1.metric("Prix", f"{row['price']:.6f}")
            c2.metric("Std", f"{row['std']:.6f}")
            c3.metric("SE", f"{row['se']:.6f}")
            c4.metric("Temps", f"{row['time_s']:.3f}s")
            st.dataframe(pd.DataFrame([row]), width='stretch')

    if "Greeks" in tab_map:
        with tab_map["Greeks (via core_greeks)"]:
            if params.method != "vector":
                st.warning("Les greeks MC du module central sont vectoriels: calcul forcé en mode vector.")
            greek_params = build_params(
                int(n_paths), int(n_steps), int(seed), bool(antithetic), "vector", american_algo, basis, int(degree)
            )
            mc_greeks, ref_greeks = core_greeks(
                market,
                trade,
                greek_params,
                shift_spot=float(greek_shift_spot),
                shift_vol=float(greek_shift_vol),
                tree_N=int(greek_tree_n),
            )
            grec_rows = []
            keys = sorted(set(mc_greeks.keys()) | set(ref_greeks.keys()))
            for k in keys:
                mc_v = mc_greeks.get(k, np.nan)
                ref_v = ref_greeks.get(k, np.nan)
                grec_rows.append({"greek": k, "mc": mc_v, "ref": ref_v, "diff": mc_v - ref_v})
            grec_df = pd.DataFrame(grec_rows)
            st.dataframe(grec_df, use_container_width=True)
            st.caption("Epsilons utilisés: ΔS = {:.4f}, Δσ = {:.4f}. (rho/theta suivent les réglages internes du module greeks.)".format(float(greek_shift_spot), float(greek_shift_vol)))

    if "Convergence Prix / SE / Erreur" in tab_map:
        with tab_map["Convergence Prix / SE / Erreur"]:
            ref, ref_name = benchmark_price(market, trade)
            rows = []
            for n in convergence_grid(int(conv_min_paths), int(conv_max_paths), int(conv_n_points)):
                pconv = build_params(n, int(n_steps), int(seed), bool(antithetic), method, american_algo, basis, int(degree))
                rconv = one_run(market, trade, pconv)
                rows.append({
                    "n_paths": n,
                    "price": rconv["price"],
                    "se": rconv["se"],
                    "error_vs_ref": abs(rconv["price"] - ref),
                    "se_times_sqrt_n": rconv["se"] * math.sqrt(n),
                    "time_s": rconv["time_s"],
                })
            conv_df = pd.DataFrame(rows)
            st.caption(f"Benchmark utilisé: {ref_name} = {ref:.6f}")
            st.dataframe(conv_df, width='stretch')
            c1, c2, c3, c4 = st.columns(4)
            c1.line_chart(conv_df.set_index("n_paths")["price"])
            c2.line_chart(conv_df.set_index("n_paths")[["se", "error_vs_ref"]])
            c3.line_chart(conv_df.set_index("n_paths")["se_times_sqrt_n"])
            c4.line_chart(conv_df.set_index("n_paths")["time_s"])

    if "Convergence Delta" in tab_map:
        with tab_map["Convergence Delta"]:
            if method != "vector":
                st.warning("Convergence delta calculée avec méthode vector (module greeks central).")
            delta_rows = []
            for n in convergence_grid(int(n_paths)):
                pdelta = build_params(n, int(n_steps), int(seed), bool(antithetic), "vector", american_algo, basis, int(degree))
                mc_greeks, ref_greeks = core_greeks(market, trade, pdelta, tree_N=300)
                delta_rows.append(
                    {
                        "n_paths": n,
                        "delta_mc": mc_greeks.get("delta", np.nan),
                        "delta_ref": ref_greeks.get("delta", np.nan),
                        "abs_error_delta": abs(mc_greeks.get("delta", np.nan) - ref_greeks.get("delta", np.nan)),
                    }
                )
            ddf = pd.DataFrame(delta_rows)
            st.dataframe(ddf, width='stretch')
            st.line_chart(ddf.set_index("n_paths")[["delta_mc", "delta_ref", "abs_error_delta"]])
    
    
    if "Convergence Orix (EU naive / AM naive / AM LS)" in tab_map:
        with tab_map["Convergence Orix (EU naive / AM naive / AM LS)"]:
            orix_df = convergence_multi_tests(
                market=market,
                trade=trade,
                test_cfg=orix_test_cfg,
                basis=basis,
                degree=int(degree),
            )
            st.dataframe(orix_df, use_container_width=True)
            st.line_chart(orix_df.pivot(index="n_paths", columns="test", values="price"))
            st.line_chart(orix_df.pivot(index="n_paths", columns="test", values="time_s"))

    if "Convergence SE / Error (3 tests)" in tab_map:
        with tab_map["Convergence SE / Error (3 tests)"]:
            se_df = convergence_multi_tests(
                market=market,
                trade=trade,
                test_cfg=orix_test_cfg,
                basis=basis,
                degree=int(degree),
            )
            st.dataframe(se_df[["test", "n_paths", "se", "error_vs_ref", "time_s"]], use_container_width=True)
            c1, c2 = st.columns(2)
            c1.line_chart(se_df.pivot(index="n_paths", columns="test", values="se"))
            c2.line_chart(se_df.pivot(index="n_paths", columns="test", values="error_vs_ref"))

    if "Comparaisons performance" in tab_map:
        with tab_map["Comparaisons performance"]:
            combos = [("vector", False), ("vector", True), ("scalar", False), ("scalar", True)]
            comp_rows = []
            for m, anti in combos:
                pp = build_params(int(n_paths), int(n_steps), int(seed), anti, m, american_algo, basis, int(degree))
                comp_rows.append(one_run(market, trade, pp))
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True)

            st.markdown("**Moyennes anti vs non anti**")
            st.dataframe(comp_df.groupby("antithetic", as_index=False)[["price", "std", "se", "time_s"]].mean(), use_container_width=True)

            st.markdown("**Moyennes scalar vs vector**")
            st.dataframe(comp_df.groupby("method", as_index=False)[["price", "std", "se", "time_s"]].mean(), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.bar_chart(comp_df.set_index(comp_df["method"] + "|anti=" + comp_df["antithetic"].astype(str))["time_s"])
            c2.bar_chart(comp_df.set_index(comp_df["method"] + "|anti=" + comp_df["antithetic"].astype(str))["se"])

    if "Effet degré régression LS" in tab_map:
        with tab_map["Effet degré régression LS"]:
            if exercise != "american" or american_algo != "ls":
                st.info("Cette analyse est pertinente pour American + LS.")
            degree_rows = []
            for d in range(1, 7):
                pdg = build_params(int(n_paths), int(n_steps), int(seed), bool(antithetic), method, "ls", basis, d)
                res = one_run(market, trade, pdg)
                degree_rows.append({"degree": d, "price": res["price"], "se": res["se"], "time_s": res["time_s"]})
            deg_df = pd.DataFrame(degree_rows)
            st.dataframe(deg_df, use_container_width=True)
            st.line_chart(deg_df.set_index("degree")[["price", "se", "time_s"]])

    if "Valeur moyenne actualisée par date" in tab_map:
        with tab_map["Valeur moyenne actualisée par date"]:
            prof_df = discounted_option_profile(market, trade, params)
            st.dataframe(prof_df, use_container_width=True)
            st.line_chart(prof_df.set_index("step")[["mean_discounted_option_value", "std_discounted_option_value"]])
else:
    st.info("Configure les paramètres puis clique sur Lancer.")