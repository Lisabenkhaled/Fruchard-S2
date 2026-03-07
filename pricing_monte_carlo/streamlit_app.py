import datetime as dt
from pathlib import Path
import time
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
    method: str = "vector",
    american_algo: str = "ls",
    basis: str = "laguerre",
    degree: int = 2,
) -> CorePricingParams:
    return CorePricingParams(
        n_paths=normalize_n_paths(int(n_paths), bool(antithetic)),
        n_steps=int(n_steps),
        seed=int(seed),
        antithetic=bool(antithetic),
        method=method,
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

    out = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=500,
        optimize=False,
        threshold=0.0,
        return_tree=False,
    )
    return float(out["tree_price"]), "Arbre trinomial"


def one_run(market: Market, trade: OptionTrade, p: CorePricingParams) -> dict:
    price, std, se, elapsed = core_price(market, trade, p)
    return {
        "Méthode": "Vectorielle" if p.method == "vector" else "Scalaire",
        "Antithétique": p.antithetic,
        "Nombre de chemins": p.n_paths,
        "Prix": price,
        "Standard deviation": std,
        "Standard error": se,
        "IC95 bas": price - 1.96 * se,
        "IC95 haut": price + 1.96 * se,
        "Temps (s)": elapsed,
    }


def convergence_grid(min_n: int, max_n: int, n_points: int) -> list[int]:
    min_safe = max(int(min_n), 100)
    max_safe = max(int(max_n), min_safe + 1)
    pts = max(int(n_points), 2)
    grid = np.unique(np.round(np.logspace(np.log10(min_safe), np.log10(max_safe), pts)).astype(int))
    return list(map(int, grid))


def run_convergence(
    market: Market,
    trade: OptionTrade,
    *,
    grid: list[int],
    method: str,
    antithetic: bool,
    algo: str,
    basis: str,
    degree: int,
    n_steps: int,
    seed: int,
) -> pd.DataFrame:
    ref, ref_name = benchmark_price(market, trade)

    rows = []
    
    for n in grid:
        p = build_params(n, n_steps, seed, antithetic, method, algo, basis, degree)
        out = one_run(market, trade, p)
        rows.append(
            {
                "n_paths": int(n),
                "Prix": out["Prix"],
                "Standard deviation": out["Standard deviation"],
                "Standard error": out["Standard error"],
                "Erreur benchmark": abs(out["Prix"] - ref),
                "SE * sqrt(N)": out["Standard error"] * math.sqrt(int(n)),
                "Temps (s)": out["Temps (s)"],
                "Benchmark": ref,
                "Nom benchmark": ref_name,
            }
        )
    return pd.DataFrame(rows)

    
def discounted_option_profile(market: Market, trade: OptionTrade, p: CorePricingParams) -> pd.DataFrame:
    if p.method == "vector":
        times, paths = simulate_gbm_paths_vector(
            market, trade, p.n_paths, p.n_steps, seed=p.seed, antithetic=p.antithetic)
        
    else:
        times, paths = simulate_gbm_paths_scalar(
            market, trade, p.n_paths, p.n_steps, seed=p.seed, antithetic=p.antithetic)
        
    intrinsic = trade.payoff_vector(paths)
    discounts = np.exp(-float(market.r) * np.asarray(times, dtype=float))
    discounted = intrinsic * discounts[None, :]

    return pd.DataFrame(
        {
            "Pas": np.arange(len(times)),
            "Temps (années)": np.asarray(times, dtype=float),
            "Moyenne actualisée": discounted.mean(axis=0),
            "Standard deviation": discounted.std(axis=0, ddof=1),
        }
    )


def plot_metrics(df: pd.DataFrame, *, index_col: str, choices: list[str], color_col: str | None = None) -> None:
    if not choices:
        st.info("Sélectionne au moins une métrique à afficher.")
        return
    for metric in choices:
        st.subheader(f"Graphe — {metric}")
        if color_col:
            plot_df = df.pivot(index=index_col, columns=color_col, values=metric)
            st.line_chart(plot_df)
        else:
            st.line_chart(df.set_index(index_col)[[metric]])



st.set_page_config(page_title="Dashboard Monte Carlo", layout="wide")
st.title("Dashboard Monte Carlo")
st.caption("Chaque onglet a ses propres paramètres. Les graphes sont au choix.")

with st.sidebar:
    st.header("Marché")
    s0 = st.number_input("Spot S0", min_value=0.01, value=100.0, step=1.0)
    r = st.number_input("Taux r", value=0.04, step=0.005, format="%.4f")
    sigma = st.number_input("Volatilité sigma", min_value=0.0001, value=0.25, step=0.01, format="%.4f")

    st.header("Option")
    is_call = st.toggle("Call (sinon Put)", value=False)
    strike = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    pricing_date = st.date_input("Date de pricing", value=dt.date(2026, 2, 26))
    maturity_date = st.date_input("Maturité", value=dt.date(2027, 4, 26))
    q = st.number_input("Dividende continu q", value=0.0, step=0.005, format="%.4f")
    has_div = st.toggle("Dividende discret", value=True)
    ex_div_date = st.date_input("Ex-div date", value=dt.date(2026, 6, 21), disabled=not has_div)
    div_amount = st.number_input("Montant dividende", min_value=0.0, value=3.0, step=0.5, disabled=not has_div)
    
if maturity_date <= pricing_date:
    st.error("La maturité doit être > date de pricing.")
    st.stop()

market = Market(S0=float(s0), r=float(r), sigma=float(sigma))

(
    tab_price,
    tab_greeks,
    tab_conv_eu,
    tab_conv_am,
    tab_conv_delta,
    tab_perf,
    tab_degree,
    tab_profile,
) = st.tabs(
    [
        "Prix principal",
        "Greeks",
        "Convergence EU",
        "Convergence AM",
        "Convergence Delta",
        "Comparaison performance",
        "Test degré LS",
        "Valeur actualisée",
    ]
)

with tab_price:
    st.subheader("Prix principal")
    c1, c2, c3, c4 = st.columns(4)
    exercise_price = c1.selectbox("Type d'exercice", ["european", "american"], index=1)
    method_price = c2.selectbox("Simulation", ["vector", "scalar"], index=0)
    algo_price = c3.selectbox("Algorithme américain", ["ls", "naive"], index=0)
    anti_price = c4.toggle("Antithétique", value=True)

    d1, d2, d3, d4 = st.columns(4)
    n_paths_price = d1.number_input("Nombre de chemins", min_value=100, value=40000, step=1000)
    n_steps_price = d2.number_input("Nombre de pas", min_value=5, value=120, step=5)
    seed_price = d3.number_input("Seed", min_value=0, value=127, step=1)
    basis_price = d4.selectbox("Base LS", ["laguerre", "power"])
    degree_price = st.slider("Degré LS", min_value=1, max_value=8, value=2)

    if st.button("Lancer le pricing", type="primary"):
        trade = build_trade(
            exercise=exercise_price,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        params = build_params(
            int(n_paths_price),
            int(n_steps_price),
            int(seed_price),
            bool(anti_price),
            method_price,
            algo_price,
            basis_price,
            int(degree_price),
        )
        row = one_run(market, trade, params)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix", f"{row['Prix']:.6f}")
        m2.metric("Standard deviation", f"{row['Standard deviation']:.6f}")
        m3.metric("Standard error", f"{row['Standard error']:.6f}")
        m4.metric("Temps", f"{row['Temps (s)']:.3f}s")
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

with tab_greeks:
    st.subheader("Greeks")
    g1, g2, g3 = st.columns(3)
    exercise_greeks = g1.selectbox("Type d'exercice", ["european", "american"], index=0, key="g_ex")
    algo_greeks = g2.selectbox("Algorithme américain", ["ls", "naive"], index=0, key="g_algo")
    anti_greeks = g3.toggle("Antithétique", value=True, key="g_anti")

    h1, h2, h3, h4 = st.columns(4)
    n_paths_greeks = h1.number_input("Nombre de chemins", min_value=1000, value=30000, step=2000, key="g_np")
    n_steps_greeks = h2.number_input("Nombre de pas", min_value=10, value=100, step=10, key="g_ns")
    seed_greeks = h3.number_input("Seed", min_value=0, value=127, step=1, key="g_seed")
    tree_n = h4.number_input("N benchmark arbre", min_value=50, value=250, step=50, key="g_tree")

    e1, e2 = st.columns(2)
    eps_spot = e1.number_input("Epsilon spot (ΔS)", min_value=0.0001, value=0.5, step=0.1, format="%.4f")
    eps_vol = e2.number_input("Epsilon vol (Δσ)", min_value=0.0001, value=0.01, step=0.005, format="%.4f")

    if st.button("Lancer les greeks", type="primary"):
        trade = build_trade(
            exercise=exercise_greeks,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        params = build_params(
            int(n_paths_greeks),
            int(n_steps_greeks),
            int(seed_greeks),
            bool(anti_greeks),
            method="vector",
            american_algo=algo_greeks,
            basis="laguerre",
            degree=2,
        )
        t0 = time.time()
        mc_greeks, ref_greeks = core_greeks(
            market,
            trade,
            params,
            shift_spot=float(eps_spot),
            shift_vol=float(eps_vol),
            tree_N=int(tree_n),
        )
        elapsed = time.time() - t0

        keys = sorted(set(mc_greeks.keys()) | set(ref_greeks.keys()))
        rows = []
        for k in keys:
            mc_v = mc_greeks.get(k, np.nan)
            ref_v = ref_greeks.get(k, np.nan)
            rows.append({"Greek": k, "Monte Carlo": mc_v, "Benchmark": ref_v, "Écart": mc_v - ref_v})
        st.metric("Temps de calcul greeks", f"{elapsed:.1f}s")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_conv_eu:
    st.subheader("Convergence EU (EU vs benchmark EU)")
    c1, c2, c3, c4 = st.columns(4)
    min_paths = c1.number_input("Min paths", min_value=100, value=1000, step=100, key="ceu_min")
    max_paths = c2.number_input("Max paths", min_value=500, value=30000, step=500, key="ceu_max")
    n_points = c3.slider("Nombre de points", min_value=3, max_value=12, value=6, key="ceu_pts")
    n_steps = c4.number_input("Pas MC", min_value=5, value=100, step=5, key="ceu_steps")
    d1, d2, d3, d4 = st.columns(4)
    seed = d1.number_input("Seed", min_value=0, value=127, step=1, key="ceu_seed")
    method = d2.selectbox("Simulation", ["vector", "scalar"], index=0, key="ceu_method")
    anti = d3.toggle("Antithétique", value=True, key="ceu_anti")
    include_eu_ls = d4.toggle("Afficher EU LS aussi", value=False, key="ceu_ls")

    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Erreur benchmark", "SE * sqrt(N)", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="ceu_plot",
    )

    if st.button("Lancer convergence EU", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
        trade_eu = build_trade(
            exercise="european",
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        eu_naive = run_convergence(
            market,
            trade_eu,
            grid=grid,
            method=method,
            antithetic=anti,
            algo="naive",
            basis="laguerre",
            degree=2,
            n_steps=int(n_steps),
            seed=int(seed),
        )
        eu_naive["Test"] = "EU naive"

        parts = [eu_naive]
        if include_eu_ls:
            eu_ls = run_convergence(
                market,
                trade_eu,
                grid=grid,
                method=method,
                antithetic=anti,
                algo="ls",
                basis="laguerre",
                degree=2,
                n_steps=int(n_steps),
                seed=int(seed), 
            )
            eu_ls["Test"] = "EU LS"
            parts.append(eu_ls)

        eu_df = pd.concat(parts, ignore_index=True)
        st.caption(f"Benchmark EU: {eu_df['Nom benchmark'].iloc[0]} = {eu_df['Benchmark'].iloc[0]:.6f}")
        st.dataframe(eu_df, use_container_width=True)
        plot_metrics(eu_df, index_col="n_paths", choices=graph_metrics, color_col="Test")

with tab_conv_am:
    st.subheader("Convergence AM (AM vs benchmark AM)")
    c1, c2, c3, c4 = st.columns(4)
    min_paths = c1.number_input("Min paths", min_value=100, value=1000, step=100, key="cam_min")
    max_paths = c2.number_input("Max paths", min_value=500, value=30000, step=500, key="cam_max")
    n_points = c3.slider("Nombre de points", min_value=3, max_value=12, value=6, key="cam_pts")
    n_steps = c4.number_input("Pas MC", min_value=5, value=100, step=5, key="cam_steps")
    d1, d2, d3, d4 = st.columns(4)
    seed = d1.number_input("Seed", min_value=0, value=127, step=1, key="cam_seed")
    method = d2.selectbox("Simulation", ["vector", "scalar"], index=0, key="cam_method")
    anti = d3.toggle("Antithétique", value=True, key="cam_anti")
    basis = d4.selectbox("Base LS", ["laguerre", "power"], key="cam_basis")
    degree = st.slider("Degré LS", min_value=1, max_value=8, value=2, key="cam_degree")

    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Erreur benchmark", "SE * sqrt(N)", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="cam_plot",
    )

    if st.button("Lancer convergence AM", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
        trade_am = build_trade(
            exercise="american",
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        am_naive = run_convergence(
            market,
            trade_am,
            grid=grid,
            method=method,
            antithetic=anti,
            algo="naive",
            basis=basis,
            degree=int(degree),
            n_steps=int(n_steps),
            seed=int(seed),
        )
        am_naive["Test"] = "AM naive"

        am_ls = run_convergence(
            market,
            trade_am,
            grid=grid,
            method=method,
            antithetic=anti,
            algo="ls",
            basis=basis,
            degree=int(degree),
            n_steps=int(n_steps),
            seed=int(seed),
        )
        am_ls["Test"] = "AM LS"

        am_df = pd.concat([am_naive, am_ls], ignore_index=True)
        st.caption(f"Benchmark AM: {am_df['Nom benchmark'].iloc[0]} = {am_df['Benchmark'].iloc[0]:.6f}")
        st.dataframe(am_df, use_container_width=True)
        plot_metrics(am_df, index_col="n_paths", choices=graph_metrics, color_col="Test")

with tab_conv_delta:
    st.subheader("Convergence Delta")
    c1, c2, c3, c4 = st.columns(4)
    exercise = c1.selectbox("Type d'exercice", ["european", "american"], index=0, key="cd_ex")
    min_paths = c2.number_input("Min paths", min_value=100, value=1000, step=100, key="cd_min")
    max_paths = c3.number_input("Max paths", min_value=500, value=30000, step=500, key="cd_max")
    n_points = c4.slider("Nombre de points", min_value=3, max_value=10, value=6, key="cd_pts")

    d1, d2, d3, d4 = st.columns(4)
    n_steps = d1.number_input("Pas MC", min_value=10, value=100, step=10, key="cd_steps")
    seed = d2.number_input("Seed", min_value=0, value=127, step=1, key="cd_seed")
    anti = d3.toggle("Antithétique", value=True, key="cd_anti")
    tree_n = d4.number_input("N benchmark arbre", min_value=50, value=250, step=50, key="cd_tree")

    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Delta MC", "Delta benchmark", "Erreur absolue delta", "Temps (s)"],
        default=["Delta MC", "Delta benchmark", "Erreur absolue delta"],
        key="cd_plot",
    )

    if st.button("Lancer convergence delta", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
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

        rows = []
        for n in grid:
            params = build_params(
                int(n),
                int(n_steps),
                int(seed),
                bool(anti),
                method="vector",
                american_algo="ls",
                basis="laguerre",
                degree=2,
            )
            t0 = time.time()
            mc_g, ref_g = core_greeks(market, trade, params, tree_N=int(tree_n))
            elapsed = time.time() - t0
            delta_mc = mc_g.get("delta", np.nan)
            delta_ref = ref_g.get("delta", np.nan)
            rows.append(
                {
                    "n_paths": int(n),
                    "Delta MC": delta_mc,
                    "Delta benchmark": delta_ref,
                    "Erreur absolue delta": abs(delta_mc - delta_ref),
                    "Temps (s)": elapsed,
                }
            )

        delta_df = pd.DataFrame(rows)
        st.dataframe(delta_df, use_container_width=True)
        plot_metrics(delta_df, index_col="n_paths", choices=graph_metrics)

with tab_perf:
    st.subheader("Comparaison performance")
    p1, p2, p3, p4 = st.columns(4)
    exercise_perf = p1.selectbox("Type d'exercice", ["european", "american"], index=1, key="pf_ex")
    n_paths_perf = p2.number_input("Nombre de chemins", min_value=100, value=25000, step=1000, key="pf_np")
    n_steps_perf = p3.number_input("Pas MC", min_value=5, value=100, step=5, key="pf_ns")
    algo_perf = p4.selectbox("Algorithme américain", ["ls", "naive"], key="pf_algo")
    s1, s2 = st.columns(2)
    seed_start = s1.number_input("Seed min", min_value=0, value=120, step=1, key="pf_smin")
    seed_count = s2.slider("Nombre de seeds", min_value=2, max_value=25, value=6, key="pf_sc")

    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="pf_plot",
    )

    if st.button("Lancer comparaison performance", type="primary"):
        trade = build_trade(
            exercise=exercise_perf,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )

        seeds = [int(seed_start) + k for k in range(int(seed_count))]
        rows = []
        for method_perf in ["vector", "scalar"]:
            for anti_perf in [False, True]:
                for seed_perf in seeds:
                    p = build_params(
                        int(n_paths_perf),
                        int(n_steps_perf),
                        int(seed_perf),
                        bool(anti_perf),
                        method_perf,
                        algo_perf,
                        "laguerre",
                        2,
                    )
                    row = one_run(market, trade, p)
                    row["Seed"] = seed_perf
                    rows.append(row)

        perf_df = pd.DataFrame(rows)
        perf_df["Configuration"] = perf_df["Méthode"] + " | anti=" + perf_df["Antithétique"].astype(str)
        st.dataframe(perf_df, use_container_width=True)

        st.markdown("**Moyennes anti / non-anti (toutes méthodes confondues)**")
        anti_mean = (
            perf_df.groupby("Antithétique", as_index=False)[["Prix", "Standard deviation", "Standard error", "Temps (s)"]]
            .mean()
            .sort_values("Antithétique")
        )
        st.dataframe(anti_mean, use_container_width=True)

        st.markdown("**Moyennes scalaire / vectorielle (tous modes anti confondus)**")
        method_mean = (
            perf_df.groupby("Méthode", as_index=False)[["Prix", "Standard deviation", "Standard error", "Temps (s)"]]
            .mean()
        )
        st.dataframe(method_mean, use_container_width=True)

        plot_metrics(perf_df.groupby("Configuration", as_index=False).mean(numeric_only=True), index_col="Configuration", choices=graph_metrics)

with tab_degree:
    st.subheader("Test du degré de régression LS")
    q1, q2, q3, q4 = st.columns(4)
    n_paths_deg = q1.number_input("Nombre de chemins", min_value=100, value=25000, step=1000, key="d_np")
    n_steps_deg = q2.number_input("Pas MC", min_value=5, value=100, step=5, key="d_ns")
    seed_deg = q3.number_input("Seed", min_value=0, value=127, step=1, key="d_seed")
    anti_deg = q4.toggle("Antithétique", value=True, key="d_anti")

    degrees = st.slider("Intervalle de degrés", min_value=1, max_value=10, value=(1, 6), key="d_range")
    basis_choices = st.multiselect("Base(s) à comparer", ["laguerre", "power"], default=["laguerre", "power"])
    metric_deg = st.selectbox("Métrique à tracer", ["Prix", "Standard error", "Temps (s)"])

    if st.button("Lancer test de degré", type="primary"):
        if not basis_choices:
            st.warning("Sélectionne au moins une base de régression.")
        else:
            trade = build_trade(
                exercise="american",
                strike=float(strike),
                is_call=bool(is_call),
                pricing_date=pricing_date,
                maturity_date=maturity_date,
                q=float(q),
                ex_div_date=ex_div_date if has_div else None,
                div_amount=float(div_amount) if has_div else 0.0,
            )

            rows = []
            for basis in basis_choices:
                for degree in range(int(degrees[0]), int(degrees[1]) + 1):
                    p = build_params(
                        int(n_paths_deg),
                        int(n_steps_deg),
                        int(seed_deg),
                        bool(anti_deg),
                        method="vector",
                        american_algo="ls",
                        basis=basis,
                        degree=degree,
                    )
                    out = one_run(market, trade, p)
                    rows.append(
                        {
                            "Base": basis,
                            "Degré": degree,
                            "Prix": out["Prix"],
                            "Standard error": out["Standard error"],
                            "Standard deviation": out["Standard deviation"],
                            "Temps (s)": out["Temps (s)"],
                        }
                    )
            deg_df = pd.DataFrame(rows)
            st.dataframe(deg_df, use_container_width=True)
            st.line_chart(deg_df.pivot(index="Degré", columns="Base", values=metric_deg))

with tab_profile:
    st.subheader("Profil de valeur moyenne actualisée")
    v1, v2, v3, v4 = st.columns(4)
    exercise_prof = v1.selectbox("Type d'exercice", ["european", "american"], index=1, key="pr_ex")
    method_prof = v2.selectbox("Simulation", ["vector", "scalar"], key="pr_m")
    n_paths_prof = v3.number_input("Nombre de chemins", min_value=100, value=15000, step=1000, key="pr_np")
    n_steps_prof = v4.number_input("Pas MC", min_value=5, value=100, step=5, key="pr_ns")
    w1, w2, w3 = st.columns(3)
    seed_prof = w1.number_input("Seed", min_value=0, value=127, step=1, key="pr_seed")
    anti_prof = w2.toggle("Antithétique", value=True, key="pr_anti")
    algo_prof = w3.selectbox("Algo américain", ["ls", "naive"], key="pr_algo")

    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Moyenne actualisée", "Standard deviation"],
        default=["Moyenne actualisée", "Standard deviation"],
        key="pr_plot",
    )

    if st.button("Lancer profil actualisé", type="primary"):
        trade = build_trade(
            exercise=exercise_prof,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        params = build_params(
            int(n_paths_prof),
            int(n_steps_prof),
            int(seed_prof),
            bool(anti_prof),
            method_prof,
            algo_prof,
            "laguerre",
            2,
        )
        prof_df = discounted_option_profile(market, trade, params)
        st.dataframe(prof_df, use_container_width=True)
        plot_metrics(prof_df, index_col="Pas", choices=graph_metrics)