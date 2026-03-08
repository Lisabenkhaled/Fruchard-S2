import datetime as dt
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import replace
import math
import sys
from typing import Any, Dict, List, Optional, Literal
# Ensure repository root is importable when app is launched from subfolders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# imports from project
from pricing_monte_carlo.core_pricer import (
    AmericanAlgo,
    Basis,
    CorePricingParams,
    Method,
    core_price,
)
from pricing_monte_carlo.core_greeks import core_greeks
from pricing_monte_carlo.model.market import Market
from pricing_monte_carlo.convergence_rate_se import _build_trade as _build_trade_from_convergence_rate_se
from pricing_monte_carlo.convergence_rate_se import _prepare_n_values
from pricing_monte_carlo.model.option import OptionTrade
from pricing_monte_carlo.model.path_simulator import (
    simulate_gbm_paths_scalar,
    simulate_gbm_paths_vector,
)
from pricing_monte_carlo.utils.utils_bs import bs_price
from pricing_tree.adaptateur import tree_price_from_mc
# Display options
DisplayMode = Literal["Prix direct", "Écart à une référence", 
                      "Écart au premier point", "Écart à la moyenne"]
def build_trade(
    exercise: str,
    strike: float,
    is_call: bool,
    pricing_date: dt.date,
    maturity_date: dt.date,
    q: float,
    ex_div_date: Optional[dt.date],
    div_amount: float,
) -> OptionTrade:
    # Reuse the trade-construction entrypoint from convergence_rate_se, then override with UI inputs.
    template_trade = _build_trade_from_convergence_rate_se()
    return replace(
        template_trade,
        strike=float(strike),
        is_call=bool(is_call),
        exercise=exercise,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=float(q),
        ex_div_date=ex_div_date,
        div_amount=float(div_amount),
    )
# pricing parameters
def build_core_pricing_params(
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool,
    method: Method = "vector",
    american_algo: AmericanAlgo = "ls",
    basis: Basis = "laguerre",
    degree: int = 2,
) -> CorePricingParams:
    # normalize paths 
    normalized_n_paths = int(_prepare_n_values([max(int(n_paths), 1)], bool(antithetic))[0])
    return CorePricingParams(
        n_paths=normalized_n_paths,
        n_steps=int(n_steps),
        seed=int(seed),
        antithetic=bool(antithetic),
        method=method,
        american_algo=american_algo,
        basis=basis,
        degree=int(degree),
        payoff="vanilla",
    )
# convergence
def convergence_grid(min_n: int, max_n: int, n_points: int) -> List[int]:
    min_safe = max(int(min_n), 100)
    max_safe = max(int(max_n), min_safe + 1)
    pts = max(int(n_points), 2)
    grid = np.unique(np.round(np.logspace(np.log10(min_safe), np.log10(max_safe), pts)).astype(int))
    return list(map(int, grid))
# Benchmarks : tree and BS
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
    # tree
    out = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=500,
        optimize=False,
        threshold=0.0,
        return_tree=False,
    )
    return float(out["tree_price"]), "Arbre trinomial"
# for price in UI
def one_run(market: Market, trade: OptionTrade, p: CorePricingParams) -> Dict[str, Any]:
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
def _convergence_row(n: int, out: dict, ref: float, ref_name: str) -> dict:
    return {
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
# function to run convergence
def run_convergence(
    market: Market,
    trade: OptionTrade,
    *,
    grid: List[int],
    method: Method,
    antithetic: bool,
    algo: AmericanAlgo,
    basis: Basis,
    degree: int,
    n_steps: int,
    seed: int,
) -> pd.DataFrame:
    # convergence principal boucle: stock price/std/se/time for each N
    ref, ref_name = benchmark_price(market, trade)
    rows = []    
    for n in grid:
        p = build_core_pricing_params(n, n_steps, seed, antithetic, method, algo, basis, degree)
        out = one_run(market, trade, p)
        rows.append(_convergence_row(int(n), out, float(ref), str(ref_name)))
    return pd.DataFrame(rows)
# Paths
def _paths_row(n_paths: int, out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_paths": int(n_paths),
        "Prix": out["Prix"],
        "Standard error": out["Standard error"],
        "Temps (s)": out["Temps (s)"],
    }
# for the analysis with number of paths
def run_paths_analysis(
    market: Market,
    trade: OptionTrade,
    *,
    paths_grid: List[int],
    n_steps: int,
    seed: int,
    algo: AmericanAlgo,
    basis: Basis,
    degree: int,
    method: Method,
    antithetic: bool,
) -> pd.DataFrame:
    # Sensibility analysis function of the number of paths
    rows = []
    for n_paths in paths_grid:
        p = build_core_pricing_params(
            n_paths=int(n_paths),
            n_steps=int(n_steps),
            seed=int(seed),
            antithetic=bool(antithetic),
            method=method,
            american_algo=algo,
            basis=basis,
            degree=int(degree),
        )
        out = one_run(market, trade, p)
        rows.append(_paths_row(int(n_paths), out))
    return pd.DataFrame(rows)
def add_inv_sqrt_n_fit(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # speed of convergence: metric in function of 1/sqrt(N), + droite fit.
    x = 1.0 / np.sqrt(pd.to_numeric(df["n_paths"], errors="coerce").to_numpy(dtype=float))
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return pd.DataFrame({"1/sqrt(N)": x, metric: y, f"Fit {metric}": y})
    slope, intercept = np.polyfit(x, y, 1)
    fit_vals = slope * x + intercept
    out = pd.DataFrame({"1/sqrt(N)": x, metric: y, f"Fit {metric}": fit_vals})
    return out.sort_values("1/sqrt(N)").reset_index(drop=True)
def _tree_reference(market: Market, trade: OptionTrade, tree_n: int) -> float: 
    out_tree = tree_price_from_mc(
        mc_market=market,
        mc_trade=trade,
        N=int(tree_n),
        optimize=False,
        threshold=0.0,
        return_tree=False,
    )
    return float(out_tree["tree_price"])
def _equivalent_rate(market: Market, trade: OptionTrade) -> tuple[float, float]:
    # Convert dividends in equivalent BS  
    T = max(float(trade.T), 1e-10)
    q_cont = float(trade.q)
    t_div = trade.ex_div_time()
    pv_div = 0.0
    if t_div is not None and 0.0 < float(t_div) <= T and float(trade.div_amount) > 0.0:
        pv_div = float(trade.div_amount) * math.exp(-float(market.r) * float(t_div))
    q_equiv = q_cont + pv_div / max(float(market.S0) * T, 1e-10)
    return float(market.r) - q_equiv, T
# convergence references
def compute_reference_lines(
    market: Market,
    trade: OptionTrade,
    *,
    tree_n: int = 500,
) -> Dict[str, float]:
    """References de convergence: arbre + Black-Scholes taux équivalent."""
    tree_price = _tree_reference(market, trade, int(tree_n))
    r_equiv, T = _equivalent_rate(market, trade)
# return BS, tree and equivalent rate
    bs_price_level = float(
        bs_price(
            S=float(market.S0),
            K=float(trade.strike),
            r=r_equiv,
            sigma=float(market.sigma),
            T=T,   
            is_call=bool(trade.is_call)))
    return {
        "Arbre trinomial": tree_price,
        "Black-Scholes": bs_price_level,
        "Taux équivalent BS": r_equiv,
    }
# simulate paths 
def _simulate_paths(market: Market, trade: OptionTrade, p: CorePricingParams) -> tuple[np.ndarray, np.ndarray]:
    if p.method == "vector":
        return simulate_gbm_paths_vector(market, trade, p.n_paths, p.n_steps, 
                                         seed=p.seed, antithetic=p.antithetic)
    return simulate_gbm_paths_scalar(market, trade, p.n_paths, p.n_steps, seed=p.seed, 
                                     antithetic=p.antithetic)
# discounted option
def discounted_option_profile(market: Market, trade: OptionTrade, p: CorePricingParams) -> pd.DataFrame:
    times, paths = _simulate_paths(market, trade, p)  
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
# display mode with multiple choices
def _apply_display_mode(
    df: pd.DataFrame,
    series_cols: List[str],
    display_mode: DisplayMode,
    reference_name: Optional[str],
) -> pd.DataFrame:
    out = pd.DataFrame(df)
# Option 1: ecart with a reference (to choose)
    if display_mode == "Écart à une référence":
        ref_col = reference_name if reference_name in series_cols else series_cols[0]
        ref_vals = pd.to_numeric(out[ref_col], errors="coerce")
        for col in series_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce") - ref_vals
        st.caption(f"Affichage: (Série - {ref_col})")
# Option 2: ecart with first point 
    elif display_mode == "Écart au premier point":
        for col in series_cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            first_valid = vals.dropna().iloc[0] if vals.notna().any() else 0.0
            out[col] = vals - float(first_valid)
        st.caption("Affichage: écart au premier point")
# Option 3: ecart with mean
    elif display_mode == "Écart à la moyenne":
        for col in series_cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals - float(vals.mean())
        st.caption("Affichage: écart à la moyenne")
    return out
# return x and y 
def _vega_line_spec(index_col: str, title: str, y_min: float, y_max: float) -> dict:
    return {   
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": index_col, "type": "quantitative", "title": index_col},
            "y": {"field": "Valeur", "type": "quantitative", "title": "Valeur", "scale": {"domain": [y_min, y_max]}},
            "color": {"field": "Série", "type": "nominal"},
            "tooltip": [{"field": index_col, "type": "quantitative"}, {"field": "Série", "type": "nominal"}, 
                        {"field": "Valeur", "type": "quantitative", "format": ".8f"}],
        },
        "height": 340,
        "title": title,
    }
# include an option to choose how much you want to zoom for graphics
def plot_zoomable_multiline(
    data: pd.DataFrame,
    index_col: str,
    title: str,
    display_mode: DisplayMode,
    reference_name: Optional[str] = None,
    zoom_padding_pct: int = 5,
) -> None:
    if data.empty or index_col not in data.columns:
        return
    series_cols = [c for c in data.columns if c != index_col]
    if not series_cols:
        return
    df = _apply_display_mode(data, series_cols, display_mode, reference_name)
    long_df = df.melt(id_vars=[index_col], value_vars=series_cols, var_name="Série", value_name="Valeur")
    long_df[index_col] = pd.to_numeric(long_df[index_col], errors="coerce")
    long_df["Valeur"] = pd.to_numeric(long_df["Valeur"], errors="coerce")
    long_df = long_df.dropna(subset=[index_col, "Valeur"])
    if long_df.empty:
        return
    # min and max values
    y_min = float(long_df["Valeur"].min())
    y_max = float(long_df["Valeur"].max())
    span = y_max - y_min
    # compute a dynamic padding to keep a zoomed chart readable even on flat series
    pad = span * (max(int(zoom_padding_pct), 0) / 100.0) if span > 0.0 else max(abs(y_max), 1.0) * 0.02
    spec = _vega_line_spec(index_col, title, y_min - pad, y_max + pad)
    st.vega_lite_chart(long_df, spec, use_container_width=True)
# plot metrics
def plot_metrics(
    df: pd.DataFrame,
    index_col: str,
    choices: List[str],
    color_col: Optional[str] = None,
    display_mode: DisplayMode = "Prix direct",
    reference_name: Optional[str] = None,
    zoom_padding_pct: int = 5,
) -> None:
    # If any metric is chosen by the user
    if not choices:
        st.info("Sélectionne au moins une métrique à afficher.")
        return
# boucle for on all the metrics chosen
    for metric in choices:
        if color_col:
            plot_df = df.pivot(index=index_col, columns=color_col, values=metric).reset_index()
        else:
            plot_df = df[[index_col, metric]]
        plot_zoomable_multiline(
            plot_df,
            index_col=index_col,
            title=f"Graphe — {metric}",
            display_mode=display_mode,
            reference_name=reference_name,
            zoom_padding_pct=zoom_padding_pct,
        )
# Bars for performance
def plot_mean_bars_by_configuration(df: pd.DataFrame, metric_choices: List[str]) -> None:
    if not metric_choices:
        st.info("Sélectionne au moins une métrique pour afficher les histogrammes.")
        return
    mean_df = df.groupby("Configuration", as_index=False)[metric_choices].mean(numeric_only=True)
    for metric in metric_choices:
        st.subheader(f"Moyenne par configuration — {metric}")
        st.bar_chart(mean_df.set_index("Configuration")[[metric]])
# set up of the UI (title etc)
st.set_page_config(page_title="Dashboard Monte Carlo", layout="wide")
st.title("Dashboard Monte Carlo — analyses")
st.caption("Chaque onglet a ses propres paramètres. Les tableaux affichent tout, les graphes sont au choix.")
# sidebar for option and market
with st.sidebar:
    st.header("Marché")
    s0 = st.number_input("Spot S0", min_value=0.01, value=100.0, step=1.0)
    r = st.number_input("Taux r", value=0.04, step=0.005, format="%.4f")
    sigma = st.number_input("Volatilité sigma", min_value=0.0001, value=0.25, step=0.01, format="%.4f")
# Option
    st.header("Option")
    is_call = st.toggle("Call (sinon Put)", value=False)
    strike = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    pricing_date = st.date_input("Date de pricing", value=dt.date(2026, 2, 26))
    maturity_date = st.date_input("Maturité", value=dt.date(2027, 4, 26))
    q = st.number_input("Dividende continu q", value=0.0, step=0.005, format="%.4f")
    has_div = st.toggle("Dividende discret", value=True)
    ex_div_date = st.date_input("Ex-div date", value=dt.date(2026, 6, 21), disabled=not has_div)
    div_amount = st.number_input("Montant dividende", min_value=0.0, value=3.0, step=0.5, disabled=not has_div)
# errors handling if maturity < pricing   
if maturity_date <= pricing_date:
    st.error("La maturité doit être > date de pricing.")
    st.stop()
market = Market(S0=float(s0), r=float(r), sigma=float(sigma))
# name of all the tabs (onglets) whiwh appears in streamlit
tab_labels = [
    "Prix", "Greeks",
    "Convergence EU", "Convergence AM",
    "Convergence Delta", "Comparaison performance",
    "Test degré LS","Valeur actualisée",]
(
    tab_price, tab_greeks,
    tab_conv_eu, tab_conv_am,
    tab_conv_delta, tab_perf,
    tab_degree, tab_profile,
) = st.tabs(tab_labels)
with tab_price:
    # tab 1 = pricing 
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
# after choosing parameters you press the button to price
    if st.button("Lancer le pricing", type="primary"):
        trade = OptionTrade(
            # option parameters
            exercise=exercise_price,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # MC etc parameters
        params = build_core_pricing_params(
            int(n_paths_price),
            int(n_steps_price),
            int(seed_price),
            bool(anti_price),
            method_price,
            algo_price,
            basis_price,
            int(degree_price),
        )
        # all the metrics shown
        row = one_run(market, trade, params)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix", f"{row['Prix']:.6f}")
        m2.metric("Standard deviation", f"{row['Standard deviation']:.6f}")
        m3.metric("Standard error", f"{row['Standard error']:.6f}")
        m4.metric("Temps", f"{row['Temps (s)']:.3f}s")
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
# Tab 2 = greeks
with tab_greeks:
    st.subheader("Greeks")
    g1, g2, g3 = st.columns(3)
    exercise_greeks = g1.selectbox("Type d'exercice", ["european", "american"], index=0, key="g_ex")
    algo_greeks = g2.selectbox("Algorithme américain", ["ls", "naive"], index=0, key="g_algo")
    anti_greeks = g3.toggle("Antithétique", value=True, key="g_anti")
# choose the number of paths etc
    h1, h2, h3, h4 = st.columns(4)
    n_paths_greeks = h1.number_input("Nombre de chemins", min_value=1000, value=30000, step=2000, key="g_np")
    n_steps_greeks = h2.number_input("Nombre de pas", min_value=10, value=100, step=10, key="g_ns")
    seed_greeks = h3.number_input("Seed", min_value=0, value=127, step=1, key="g_seed")
    tree_n = h4.number_input("N benchmark arbre", min_value=50, value=250, step=50, key="g_tree")
    e1, e2 = st.columns(2)
    eps_spot = e1.number_input("Epsilon spot (ΔS)", min_value=0.0001, value=0.5, step=0.1, format="%.4f")
    eps_vol = e2.number_input("Epsilon vol (Δσ)", min_value=0.0001, value=0.01, step=0.005, format="%.4f")
# press the button to run
    if st.button("Lancer les greeks", type="primary"):
        trade = OptionTrade(
            exercise=exercise_greeks,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # special parameters for MC etc
        params = build_core_pricing_params(
            int(n_paths_greeks),
            int(n_steps_greeks),
            int(seed_greeks),
            bool(anti_greeks),
            method="vector",
            american_algo=algo_greeks,
            basis="laguerre",
            degree=2,
        )
        # record time
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
        # greeks for MC and bench 
        for k in keys:
            mc_v = mc_greeks.get(k, np.nan)
            ref_v = ref_greeks.get(k, np.nan)
            rows.append({"Greek": k, "Monte Carlo": mc_v, "Benchmark": ref_v, "Écart": mc_v - ref_v})
        st.metric("Temps de calcul greeks", f"{elapsed:.1f}s")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
# Tab 3 european convergence
with tab_conv_eu:
    st.subheader("Convergence EU")
    c1, c2, c3, c4 = st.columns(4)
    min_paths = c1.number_input("Min paths", min_value=100, value=1000, step=100, key="ceu_min")
    max_paths = c2.number_input("Max paths", min_value=500, value=30000, step=500, key="ceu_max")
    n_points = c3.slider("Nombre de points", min_value=3, max_value=12, value=6, key="ceu_pts")
    n_steps = c4.number_input("Pas MC", min_value=5, value=100, step=5, key="ceu_steps")
    d1, d2, d3, d4 = st.columns(4)
    seed = d1.number_input("Seed", min_value=0, value=127, step=1, key="ceu_seed")
    method = d2.selectbox("Simulation", ["vector", "scalar"], index=0, key="ceu_method")
    anti = d3.toggle("Antithétique", value=True, key="ceu_anti")
    r1, r2 = st.columns(2)
    show_tree_ref = r1.toggle("Afficher référence Arbre", value=True, key="ceu_show_tree")
    show_bs_ref = r2.toggle("Afficher référence Black-Scholes", value=True, key="ceu_show_bs")
# choose the metrics to draw
    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Erreur benchmark", "SE * sqrt(N)", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="ceu_plot",
    )
    display_mode = st.selectbox(
        "Type d'affichage",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        index=0,
        key="ceu_display_mode",
    )
    # if you want to see ecart about a ref: choose the reference
    reference_name = st.selectbox(
        "Référence (si mode = Écart à une référence)",
        ["Arbre trinomial", "Black-Scholes", "EU naive"],
        index=0,
        key="ceu_reference",
    )
    zoom_padding = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="ceu_zoom")
    # if you want to see metrics in function of "speed of convergence"
    show_inv_sqrt_curve = st.toggle("Afficher métrique vs 1/sqrt(N)", value=False, key="ceu_se_inv")
    inv_metric = st.selectbox(
        "Métrique pour vitesse de convergence",
        ["Prix", "Standard error", "Temps (s)", "Erreur benchmark"],
        key="ceu_inv_metric",
    )
    # press the button to run
    if st.button("Lancer convergence EU", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
        trade_eu = OptionTrade(
            exercise="european",
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # run convergence for naive european
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
        eu_df = eu_naive
        # Compare with BS and tree
        refs = compute_reference_lines(market, trade_eu)
        eu_df["Réf arbre"] = refs["Arbre trinomial"]
        eu_df["Réf Black-Scholes"] = refs["Black-Scholes"]
        st.caption(f"Benchmark EU: {eu_df['Nom benchmark'].iloc[0]} = {eu_df['Benchmark'].iloc[0]:.6f}")
        st.caption(f"Black-Scholes avec taux équivalent: r_eq = {refs['Taux équivalent BS']:.6f}")
        st.dataframe(eu_df, use_container_width=True)
        # you can choose to see BS and tree in graphics or not
        if "Prix" in graph_metrics:
            price_pivot = eu_df.pivot(index="n_paths", columns="Test", values="Prix")
            if show_tree_ref:
                price_pivot["Arbre trinomial"] = refs["Arbre trinomial"]
            if show_bs_ref:
                price_pivot["Black-Scholes"] = refs["Black-Scholes"]
            # to clearly see differences
            plot_zoomable_multiline(
                price_pivot.reset_index(),
                index_col="n_paths",
                title="Prix convergence EU",
                display_mode=display_mode,
                reference_name=reference_name,
                zoom_padding_pct=int(zoom_padding),
            )
            graph_metrics = [m for m in graph_metrics if m != "Prix"]
        plot_metrics(eu_df, index_col="n_paths", choices=graph_metrics, color_col="Test", 
                     display_mode=display_mode, reference_name=reference_name, 
                     zoom_padding_pct=int(zoom_padding))
        # if you want to see with 1/sqrt(N)
        if show_inv_sqrt_curve:
            se_inv_df = add_inv_sqrt_n_fit(eu_df, inv_metric)
            plot_zoomable_multiline(
                se_inv_df,
                index_col="1/sqrt(N)",
                title=f"{inv_metric} vs 1/sqrt(N) — vitesse convergence (EU)",
                display_mode=display_mode,
                reference_name=inv_metric,
                zoom_padding_pct=int(zoom_padding),
            )
# Tab 4: American convergence
with tab_conv_am:
    st.subheader("Convergence AM")
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
    r1, r2 = st.columns(2)
    show_tree_ref = r1.toggle("Afficher référence Arbre", value=True, key="cam_show_tree")
    show_bs_ref = r2.toggle("Afficher référence Black-Scholes", value=True, key="cam_show_bs")
# metrics to draw
    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Erreur benchmark", "SE * sqrt(N)", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="cam_plot",
    )
    display_mode = st.selectbox(
        "Type d'affichage",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        index=0,
        key="cam_display_mode",
    )
    # choose a reference 
    reference_name = st.selectbox(
        "Référence (si mode = Écart à une référence)",
        ["Arbre trinomial", "Black-Scholes", "AM naive", "AM LS"],
        index=0,
        key="cam_reference",
    )
    zoom_padding = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="cam_zoom")
    show_inv_sqrt_curve = st.toggle("Afficher métrique vs 1/sqrt(N)", value=False, key="cam_se_inv")
    inv_metric = st.selectbox(
        "Métrique pour vitesse de convergence",
        ["Prix", "Standard error", "Temps (s)", "Erreur benchmark"],
        key="cam_inv_metric",
    )
    # button to press to run
    if st.button("Lancer convergence AM", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
        trade_am = OptionTrade(
            exercise="american",
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # convergence naive
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
# LS
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
# tree and BS as references
        am_df = pd.concat([am_naive, am_ls], ignore_index=True)
        refs = compute_reference_lines(market, trade_am)
        am_df["Réf arbre"] = refs["Arbre trinomial"]
        am_df["Réf Black-Scholes"] = refs["Black-Scholes"]
        st.caption(f"Benchmark AM: {am_df['Nom benchmark'].iloc[0]} = {am_df['Benchmark'].iloc[0]:.6f}")
        st.caption(f"Black-Scholes avec taux équivalent: r_eq = {refs['Taux équivalent BS']:.6f}")
        st.dataframe(am_df, use_container_width=True)
        if "Prix" in graph_metrics:
            price_pivot = am_df.pivot(index="n_paths", columns="Test", values="Prix")
            if show_tree_ref:
                price_pivot["Arbre trinomial"] = refs["Arbre trinomial"]
            if show_bs_ref:
                price_pivot["Black-Scholes"] = refs["Black-Scholes"]
# plot with zoom
            plot_zoomable_multiline(
                price_pivot.reset_index(),
                index_col="n_paths",
                title="Prix convergence AM",
                display_mode=display_mode,
                reference_name=reference_name,
                zoom_padding_pct=int(zoom_padding),
            )
            graph_metrics = [m for m in graph_metrics if m != "Prix"]
        plot_metrics(am_df, index_col="n_paths", choices=graph_metrics, 
                     color_col="Test", display_mode=display_mode, 
                     reference_name=reference_name, zoom_padding_pct=int(zoom_padding))
        # see speed of convergence
        if show_inv_sqrt_curve:
            am_ls_only = am_df[am_df["Test"] == "AM LS"]
            se_inv_df = add_inv_sqrt_n_fit(am_ls_only, inv_metric)
            plot_zoomable_multiline(
                se_inv_df,
                index_col="1/sqrt(N)",
                title=f"{inv_metric} vs 1/sqrt(N) — vitesse convergence (AM LS)",
                display_mode=display_mode,
                reference_name=inv_metric,
                zoom_padding_pct=int(zoom_padding),
            )
# Tab 5: delta convergence
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
# graphics to draw (users choice)
    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Delta MC", "Delta benchmark", "Erreur absolue delta", "Temps (s)"],
        default=["Delta MC", "Delta benchmark", "Erreur absolue delta"],
        key="cd_plot",
    )
    display_mode = st.selectbox(
        "Type d'affichage",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        index=0,
        key="cd_display_mode",
    )
    # reference if ecart to a reference chosen
    reference_name = st.selectbox(
        "Référence (si mode = Écart à une référence)",
        ["Delta MC", "Delta benchmark"],
        index=1,
        key="cd_reference",
    )
    zoom_padding = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="cd_zoom")
    show_delta_pair = st.toggle("Tracer Delta MC et benchmark", value=True, key="cd_pair")
    if st.button("Lancer convergence delta", type="primary"):
        grid = convergence_grid(int(min_paths), int(max_paths), int(n_points))
        trade = OptionTrade(
            # option parameters
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
            # parameters chose in the tab
            params = build_core_pricing_params(
                int(n),
                int(n_steps),
                int(seed),
                bool(anti),
                method="vector",
                american_algo="ls",
                basis="laguerre",
                degree=2,
            )
            # record time
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
        # show delta MC and bench 
        delta_df = pd.DataFrame(rows)
        st.dataframe(delta_df, use_container_width=True)
        remaining_metrics = list(graph_metrics)
        if show_delta_pair:
            plot_zoomable_multiline(
                delta_df[["n_paths", "Delta MC", "Delta benchmark"]],
                index_col="n_paths",
                title="Delta MC vs benchmark",
                display_mode=display_mode,
                reference_name=reference_name,
                zoom_padding_pct=int(zoom_padding),
            )
            remaining_metrics = [m for m in remaining_metrics if m not in {"Delta MC", "Delta benchmark"}]
        plot_metrics(delta_df, index_col="n_paths", choices=remaining_metrics, 
                     display_mode=display_mode, reference_name=reference_name, 
                     zoom_padding_pct=int(zoom_padding))
# Tab 6: performance comparisons
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
# graphics to draw
    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Prix", "Standard deviation", "Standard error", "Temps (s)"],
        default=["Prix", "Standard error", "Temps (s)"],
        key="pf_plot",
    )
    # comparison in function of N
    st.markdown("**Étude en fonction du nombre de chemins N**")
    a1, a2, a3, a4 = st.columns(4)
    paths_min = a1.number_input("N min", min_value=100, value=1000, step=100, key="pf_paths_min")
    paths_max = a2.number_input("N max", min_value=200, value=30000, step=200, key="pf_paths_max")
    paths_points = a3.slider("Nb points N", min_value=3, max_value=20, value=8, key="pf_paths_pts")
    metric_steps = a4.selectbox("Métrique vs N", ["Prix", "Standard error", "Temps (s)"], key="pf_steps_metric")
    b1, b2, b3 = st.columns(3)
    method_for_anti = b1.selectbox("Méthode pour anti/non-anti", ["vector", "scalar"], key="pf_steps_method")
    anti_for_method = b2.toggle("Antithétique pour scalar/vector", value=True, key="pf_steps_anti")
    display_mode_steps = b3.selectbox(
        "Type affichage vs N",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        key="pf_steps_display",
    )
    c1, c2 = st.columns(2)
    # choose references for comparison
    reference_steps_anti = c1.selectbox("Référence anti/non-anti", ["anti=False", "anti=True"], key="pf_ref_anti")
    reference_steps_method = c2.selectbox("Référence scalar/vector", ["Vectorielle", "Scalaire"], key="pf_ref_method")
    zoom_steps = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="pf_zoom_steps")
    # button to press
    if st.button("Lancer comparaison performance", type="primary"):
        trade = OptionTrade(
            exercise=exercise_perf,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # all seeds to calculate perfs
        seeds = [int(seed_start) + k for k in range(int(seed_count))]
        rows = []
        # scalar vs vector
        for method_perf in ["vector", "scalar"]:
            for anti_perf in [False, True]:
                for seed_perf in seeds:
                    p = build_core_pricing_params(
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
        # Anti vs non anti
        st.markdown("**Moyennes anti / non-anti (toutes méthodes confondues)**")
        anti_mean = (
            perf_df.groupby("Antithétique", as_index=False)[["Prix", "Standard deviation", "Standard error", "Temps (s)"]]
            .mean()
            .sort_values("Antithétique")
        )
        st.dataframe(anti_mean, use_container_width=True)
        # Moyennes
        st.markdown("**Moyennes scalaire / vectorielle (tous modes anti confondus)**")
        method_mean = (
            perf_df.groupby("Méthode", as_index=False)[["Prix", "Standard deviation", "Standard error", "Temps (s)"]]
            .mean()
        )
        st.dataframe(method_mean, use_container_width=True)
        st.markdown("**Barres de moyenne (1 seule série)**")
        plot_mean_bars_by_configuration(perf_df, graph_metrics)
    # button to press to run study by number of paths
    if st.button("Lancer étude par nombre de chemins", type="secondary"):
        trade_steps = OptionTrade(
            exercise=exercise_perf,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        paths_grid = convergence_grid(int(paths_min), int(paths_max), int(paths_points))
        # Non anti 
        anti_false_df = run_paths_analysis(
            market,
            trade_steps,
            paths_grid=list(map(int, paths_grid)),
            n_steps=int(n_steps_perf),
            seed=int(seed_start),
            algo=algo_perf,
            basis="laguerre",
            degree=2,
            method=method_for_anti,
            antithetic=False,
        )
        # Anti = true
        anti_true_df = run_paths_analysis(
            market,
            trade_steps,
            paths_grid=list(map(int, paths_grid)),
            n_steps=int(n_steps_perf),
            seed=int(seed_start),
            algo=algo_perf,
            basis="laguerre",
            degree=2,
            method=method_for_anti,
            antithetic=True,
        )
        # plot
        anti_plot_df = pd.DataFrame({
            "n_paths": anti_false_df["n_paths"],
            "anti=False": anti_false_df[metric_steps],
            "anti=True": anti_true_df[metric_steps],
        })
        st.markdown("**Anti vs non-anti en fonction de N**")
        plot_zoomable_multiline(
            anti_plot_df,
            index_col="n_paths",
            title=f"{metric_steps} vs N — anti/non-anti",
            display_mode=display_mode_steps,
            reference_name=reference_steps_anti,
            zoom_padding_pct=int(zoom_steps),
        )
        # analysis per path for vector
        vector_df = run_paths_analysis(
            market,
            trade_steps,
            paths_grid=list(map(int, paths_grid)),
            n_steps=int(n_steps_perf),
            seed=int(seed_start),
            algo=algo_perf,
            basis="laguerre",
            degree=2,
            method="vector",
            antithetic=bool(anti_for_method),
        )
        # analysis per path for scalar
        scalar_df = run_paths_analysis(
            market,
            trade_steps,
            paths_grid=list(map(int, paths_grid)),
            n_steps=int(n_steps_perf),
            seed=int(seed_start),
            algo=algo_perf,
            basis="laguerre",
            degree=2,
            method="scalar",
            antithetic=bool(anti_for_method),
        )
        # plot
        method_plot_df = pd.DataFrame({
            "n_paths": vector_df["n_paths"],
            "Vectorielle": vector_df[metric_steps],
            "Scalaire": scalar_df[metric_steps],
        })
        st.markdown("**Scalaire vs vectorielle en fonction de N**")
        plot_zoomable_multiline(
            method_plot_df,
            index_col="n_paths",
            title=f"{metric_steps} vs N — scalar/vector",
            display_mode=display_mode_steps,
            reference_name=reference_steps_method,
            zoom_padding_pct=int(zoom_steps),
        )
# Tab 7: comparison with regression degree LS
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
    # choose what to display
    display_mode = st.selectbox(
        "Type d'affichage",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        index=0,
        key="deg_display_mode",
    )
    reference_name = st.selectbox(
        "Référence (si mode = Écart à une référence)",
        ["laguerre", "power"],
        index=0,
        key="deg_reference",
    )
    zoom_padding = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="deg_zoom")
    # press button to run
    if st.button("Lancer test de degré", type="primary"):
        if not basis_choices:
            st.warning("Sélectionne au moins une base de régression.")
        else:
            trade = OptionTrade(
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
            # parameters
            for basis in basis_choices:
                for degree in range(int(degrees[0]), int(degrees[1]) + 1):
                    p = build_core_pricing_params(
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
                    # output with basic metrics
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
            deg_plot_df = deg_df.pivot(index="Degré", columns="Base", values=metric_deg).reset_index()
            plot_zoomable_multiline(
                deg_plot_df,
                index_col="Degré",
                title=f"Graphe — {metric_deg}",
                display_mode=display_mode,
                reference_name=reference_name,
                zoom_padding_pct=int(zoom_padding),
            )
# Tab 8: valeur moyenne actualisée
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
    # choose what to display
    graph_metrics = st.multiselect(
        "Métriques à tracer",
        ["Moyenne actualisée", "Standard deviation"],
        default=["Moyenne actualisée", "Standard deviation"],
        key="pr_plot",
    )
    display_mode = st.selectbox(
        "Type d'affichage",
        ["Prix direct", "Écart à une référence", "Écart au premier point", "Écart à la moyenne"],
        index=0,
        key="pr_display_mode",
    )
    # choose a reference
    reference_name = st.selectbox(
        "Référence (si mode = Écart à une référence)",
        ["Moyenne actualisée", "Standard deviation"],
        index=0,
        key="pr_reference",
    )
    zoom_padding = st.slider("Zoom marge Y (%)", min_value=0, max_value=25, value=2, key="pr_zoom")
    # a button to run
    if st.button("Lancer profil actualisé", type="primary"):
        trade = OptionTrade(
            exercise=exercise_prof,
            strike=float(strike),
            is_call=bool(is_call),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            q=float(q),
            ex_div_date=ex_div_date if has_div else None,
            div_amount=float(div_amount) if has_div else 0.0,
        )
        # specific parameters chosen in the tab
        params = build_core_pricing_params(
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
        plot_metrics(
            prof_df,
            index_col="Pas",
            choices=graph_metrics,
            display_mode=display_mode,
            reference_name=reference_name,
            zoom_padding_pct=int(zoom_padding),
        )