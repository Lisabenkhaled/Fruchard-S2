import datetime as dt
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is importable when app is launched from subfolders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core_pricer import CorePricingParams, core_price
from model.market import Market
from model.option import OptionTrade
from pricing_tree.pricer import price_tree_backward_direct


def bermudan_exercise_steps(pricing_date: dt.date, maturity_date: dt.date, n_steps: int) -> list[int]:
    """Build monthly exercise grid (day 26) plus maturity."""
    dates: list[dt.date] = []
    cur = pricing_date

    while cur <= maturity_date:
        d = dt.date(cur.year, cur.month, 26)
        if pricing_date <= d <= maturity_date:
            dates.append(d)

        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

    if maturity_date not in dates:
        dates.append(maturity_date)

    total_days = max((maturity_date - pricing_date).days, 1)
    steps = []
    for d in dates:
        tau = (d - pricing_date).days / total_days
        idx = int(round(tau * n_steps))
        steps.append(min(max(idx, 0), n_steps))

    return sorted(set(steps))


def build_trade(
    option_kind: str,
    strike: float,
    pricing_date: dt.date,
    maturity_date: dt.date,
    is_call: bool,
    q: float,
    ex_div_date: dt.date | None,
    div_amount: float,
) -> OptionTrade:
    return OptionTrade(
        strike=float(strike),
        is_call=is_call,
        exercise=option_kind,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=q,
        ex_div_date=ex_div_date,
        div_amount=div_amount,
    )


def normalize_n_paths(n_paths: int, antithetic: bool) -> int:
    n = max(int(n_paths), 1)
    if antithetic and n % 2 == 1:
        return n + 1
    return n


def run_batch_pricing(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int,
    methods: list[str],
    antithetics: list[bool],
    american_algo: str,
    basis: str,
    degree: int,
    bermudan_steps: list[int],
    digital_strike: float | None,
    digital_payout: float,
) -> pd.DataFrame:
    rows = []

    for method, antithetic in product(methods, antithetics):
        n_paths_used = normalize_n_paths(n_paths, antithetic)
        params = CorePricingParams(
            n_paths=n_paths_used,
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
            method=method,
            american_algo=american_algo,
            basis=basis,
            degree=degree,
            exercise_steps=bermudan_steps,
            digital_strike=digital_strike,
            digital_payout=digital_payout,
        )

        price, std, se, elapsed = core_price(market, trade, params)

        rows.append(
            {
                "engine": "monte_carlo",
                "method": method,
                "antithetic": antithetic,
                "n_paths_used": n_paths_used,
                "price": price,
                "std": std,
                "se": se,
                "ci_low_95": price - 1.96 * se,
                "ci_high_95": price + 1.96 * se,
                "time_s": elapsed,
            }
        )

    df = pd.DataFrame(rows).sort_values(["method", "antithetic"]).reset_index(drop=True)
    baseline_idx = df["time_s"].idxmin()
    baseline = df.loc[baseline_idx]
    df["diff_price_vs_fastest"] = df["price"] - baseline["price"]
    df["time_ratio_vs_fastest"] = df["time_s"] / baseline["time_s"]
    return df


def run_single_mc(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int,
    method: str,
    antithetic: bool,
    american_algo: str,
    basis: str,
    degree: int,
    bermudan_steps: list[int],
    digital_strike: float | None,
    digital_payout: float,
) -> dict:
    n_paths_used = normalize_n_paths(n_paths, antithetic)
    params = CorePricingParams(
        n_paths=n_paths_used,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
        method=method,
        american_algo=american_algo,
        basis=basis,
        degree=degree,
        exercise_steps=bermudan_steps,
        digital_strike=digital_strike,
        digital_payout=digital_payout,
    )
    price, std, se, elapsed = core_price(market, trade, params)
    return {
        "engine": "monte_carlo",
        "method": method,
        "antithetic": antithetic,
        "n_paths_used": n_paths_used,
        "price": price,
        "std": std,
        "se": se,
        "ci_low_95": price - 1.96 * se,
        "ci_high_95": price + 1.96 * se,
        "time_s": elapsed,
    }


def run_tree_single(
    *,
    s0: float,
    r: float,
    sigma: float,
    strike: float,
    is_call: bool,
    exercise: str,
    pricing_date: dt.date,
    maturity_date: dt.date,
    tree_steps: int,
    ex_div_date: dt.date | None,
    div_amount: float,
    optimize: bool,
    threshold: float,
) -> dict:
    out = price_tree_backward_direct(
        S0=float(s0),
        r=float(r),
        sigma=float(sigma),
        K=float(strike),
        is_call=bool(is_call),
        exercise=exercise,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        N=int(tree_steps),
        ex_div_date=ex_div_date,
        div_amount=float(div_amount),
        optimize=bool(optimize),
        threshold=float(threshold),
        return_tree=False,
    )
    return {
        "engine": "pricing_tree",
        "method": "backward",
        "antithetic": False,
        "price": out["tree_price"],
        "std": np.nan,
        "se": np.nan,
        "ci_low_95": np.nan,
        "ci_high_95": np.nan,
        "time_s": out["tree_time"],
    }


def regression_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def run_mc_convergence_regression(
    market: Market,
    trade: OptionTrade,
    n_steps: int,
    seed: int,
    antithetic: bool,
    method: str,
    american_algo: str,
    basis: str,
    degree: int,
    exercise_steps: list[int],
    digital_strike: float | None,
    digital_payout: float,
    n_paths_grid: list[int],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for n_paths in n_paths_grid:
        n_paths_used = normalize_n_paths(int(n_paths), bool(antithetic))
        params = CorePricingParams(
            n_paths=n_paths_used,
            n_steps=int(n_steps),
            seed=int(seed),
            antithetic=bool(antithetic),
            method=method,
            american_algo=american_algo,
            basis=basis,
            degree=degree,
            exercise_steps=exercise_steps,
            digital_strike=digital_strike,
            digital_payout=float(digital_payout),
        )
        price, std, se, elapsed = core_price(market, trade, params)
        rows.append({"n_paths": n_paths, "n_paths_used": n_paths_used, "price": price, "std": std, "se": se, "time_s": elapsed})

    df = pd.DataFrame(rows).sort_values("n_paths").reset_index(drop=True)
    slope, _ = regression_line(np.log(df["n_paths"].to_numpy()), np.log(df["se"].to_numpy()))
    return df, slope


def run_stress_regression(
    var_name: str,
    var_values: np.ndarray,
    base_inputs: dict,
    mc_enabled: bool,
    tree_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_rows = []

    for val in var_values:
        local = base_inputs.copy()
        local[var_name] = float(val)

        market = Market(S0=local["s0"], r=local["r"], sigma=local["sigma"])
        trade = build_trade(
            option_kind=local["option_kind"],
            strike=local["strike"],
            pricing_date=local["pricing_date"],
            maturity_date=local["maturity_date"],
            is_call=local["is_call"],
            q=local["q"],
            ex_div_date=local["ex_div_date"] if local["has_discrete_div"] else None,
            div_amount=local["div_amount"] if local["has_discrete_div"] else 0.0,
        )

        if mc_enabled:
            n_paths_used = normalize_n_paths(local["n_paths"], local["reg_antithetic"])
            mc_params = CorePricingParams(
                n_paths=n_paths_used,
                n_steps=local["n_steps"],
                seed=local["seed"],
                antithetic=local["reg_antithetic"],
                method=local["reg_method"],
                american_algo=local["american_algo"],
                basis=local["basis"],
                degree=local["degree"],
                exercise_steps=local["bermudan_steps"],
                digital_strike=local["digital_strike_in"],
                digital_payout=local["digital_payout"],
            )
            mc_price, _, _, _ = core_price(market, trade, mc_params)
            price_rows.append({"x": val, "engine": "monte_carlo", "price": mc_price})

        if tree_enabled and local["option_kind"] in ("american", "european"):
            tree_row = run_tree_single(
                s0=local["s0"],
                r=local["r"],
                sigma=local["sigma"],
                strike=local["strike"],
                is_call=local["is_call"],
                exercise=local["option_kind"],
                pricing_date=local["pricing_date"],
                maturity_date=local["maturity_date"],
                tree_steps=local["tree_steps"],
                ex_div_date=local["ex_div_date"] if local["has_discrete_div"] else None,
                div_amount=local["div_amount"] if local["has_discrete_div"] else 0.0,
                optimize=local["tree_optimize"],
                threshold=local["tree_threshold"],
            )
            price_rows.append({"x": val, "engine": "pricing_tree", "price": tree_row["price"]})

    prices_df = pd.DataFrame(price_rows)

    reg_rows = []
    for engine, sub in prices_df.groupby("engine"):
        slope, intercept = regression_line(sub["x"].to_numpy(), sub["price"].to_numpy())
        reg_rows.append({"engine": engine, "slope": slope, "intercept": intercept})

    reg_df = pd.DataFrame(reg_rows)
    return prices_df, reg_df


st.set_page_config(page_title="Unified Pricing Dashboard (MC + Tree)", layout="wide")
st.title("📈 Unified Pricing Dashboard: Monte Carlo + Pricing Tree")
st.caption("Choisis les onglets à exécuter : tu peux lancer seulement prix+écart-type, ou ajouter les comparatifs.")

with st.sidebar:
    st.header("Paramètres de marché")
    s0 = st.number_input("Spot S0", min_value=0.01, value=100.0, step=1.0)
    r = st.number_input("Taux r", value=0.04, step=0.005, format="%.4f")
    sigma = st.number_input("Volatilité sigma", min_value=0.0001, value=0.25, step=0.01, format="%.4f")

    st.header("Paramètres option")
    option_kind = st.selectbox("Style d'exercice", options=["european", "american", "bermudan", "digital_american"], index=1)
    is_call = st.toggle("Call (désactivé = Put)", value=False)
    strike = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    pricing_date = st.date_input("Date de pricing", value=dt.date(2026, 2, 26))
    maturity_date = st.date_input("Maturité", value=dt.date(2027, 4, 26))

    q = st.number_input("Dividende continu q", value=0.0, step=0.005, format="%.4f")
    has_discrete_div = st.toggle("Dividende discret", value=True)
    ex_div_date = st.date_input("Ex-div date", value=dt.date(2026, 6, 21), disabled=not has_discrete_div)
    div_amount = st.number_input("Montant dividende", min_value=0.0, value=3.0, step=0.5, disabled=not has_discrete_div)

    st.header("Réglages MC")
    n_paths = st.number_input("Nombre de chemins", min_value=100, value=30000, step=1000)
    n_steps = st.number_input("Nombre de pas MC", min_value=5, value=120, step=5)
    seed = st.number_input("Seed", min_value=0, value=127, step=1)

    st.subheader("Run principal (Prix+Écart-type)")
    simple_method = st.selectbox("Méthode simple", options=["vector", "scalar"], index=0)
    simple_antithetic = st.toggle("Antithetic (simple)", value=False)

    st.subheader("Comparatif MC (optionnel)")
    methods = st.multiselect("Modes MC à comparer", options=["vector", "scalar"], default=["vector"])
    antithetic_labels = st.multiselect("Antithetic à comparer", options=["off", "on"], default=["off"])
    antithetics = [label == "on" for label in antithetic_labels]

    american_algo = st.selectbox("Algo américain (vanilla)", options=["ls", "naive"], index=0)
    basis = st.selectbox("Base régression (LS)", options=["laguerre", "power"], index=0)
    degree = st.slider("Degré régression LS", min_value=1, max_value=6, value=2)

    st.header("Exotiques MC")
    digital_strike = st.number_input("Digital strike", min_value=0.01, value=99.0, step=1.0)
    digital_payout = st.number_input("Digital payout", min_value=0.01, value=1.0, step=0.1)

    st.header("Pricing Tree")
    tree_steps = st.number_input("Pas arbre N", min_value=10, value=400, step=10)
    tree_optimize = st.toggle("Pruning arbre", value=False)
    tree_threshold = st.number_input("Seuil pruning", min_value=1e-20, value=1e-14, format="%.1e")

    st.subheader("Convergence MC (optionnel)")
    conv_method = st.selectbox("Méthode convergence", options=["vector", "scalar"], index=0)
    conv_antithetic = st.toggle("Antithetic convergence", value=False)

    st.header("Onglets à exécuter")
    selected_panels = st.multiselect(
        "Sélectionne ce que tu veux lancer",
        options=[
            "Prix + Ecart-type",
            "Comparatif performances MC",
            "Comparatif MC vs Tree",
            "Convergence MC",
            "Régression de sensibilité",
        ],
        default=["Prix + Ecart-type"],
    )

    mc_runs = len(methods) * len(antithetics)
    workload_mc = int(n_paths) * int(n_steps) * max(mc_runs, 1)
    st.caption(f"Charge estimée (comparatif MC): {workload_mc:,} unités")

    run_btn = st.button("Lancer", type="primary")

if not selected_panels:
    st.warning("Sélectionne au moins un onglet à exécuter.")
elif run_btn:
    if maturity_date <= pricing_date:
        st.error("La maturité doit être postérieure à la date de pricing.")
        st.stop()

    market = Market(S0=s0, r=r, sigma=sigma)
    trade = build_trade(
        option_kind=option_kind,
        strike=strike,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        is_call=is_call,
        q=q,
        ex_div_date=ex_div_date if has_discrete_div else None,
        div_amount=div_amount if has_discrete_div else 0.0,
    )

    bermudan_steps = bermudan_exercise_steps(pricing_date, maturity_date, int(n_steps)) if option_kind == "bermudan" else []
    digital_strike_in = float(digital_strike) if option_kind == "digital_american" else None

    tabs = st.tabs(selected_panels)
    tab_map = {name: tab for name, tab in zip(selected_panels, tabs)}

    if "Prix + Ecart-type" in tab_map:
        with tab_map["Prix + Ecart-type"]:
            with st.spinner("Calcul du run principal..."):
                one = run_single_mc(
                    market=market,
                    trade=trade,
                    n_paths=int(n_paths),
                    n_steps=int(n_steps),
                    seed=int(seed),
                    method=simple_method,
                    antithetic=bool(simple_antithetic),
                    american_algo=american_algo,
                    basis=basis,
                    degree=degree,
                    bermudan_steps=bermudan_steps,
                    digital_strike=digital_strike_in,
                    digital_payout=float(digital_payout),
                )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prix", f"{one['price']:.6f}")
            c2.metric("Std", f"{one['std']:.6f}")
            c3.metric("SE", f"{one['se']:.6f}")
            c4.metric("Temps", f"{one['time_s']:.3f}s")
            st.dataframe(pd.DataFrame([one]), use_container_width=True)

    if "Comparatif performances MC" in tab_map:
        with tab_map["Comparatif performances MC"]:
            if not methods or not antithetics:
                st.warning("Sélectionne au moins un mode et un antithetic pour le comparatif.")
            else:
                hard_mc_limit = 60_000_000
                combos = max(1, len(methods) * len(antithetics))
                if int(n_paths) * int(n_steps) * combos > hard_mc_limit:
                    st.error("Charge comparatif trop élevée. Réduis n_paths/n_steps ou le nombre de combinaisons.")
                else:
                    with st.spinner("Calcul du comparatif MC..."):
                        mc_df = run_batch_pricing(
                            market=market,
                            trade=trade,
                            n_paths=int(n_paths),
                            n_steps=int(n_steps),
                            seed=int(seed),
                            methods=methods,
                            antithetics=antithetics,
                            american_algo=american_algo,
                            basis=basis,
                            degree=degree,
                            bermudan_steps=bermudan_steps,
                            digital_strike=digital_strike_in,
                            digital_payout=float(digital_payout),
                        )
                    st.dataframe(mc_df, use_container_width=True)

                    st.markdown("**Moyennes par configuration (method × antithetic)**")
                    by_combo = mc_df.groupby(["method", "antithetic"], as_index=False)[["price", "std", "se", "time_s"]].mean()
                    st.dataframe(by_combo, use_container_width=True)

                    st.markdown("**Moyennes par antithetic**")
                    by_anti = mc_df.groupby(["antithetic"], as_index=False)[["price", "std", "se", "time_s"]].mean()
                    st.dataframe(by_anti, use_container_width=True)

                    st.markdown("**Moyennes par méthode (scalar vs vector)**")
                    by_method = mc_df.groupby(["method"], as_index=False)[["price", "std", "se", "time_s"]].mean()
                    st.dataframe(by_method, use_container_width=True)

                    st.bar_chart(mc_df.set_index(mc_df["method"] + "|anti=" + mc_df["antithetic"].astype(str))["time_s"])

    if "Comparatif MC vs Tree" in tab_map:
        with tab_map["Comparatif MC vs Tree"]:
            if option_kind not in ("european", "american"):
                st.warning("Le pricing tree de cette vue supporte european/american vanilla uniquement.")
            else:
                with st.spinner("Calcul MC + Tree..."):
                    mc_one = run_single_mc(
                        market=market,
                        trade=trade,
                        n_paths=int(n_paths),
                        n_steps=int(n_steps),
                        seed=int(seed),
                        method=simple_method,
                        antithetic=bool(simple_antithetic),
                        american_algo=american_algo,
                        basis=basis,
                        degree=degree,
                        bermudan_steps=bermudan_steps,
                        digital_strike=digital_strike_in,
                        digital_payout=float(digital_payout),
                    )
                    tree_one = run_tree_single(
                        s0=s0,
                        r=r,
                        sigma=sigma,
                        strike=strike,
                        is_call=is_call,
                        exercise=option_kind,
                        pricing_date=pricing_date,
                        maturity_date=maturity_date,
                        tree_steps=int(tree_steps),
                        ex_div_date=ex_div_date if has_discrete_div else None,
                        div_amount=div_amount if has_discrete_div else 0.0,
                        optimize=tree_optimize,
                        threshold=tree_threshold,
                    )
                comp_df = pd.DataFrame([mc_one, tree_one])
                comp_df["diff_price_vs_mc"] = comp_df["price"] - mc_one["price"]
                comp_df["time_ratio_vs_mc"] = comp_df["time_s"] / mc_one["time_s"]
                st.dataframe(comp_df, use_container_width=True)

    if "Convergence MC" in tab_map:
        with tab_map["Convergence MC"]:
            with st.spinner("Calcul convergence MC..."):
                n_paths_grid = np.unique(np.round(np.logspace(np.log10(1000), np.log10(max(2000, int(n_paths))), 8)).astype(int))
                conv_df, conv_slope = run_mc_convergence_regression(
                    market=market,
                    trade=trade,
                    n_steps=int(n_steps),
                    seed=int(seed),
                    antithetic=bool(conv_antithetic),
                    method=conv_method,
                    american_algo=american_algo,
                    basis=basis,
                    degree=degree,
                    exercise_steps=bermudan_steps,
                    digital_strike=digital_strike_in,
                    digital_payout=float(digital_payout),
                    n_paths_grid=list(map(int, n_paths_grid)),
                )
            st.metric("Pente log(SE) vs log(Npaths)", f"{conv_slope:.4f}")
            st.dataframe(conv_df, use_container_width=True)
            st.line_chart(conv_df.set_index("n_paths")["se"])

    if "Régression de sensibilité" in tab_map:
        with tab_map["Régression de sensibilité"]:
            var_name = st.selectbox("Variable stress", options=["s0", "sigma", "r", "strike"], index=0)
            pct_span = st.slider("Amplitude ±%", min_value=5, max_value=50, value=20, step=5)
            n_points = st.slider("Nombre de points", min_value=5, max_value=21, value=9, step=2)
            base_val = {"s0": float(s0), "sigma": float(sigma), "r": float(r), "strike": float(strike)}[var_name]
            low = max(base_val * (1.0 - pct_span / 100.0), 1e-6 if var_name in ("sigma", "r") else 0.01)
            high = base_val * (1.0 + pct_span / 100.0)
            var_values = np.linspace(low, high, n_points)

            base_inputs = {
                "s0": float(s0),
                "sigma": float(sigma),
                "r": float(r),
                "strike": float(strike),
                "option_kind": option_kind,
                "pricing_date": pricing_date,
                "maturity_date": maturity_date,
                "is_call": is_call,
                "q": float(q),
                "has_discrete_div": has_discrete_div,
                "ex_div_date": ex_div_date,
                "div_amount": float(div_amount),
                "n_paths": int(max(5000, n_paths // 10)),
                "n_steps": int(n_steps),
                "seed": int(seed),
                "reg_antithetic": True,
                "reg_method": "vector",
                "american_algo": american_algo,
                "basis": basis,
                "degree": int(degree),
                "bermudan_steps": bermudan_steps,
                "digital_strike_in": digital_strike_in,
                "digital_payout": float(digital_payout),
                "tree_steps": int(max(200, tree_steps // 2)),
                "tree_optimize": bool(tree_optimize),
                "tree_threshold": float(tree_threshold),
            }

            with st.spinner("Calcul régression de sensibilité..."):
                prices_df, reg_df = run_stress_regression(
                    var_name=var_name,
                    var_values=var_values,
                    base_inputs=base_inputs,
                    mc_enabled=True,
                    tree_enabled=True,
                )
            st.dataframe(prices_df, use_container_width=True)
            st.dataframe(reg_df, use_container_width=True)
            st.line_chart(prices_df.pivot(index="x", columns="engine", values="price"))
else:
    st.info("Sélectionne des onglets, puis clique sur Lancer.")
