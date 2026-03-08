# excel_early_exercise.py
from __future__ import annotations

import numpy as np
import xlwings as xw

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_european_naive_mc_vector, price_american_ls_vector

import model.path_simulator as path_sim
from model.path_simulator import simulate_gbm_paths_vector


# =============================================================================
# Helpers
# =============================================================================

def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("true", "vrai", "yes", "oui", "1")


def _as_date_excel(x):
    return x.date() if hasattr(x, "date") else x


def _write_matrix_in_chunks(
    sht: xw.Sheet,
    top_row: int,
    left_col: int,
    M: np.ndarray,
    chunk_rows: int = 500,
) -> None:
    """Write a 2-D NumPy array to a sheet in row chunks to avoid memory issues."""
    n_rows, _ = M.shape
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        sht.range((top_row + r0, left_col)).value = M[r0:r1, :].tolist()


def _get_or_simulate_paths(
    market: Market,
    trade: OptionTrade,
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times, S) — reusing LAST_PATHS set by the simulator if available,
    otherwise re-simulating with the same parameters.
    """
    S = getattr(path_sim, "LAST_PATHS", None)
    times = getattr(path_sim, "LAST_TIMES", None)

    if S is None or times is None:
        times, S = simulate_gbm_paths_vector(
            market, trade, n_paths, n_steps, seed, antithetic
        )

    S = np.asarray(S, dtype=float)
    if S.shape != (n_paths, n_steps + 1):
        raise RuntimeError(
            f"Paths shape mismatch: got {S.shape}, expected {(n_paths, n_steps + 1)}"
        )
    return np.asarray(times, dtype=float), S


# =============================================================================
# Option value table builders
# =============================================================================

def _bs_european_value(
    S_j: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    tau: float,
    is_call: bool,
) -> np.ndarray:
    """
    Black-Scholes European option value for a vector of spot prices S_j,
    with remaining time tau = T - t_j.

    When tau == 0 (terminal column) this reduces to the intrinsic payoff,
    matching the MC pricer exactly.  For tau > 0 it gives the true
    risk-neutral continuation value — which is what the EU pricer targets —
    rather than the intrinsic at that node.

    Formula:
        d1 = (ln(S/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau))
        d2 = d1 - sigma*sqrt(tau)
        Call = S*N(d1) - K*exp(-r*tau)*N(d2)
        Put  = K*exp(-r*tau)*N(-d2) - S*N(-d1)
    """
    from scipy.stats import norm

    S_j = np.asarray(S_j, dtype=float)

    if tau <= 0.0:                              # terminal node — return intrinsic
        if is_call:
            return np.maximum(S_j - K, 0.0)
        return np.maximum(K - S_j, 0.0)

    sqrt_tau: float = float(np.sqrt(tau))
    d1 = (np.log(S_j / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    disc: float = float(np.exp(-r * tau))

    if is_call:
        return S_j * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S_j * norm.cdf(-d1)


def _european_value_table(
    trade: OptionTrade,
    market: Market,
    S: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """
    European option value table — shape (N, M+1).

    Each cell (i, j) is the Black-Scholes value of the option at time t_j
    given spot S[i, j] and remaining time tau = T - t_j:

        V[i, j] = BS(S[i,j], K, r, sigma, tau=T-t_j)

    This is the correct value the EU pricer is estimating at each node.
    It is NOT the intrinsic payoff max(S-K, 0) — for a European option,
    intermediate intrinsic values are meaningless since early exercise is
    not allowed.

    At j = M (terminal column), tau = 0 so BS reduces to intrinsic,
    consistent with the MC pricer terminal condition.

    NOTE on consistency with price_european_naive_mc_vector:
      - No dividend: the pricer collapses to a single exp of W_T; it never
        looks at intermediate steps.  The BS value here is the theoretical
        fair value at each intermediate node — not computed by the pricer,
        but the quantity the pricer converges to at t0.
      - With dividend: the pricer builds full paths; intermediate nodes are
        used. The BS formula here ignores the dividend for simplicity; use
        with caution in that case.
    """
    r: float = float(market.r)
    sigma: float = float(market.sigma)
    K: float = float(trade.strike)
    T: float = float(trade.T)
    dt: float = T / n_steps

    V = np.empty_like(S)
    for j in range(n_steps + 1):
        tau: float = T - j * dt                 # remaining time at step j
        V[:, j] = _bs_european_value(S[:, j], K, r, sigma, tau, trade.is_call)

    return V


def _american_ls_value_table(
    trade: OptionTrade,
    market: Market,
    S: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, float]:
    """
    American option value table via naive backward recursion — shape (N, M+1).

        V[:, M] = payoff(S[:, M])
        V[:, j] = max(payoff(S[:, j]),  DF * V[:, j+1])

    Returns (V_table, df_step).
    This is the slide diagnostic: it shows at each node whether immediate
    exercise dominates holding, which is what makes the AM vs EU comparison
    meaningful.
    """
    r: float = float(market.r)
    dt: float = float(trade.T) / n_steps
    df: float = float(np.exp(-r * dt))

    IV = trade.payoff_vector(S)     # intrinsic value at every node, shape (N, M+1)
    V = IV.copy()

    for j in range(n_steps - 1, -1, -1):
        V[:, j] = np.maximum(IV[:, j], df * V[:, j + 1])

    return V, df


# =============================================================================
# Sheet writers
# =============================================================================

def _write_option_sheet(
    wb: xw.Book,
    sheet_name: str,
    after_sheet: xw.Sheet,
    title: str,
    V_table: np.ndarray,
    disc_to_t0: np.ndarray,
    df_step: float,
    n_paths: int,
    n_steps: int,
    params_block: list,
) -> xw.Sheet:
    """
    Write an option value sheet with rows:
      - one row per path
      - Average (at t_j)
      - Expected (prev avg * DF)   — shows what DF*E[V_{j-1}] predicts
      - Disc Avg (to t0)           — avg * DF^j, comparable across steps
    """
    if sheet_name in [s.name for s in wb.sheets]:
        sht = wb.sheets[sheet_name]
        sht.clear()
    else:
        sht = wb.sheets.add(sheet_name, after=after_sheet)

    sht.range("A1").value = title
    sht.range("A3").value = params_block

    top = 9
    left = 2
    headers = ["t0"] + [f"Step {j}" for j in range(1, n_steps + 1)]
    sht.range((top, left)).value = [""] + headers

    # Path rows
    sht.range((top + 1, left)).value = [[f"path {i}"] for i in range(n_paths)]
    _write_matrix_in_chunks(sht, top + 1, left + 1, V_table)

    # Summary rows
    avg_V = V_table.mean(axis=0)
    expected_V = np.empty_like(avg_V)
    expected_V[0] = np.nan
    expected_V[1:] = avg_V[:-1] * df_step          # DF * E[V_{j-1}]
    avg_V_disc0 = avg_V * disc_to_t0               # E[V_j] * DF^j

    avg_row = top + 1 + n_paths
    exp_row = avg_row + 1
    disc_row = exp_row + 1

    sht.range((avg_row, left)).value = ["Average"] + list(avg_V)
    sht.range((exp_row, left)).value = ["Expected"] + list(expected_V)
    sht.range((disc_row, left)).value = ["Disc Avg (to t0)"] + list(avg_V_disc0)

    # Formatting
    fmt = "0.000000"
    sht.range((top + 1, left + 1)).resize(n_paths, n_steps + 1).number_format = fmt
    sht.range((avg_row, left + 1)).resize(3, n_steps + 1).number_format = fmt
    sht.autofit()

    return sht


def _write_spot_sheet(
    wb: xw.Book,
    sheet_name: str,
    after_sheet: xw.Sheet,
    S: np.ndarray,
    disc_to_t0: np.ndarray,
    df_step: float,
    n_paths: int,
    n_steps: int,
    trade: OptionTrade,
    div_amount: float,
) -> xw.Sheet:
    """
    Write the spot path sheet with rows:
      - one row per path
      - Average S (at t_j)
      - Disc Avg S (to t0)   — avg * DF^j
    Shared between EU and AM (same paths).
    """
    if sheet_name in [s.name for s in wb.sheets]:
        sS = wb.sheets[sheet_name]
        sS.clear()
    else:
        sS = wb.sheets.add(sheet_name, after=after_sheet)

    sS.range("A1").value = "Spot paths S (same paths used by both EU and AM pricers)"
    sS.range("A2").value = [
        ["div_time", trade.ex_div_time()],
        ["div_amount", div_amount],
        ["DF(step)", df_step],
    ]

    top = 6
    left = 2
    headers = ["t0"] + [f"Step {j}" for j in range(1, n_steps + 1)]
    sS.range((top, left)).value = [""] + headers

    sS.range((top + 1, left)).value = [[f"path {i}"] for i in range(n_paths)]
    _write_matrix_in_chunks(sS, top + 1, left + 1, S)

    avg_S = S.mean(axis=0)
    avg_S_disc0 = avg_S * disc_to_t0

    avg_row = top + 1 + n_paths
    disc_row = avg_row + 1
    sS.range((avg_row, left)).value = ["Average S"] + list(avg_S)
    sS.range((disc_row, left)).value = ["Disc Avg S (to t0)"] + list(avg_S_disc0)

    fmt = "0.000000"
    sS.range((top + 1, left + 1)).resize(n_paths, n_steps + 1).number_format = fmt
    sS.range((avg_row, left + 1)).resize(2, n_steps + 1).number_format = fmt
    sS.autofit()

    return sS



def _write_disc_avg_chart(
    main: xw.Sheet,
    n_steps: int,
    disc_avg_V: np.ndarray,
    disc_avg_S: np.ndarray,
    is_american: bool,
    anchor_cell: str = "A15",
) -> None:
    """
    Write the three Disc Avg series to a staging area on Main, then build
    an Excel line chart overlaying them on the same axes.

    Series written (M+1 values, one per step):
      - Disc Avg V  : E[V_tj] * DF^j   (option value, EU or AM)
      - Disc Avg S  : E[S_tj] * DF^j   (spot price discounted to t0)

    The staging area starts at anchor_cell and is cleared on each run.
    The chart is placed directly below the staging area on Main.
    """
    wb = main.book
    steps = list(range(n_steps + 1))
    label_V = "Disc Avg V (AM)" if is_american else "Disc Avg V (EU)"

    # ===== Write staging data (header + 1 data row per series)
    anchor = main.range(anchor_cell)
    top_row = anchor.row
    left_col = anchor.column

    # Clear previous staging area (generous range)
    main.range((top_row, left_col)).resize(5, n_steps + 2).clear()

    main.range((top_row,     left_col)).value = ["Step"] + steps
    main.range((top_row + 1, left_col)).value = [label_V] + list(disc_avg_V)
    main.range((top_row + 2, left_col)).value = ["Disc Avg S"] + list(disc_avg_S)

    # Number format for data rows
    main.range((top_row + 1, left_col + 1)).resize(2, n_steps + 1).number_format = "0.0000"

    # ===== Build chart
    chart_name = "DiscAvgChart"

    # Remove existing chart if present
    for ch in main.charts:
        if ch.name == chart_name:
            ch.delete()
            break

    # Place chart below the staging area
    chart_top = main.range((top_row + 4, left_col)).top
    chart_left = main.range((top_row + 4, left_col)).left
    chart = main.charts.add(
        left=chart_left,
        top=chart_top,
        width=560,
        height=300,
    )
    chart.name = chart_name
    chart.chart_type = "line"

    # Add series: Disc Avg V
    x_rng = main.range((top_row, left_col + 1)).resize(1, n_steps + 1)
    y_V   = main.range((top_row + 1, left_col + 1)).resize(1, n_steps + 1)
    y_S   = main.range((top_row + 2, left_col + 1)).resize(1, n_steps + 1)

    chart.set_source_data(y_V)          # seed with first series

    # Use the Excel API to set names and add the second series
    try:
        api = chart.api[1]              # xlwings Chart COM object (index 1 on Mac/Win)
        api.SeriesCollection(1).Name = f"={main.range((top_row + 1, left_col)).get_address(True, True, True)}"
        api.SeriesCollection(1).XValues = x_rng.api

        s2 = api.SeriesCollection().NewSeries()
        s2.Name   = f"={main.range((top_row + 2, left_col)).get_address(True, True, True)}"
        s2.Values = y_S.api
        s2.XValues = x_rng.api

        # Title and axes
        api.HasTitle = True
        api.ChartTitle.Text = "Discounted Average — option value vs spot (to t0)"
        api.Axes(1).HasTitle = True
        api.Axes(1).AxisTitle.Text = "Step"
        api.Axes(2).HasTitle = True
        api.Axes(2).AxisTitle.Text = "Disc Avg (t0)"
    except Exception as e:
        print(f"Chart API styling skipped: {e}")


# =============================================================================
# Main entry point
# =============================================================================

def _cell(main: xw.Sheet, named: str, fallback: str):
    """
    Try to read a named range; if it does not exist fall back to a direct
    cell address.  Prevents crashes when a named range has not yet been
    defined in the workbook.
    """
    try:
        return main.range(named).value
    except Exception:
        return main.range(fallback).value


def _read_inputs(main: xw.Sheet) -> dict:
    """
    Read all inputs from the Main sheet.
    Each field tries its named range first, then falls back to the hard-coded
    cell address matching the current sheet layout.

    Named range   cell   description
    -----------   ----   -----------
    Spot          B4     Spot price S0
    Vol           B5     Volatility sigma
    Rate          B6     Risk-free rate r
    Dividende     B7     Discrete dividend amount
    Q             B8     Continuous dividend yield q
    Date_Div      B9     Ex-dividend date
    Strike        E4     Strike K
    is_call       E5     Call or Put
    Exercice      E6     US / EU  (new — add named range or leave at E6)
    Date_pricing  E7     Pricing date
    Mat           E8     Maturity date
    n_steps       H4     Number of time steps
    n_paths       H5     Number of simulated paths
    degree        H6     Polynomial degree for LS regression
    seed          H7     RNG seed
    Antithetic    H8     Antithetic variates flag
    """
    exercise_raw = str(_cell(main, "Exercice", "E6") or "US").strip().upper()
    is_american: bool = exercise_raw in ("US", "AM", "AMERICAN")

    return {
        "S0":           float(_cell(main, "Spot",         "B4") or 0),
        "sigma":        float(_cell(main, "Vol",          "B5") or 0),
        "r":            float(_cell(main, "Rate",         "B6") or 0),
        "div_amount":   float(_cell(main, "Dividende",    "B7") or 0.0),
        "q":            float(_cell(main, "Q",            "B8") or 0.0),
        "ex_div_date":  _cell(main, "Date_Div",           "B9"),
        "K":            float(_cell(main, "Strike",       "E4") or 0),
        "is_call":      "call" in str(_cell(main, "is_call", "E5") or "").strip().lower(),
        "is_american":  is_american,
        "pricing_date": _as_date_excel(_cell(main, "Date_pricing", "E7")),
        "maturity_date":_as_date_excel(_cell(main, "Mat",          "E8")),
        "n_steps":      int(_cell(main, "n_steps",   "H4") or 10),
        "n_paths":      int(_cell(main, "n_paths",   "H5") or 1000),
        "degree":       int(_cell(main, "degree",    "H6") or 3),
        "seed":         int(_cell(main, "seed",      "H7") or 0),
        "antithetic":   _to_bool(_cell(main, "Antithetic", "H8")),
    }


def _make_market_and_trade(inp: dict) -> tuple[Market, OptionTrade]:
    """Build Market and OptionTrade from the inputs dict."""
    market = Market(S0=inp["S0"], r=inp["r"], sigma=inp["sigma"])
    trade = OptionTrade(
        strike=inp["K"],
        is_call=inp["is_call"],
        exercise="american" if inp["is_american"] else "european",
        pricing_date=inp["pricing_date"],
        maturity_date=inp["maturity_date"],
        q=inp["q"],
        ex_div_date=(
            _as_date_excel(inp["ex_div_date"]) if inp["ex_div_date"] is not None else None
        ),
        div_amount=inp["div_amount"],
    )
    return market, trade


def _run_pricer(
    inp: dict,
    market: Market,
    trade: OptionTrade,
    main: xw.Sheet,
) -> float:
    """
    Run the pricer selected by the Exercice cell:
      - US / AM  →  price_american_ls_vector   (written to Price_AM in Main)
      - EU       →  price_european_naive_mc_vector (written to Price_EU in Main)
    Returns the scalar price.
    """
    kwargs = dict(
        market=market, trade=trade,
        n_paths=inp["n_paths"], n_steps=inp["n_steps"],
        seed=inp["seed"], antithetic=inp["antithetic"],
    )

    if inp["is_american"]:
        price, _ = price_american_ls_vector(**kwargs, basis="laguerre", degree=inp["degree"])
        main.range("B12").clear_contents()
        main.range("B12").value = float(price)
    else:
        price, _ = price_european_naive_mc_vector(**kwargs)
        main.range("B12").clear_contents()
        main.range("B12").value = float(price)

    return float(price)


def build_early_exercise_excel_table_all_from_ls_paths():
    """
    Writes 2 or 3 sheets depending on the Exercice cell in Main:

      EU  →  EU_OptionPaths  +  SpotPaths
      US  →  AM_OptionPaths  +  SpotPaths

    Both cases use price_european_naive_mc_vector or price_american_ls_vector
    respectively, then expose the per-step average option value and spot price
    (raw + discounted to t0) for diagnostic comparison.
    """
    wb = xw.Book.caller()
    main = wb.sheets["Main"]

    # ===== Read inputs and build market/trade objects
    inp = _read_inputs(main)
    market, trade = _make_market_and_trade(inp)

    n_steps: int = inp["n_steps"]
    n_paths: int = inp["n_paths"]

    # ===== Run the pricer selected by Exercice cell (also writes price to Main)
    _run_pricer(inp, market, trade, main)

    # ===== Retrieve the paths produced by the pricer (LAST_PATHS set by simulator)
    times, S = _get_or_simulate_paths(
        market, trade, n_paths, n_steps, inp["seed"], inp["antithetic"]
    )

    # ===== Shared discount factors
    dt_step: float = float(trade.T) / n_steps
    df_step: float = float(np.exp(-inp["r"] * dt_step))
    disc_to_t0 = df_step ** np.arange(0, n_steps + 1, dtype=float)

    # ===== Shared params block
    params = [
        ["S0", inp["S0"], "K", inp["K"], "Type", "Call" if inp["is_call"] else "Put"],
        ["r", inp["r"], "sigma", inp["sigma"], "q", inp["q"]],
        ["div_amount", inp["div_amount"], "ex_div_date", str(trade.ex_div_date), "div_time", trade.ex_div_time()],
        ["T", float(trade.T), "n_steps", n_steps, "DF(step)", df_step],
        ["n_paths", n_paths, "seed", inp["seed"], "antithetic", inp["antithetic"]],
    ]

    # ===== Build option value table and write the appropriate sheet
    if inp["is_american"]:
        V_table, _ = _american_ls_value_table(trade, market, S, n_steps)
        sht_opt = _write_option_sheet(
            wb, "AM_OptionPaths", main,
            title="AMERICAN OPTION (LS) — Average value per step (naive backward recursion)",
            V_table=V_table, disc_to_t0=disc_to_t0, df_step=df_step,
            n_paths=n_paths, n_steps=n_steps, params_block=params,
        )
    else:
        V_table = _european_value_table(trade, market, S, n_steps)
        sht_opt = _write_option_sheet(
            wb, "EU_OptionPaths", main,
            title="EUROPEAN OPTION — Average value per step (intrinsic at each node)",
            V_table=V_table, disc_to_t0=disc_to_t0, df_step=df_step,
            n_paths=n_paths, n_steps=n_steps, params_block=params,
        )

    # ===== Always write the shared spot paths sheet
    _write_spot_sheet(
        wb, "SpotPaths", sht_opt,
        S=S, disc_to_t0=disc_to_t0, df_step=df_step,
        n_paths=n_paths, n_steps=n_steps,
        trade=trade, div_amount=inp["div_amount"],
    )

    # ===== Compute Disc Avg series for the chart
    disc_avg_V = V_table.mean(axis=0) * disc_to_t0    # E[V_tj] * DF^j
    disc_avg_S = S.mean(axis=0) * disc_to_t0          # E[S_tj] * DF^j

    # ===== Write chart on Main sheet
    _write_disc_avg_chart(
        main=main,
        n_steps=n_steps,
        disc_avg_V=disc_avg_V,
        disc_avg_S=disc_avg_S,
        is_american=inp["is_american"],
        anchor_cell="A15",              # staging data starts here; chart placed below
    )


# Excel call:
# =RunPython("import excel_early_exercise; excel_early_exercise.build_early_exercise_excel_table_all_from_ls_paths()")