# excel_early_exercise.py
import numpy as np
import xlwings as xw

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_american_ls_vector

import model.path_simulator as path_sim
from model.path_simulator import simulate_gbm_paths_vector


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


def _american_naive_table_from_paths(trade: OptionTrade, market: Market, S: np.ndarray, n_steps: int) -> tuple[np.ndarray, float]:
    """
    Slide diagnostic (naive American recursion) on given paths:
      V[:,M] = payoff(S[:,M])
      V[:,j] = max( payoff(S[:,j]), DF * V[:,j+1] )
    Returns (V_table, DF).
    """
    r = float(market.r)
    dt = float(trade.T) / n_steps
    df = float(np.exp(-r * dt))

    IV = trade.payoff_vector(S)  # (n_paths, n_steps+1)
    V = IV.copy()

    for j in range(n_steps - 1, -1, -1):
        V[:, j] = np.maximum(IV[:, j], df * V[:, j + 1])

    return V, df


def _write_matrix_in_chunks(sht: xw.Sheet, top_row: int, left_col: int, M: np.ndarray, chunk_rows: int = 500):
    n_rows, _ = M.shape
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        sht.range((top_row + r0, left_col)).value = M[r0:r1, :].tolist()


def build_early_exercise_excel_table_all_from_ls_paths():
    """
    Outputs 2 sheets:
      1) EarlyExercises: option value table V + Average + Expected + DiscAvg(t0) + highlight (slide diagnostic)
      2) SpotPaths: spot paths S + Average S + DiscAvg S (t0)

    Uses EXACT same paths as LS pricer via path_sim.LAST_PATHS (Option 2),
    with robust fallback if LAST_* isn't set due to early return branches.
    """
    wb = xw.Book.caller()
    main = wb.sheets["Main"]

    # ===== inputs (your template)
    S0 = float(main.range("B4").value)
    sigma = float(main.range("B5").value)
    r = float(main.range("B6").value)

    div_amount = float(main.range("B7").value or 0.0)
    q = float(main.range("B8").value or 0.0)
    ex_div_date = main.range("B9").value

    K = float(main.range("E4").value)
    callput = str(main.range("E5").value).strip().lower()
    is_call = "call" in callput

    pricing_date = _as_date_excel(main.range("E7").value)
    maturity_date = _as_date_excel(main.range("E8").value)

    n_steps = int(main.range("H4").value)
    n_paths = int(main.range("H5").value)
    degree = int(main.range("H6").value)
    seed = int(main.range("H7").value)
    antithetic = _to_bool(main.range("H8").value)

    market = Market(S0=S0, r=r, sigma=sigma)
    trade = OptionTrade(
        strike=K,
        is_call=is_call,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=q,
        ex_div_date=(_as_date_excel(ex_div_date) if ex_div_date is not None else None),
        div_amount=div_amount,
    )

    # ===== 1) run LS once (populates LAST_PATHS if simulator sets it)
    price, discounted_cf = price_american_ls_vector(
        market=market,
        trade=trade,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
        basis="laguerre",
        degree=degree,
    )

    main.range("B12").value = "LS Price"
    main.range("C12").value = float(price)

    # ===== 2) retrieve SAME paths (robust fallback if LAST_* not set due to early return branch)
    S = getattr(path_sim, "LAST_PATHS", None)
    times = getattr(path_sim, "LAST_TIMES", None)

    if S is None or times is None:
        times, S = simulate_gbm_paths_vector(
            market=market,
            trade=trade,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
        )

    S = np.asarray(S, dtype=float)
    if S.shape != (n_paths, n_steps + 1):
        raise RuntimeError(f"Paths shape mismatch: got {S.shape}, expected {(n_paths, n_steps+1)}")

    # ===== discount factor per step, and discount-to-t0 factors
    dt_step = float(trade.T) / n_steps
    DF = float(np.exp(-float(market.r) * dt_step))
    disc_to_t0 = DF ** np.arange(0, n_steps + 1, dtype=float)  # [1, DF, DF^2, ...]

    # ===== Spot averages (and discounted to t0)
    avg_S = S.mean(axis=0)                 # E[S_tj]
    avg_S_disc0 = avg_S * disc_to_t0       # E[ DF^j * S_tj ]

    # ===== 3) build slide diagnostic option value table on these paths
    V_table, DF_check = _american_naive_table_from_paths(trade, market, S, n_steps)

    avg_V = V_table.mean(axis=0)                 # E[V_tj]
    avg_V_disc0 = avg_V * disc_to_t0             # E[ DF^j * V_tj ]  (actualisé jusqu'au début)

    expected_V = np.empty_like(avg_V)
    expected_V[0] = np.nan
    expected_V[1:] = avg_V[:-1] * DF_check       # “Expected” row like slide (prev avg * DF)

    # ===== Output sheet: EarlyExercises
    sheet_name = "EarlyExercises"
    if sheet_name in [s.name for s in wb.sheets]:
        sht = wb.sheets[sheet_name]
        sht.clear()
    else:
        sht = wb.sheets.add(sheet_name, after=main)

    sht.range("A1").value = "OPTION PRICING MONTE CARLO - EARLY EXERCISES"

    sht.range("A3").value = [
        ["S0", S0, "K", K, "Type", "Call" if is_call else "Put"],
        ["r", r, "sigma", sigma, "q", q],
        ["div_amount", div_amount, "ex_div_date", str(trade.ex_div_date), "div_time", trade.ex_div_time()],
        ["T", float(trade.T), "n_steps", n_steps, "DF(step)", DF_check],
        ["n_paths", n_paths, "seed", seed, "antithetic", antithetic],
        ["NOTE", "Rows: Average (at t_j), Expected (prev avg * DF), DiscAvg(t0)=Avg*DF^j", "", "", "", ""],
    ]

    top = 9
    left = 2  # column B
    headers = ["t0"] + [f"Step {j}" for j in range(1, n_steps + 1)]
    sht.range((top, left)).value = [""] + headers

    # path labels + values
    sht.range((top + 1, left)).value = [[f"path {i}"] for i in range(n_paths)]
    _write_matrix_in_chunks(sht, top + 1, left + 1, V_table, chunk_rows=500)

    # Rows under the table
    avg_row = top + 1 + n_paths
    exp_row = avg_row + 1
    disc_row = exp_row + 1

    sht.range((avg_row, left)).value = ["Average"] + list(avg_V)
    sht.range((exp_row, left)).value = ["Expected"] + list(expected_V)
    sht.range((disc_row, left)).value = ["Disc Avg (to t0)"] + list(avg_V_disc0)

    # formatting
    sht.range((top + 1, left + 1)).resize(n_paths, n_steps + 1).number_format = "0.000000"
    sht.range((avg_row, left + 1)).resize(1, n_steps + 1).number_format = "0.000000"
    sht.range((exp_row, left + 1)).resize(1, n_steps + 1).number_format = "0.000000"
    sht.range((disc_row, left + 1)).resize(1, n_steps + 1).number_format = "0.000000"

    # conditional formatting: highlight where abs(V_j - DF * V_{j+1}) > eps
    eps = 1e-8
    sht.range("F5").value = DF_check  # fixed DF cell for CF formula
    cf_rng = sht.range((top + 1, left + 1)).resize(n_paths, n_steps)  # exclude last col
    tl = cf_rng.get_address(False, False).split(":")[0]
    formula = f"=ABS({tl}-$F$5*OFFSET({tl},0,1))>{eps}"
    try:
        fc = cf_rng.api.FormatConditions.Add(Type=2, Formula1=formula)
    except Exception as e:
        print("Conditional formatting could not be applied:", e)

    sht.autofit()

    # ===== Output sheet: SpotPaths
    spot_sheet = "SpotPaths"
    if spot_sheet in [s.name for s in wb.sheets]:
        sS = wb.sheets[spot_sheet]
        sS.clear()
    else:
        sS = wb.sheets.add(spot_sheet, after=sht)

    sS.range("A1").value = "Spot paths S (same paths used by LS pricer)"
    sS.range("A2").value = [["div_time", trade.ex_div_time()], ["div_amount", div_amount], ["DF(step)", DF]]

    topS = 6
    leftS = 2
    sS.range((topS, leftS)).value = [""] + headers

    sS.range((topS + 1, leftS)).value = [[f"path {i}"] for i in range(n_paths)]
    _write_matrix_in_chunks(sS, topS + 1, leftS + 1, S, chunk_rows=500)

    avgS_row = topS + 1 + n_paths
    discS_row = avgS_row + 1
    sS.range((avgS_row, leftS)).value = ["Average S"] + list(avg_S)
    sS.range((discS_row, leftS)).value = ["Disc Avg S (to t0)"] + list(avg_S_disc0)

    sS.range((topS + 1, leftS + 1)).resize(n_paths, n_steps + 1).number_format = "0.000000"
    sS.range((avgS_row, leftS + 1)).resize(1, n_steps + 1).number_format = "0.000000"
    sS.range((discS_row, leftS + 1)).resize(1, n_steps + 1).number_format = "0.000000"

    sS.autofit()


# Excel call:
# =RunPython("import excel_early_exercise; excel_early_exercise.build_early_exercise_excel_table_all_from_ls_paths()")