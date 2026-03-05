import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import xlwings as xw
import datetime as dt

from model.market import Market
from model.option import OptionTrade
from model.mc_pricer import price_american_ls_vector


# =========================
# Helpers
# =========================
def _read_date(x):
    if x is None:
        return None
    if hasattr(x, "date"):
        return x.date()
    return x

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "vrai", "1", "yes", "y")
    return False

def _get(sh, addr, default=None):
    v = sh.range(addr).value
    return default if v is None else v


# =========================
# 1) Run analysis: 20 seeds + write averages
# =========================
def run_degree_seed_analysis_xlwings(n_seeds: int = 20, seed_start: int = 1, degrees=None):
    """
    xlwings version:
    - Reads params from MAIN
    - Computes prices for n_seeds (default 20) and degrees (default 1..7)
    - Writes Power and Laguerre tables
    - Writes Average row (mean across the n_seeds) for each degree & basis
    """
    wb = xw.Book.caller()
    sh_main = wb.sheets["MAIN"]
    sh_out = wb.sheets["DegreeAnalysis"]

    if degrees is None:
        degrees = list(range(1, 8))  # 1..7

    seeds = list(range(seed_start, seed_start + n_seeds))

    # ---- read MAIN inputs (your addresses) ----
    S0 = float(_get(sh_main, "B4"))
    sigma = float(_get(sh_main, "B5"))
    r = float(_get(sh_main, "B6"))
    div_amount = float(_get(sh_main, "B7", 0.0))
    q = float(_get(sh_main, "B8", 0.0))
    ex_div_date = _read_date(_get(sh_main, "B9", None))

    strike = float(_get(sh_main, "E4"))
    callput = str(_get(sh_main, "E5")).strip().lower()
    is_call = callput.startswith("c")
    exercise = str(_get(sh_main, "E6")).strip().lower()
    exercise = "american" if exercise in ("us", "american", "am") else exercise

    pricing_date = _read_date(_get(sh_main, "E7"))
    maturity_date = _read_date(_get(sh_main, "E8"))

    n_steps = int(_get(sh_main, "H4"))
    n_paths = int(_get(sh_main, "H5"))
    antithetic = _to_bool(_get(sh_main, "H8", False))

    market = Market(S0=S0, r=r, sigma=sigma)
    trade = OptionTrade(
        strike=strike,
        is_call=is_call,
        exercise=exercise,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=q,
        ex_div_date=ex_div_date,
        div_amount=div_amount,
    )

    if trade.exercise.lower() != "american":
        raise ValueError("This table is designed for American options (LSM).")

    # ---- output layout (your sheet) ----
    header_row = 6
    first_seed_row = 8
    col_seed = 1        # A
    col_power = 2       # B
    col_lag = 10        # J

    n_rows = len(seeds)
    n_cols = len(degrees)

    # headers degrees
    sh_out.range((header_row, col_power), (header_row, col_power + n_cols - 1)).value = degrees
    sh_out.range((header_row, col_lag), (header_row, col_lag + n_cols - 1)).value = degrees

    # seeds column
    sh_out.range((first_seed_row, col_seed), (first_seed_row + n_rows - 1, col_seed)).value = [[s] for s in seeds]

    # clear only used area
    sh_out.range((first_seed_row, col_power), (first_seed_row + n_rows - 1, col_power + n_cols - 1)).value = None
    sh_out.range((first_seed_row, col_lag), (first_seed_row + n_rows - 1, col_lag + n_cols - 1)).value = None

    # ---- compute ----
    app = wb.app
    old_status = app.status_bar

    try:
        for i, seed in enumerate(seeds):
            app.status_bar = f"LSM DegreeAnalysis: seed {seed} ({i+1}/{n_rows})"

            power_row = []
            lag_row = []

            for deg in degrees:
                price_pow, _ = price_american_ls_vector(
                    market=market, trade=trade,
                    n_paths=n_paths, n_steps=n_steps,
                    seed=seed, antithetic=antithetic,
                    basis="power", degree=deg
                )
                power_row.append(float(price_pow))

                price_lag, _ = price_american_ls_vector(
                    market=market, trade=trade,
                    n_paths=n_paths, n_steps=n_steps,
                    seed=seed, antithetic=antithetic,
                    basis="laguerre", degree=deg
                )
                lag_row.append(float(price_lag))

            row = first_seed_row + i
            sh_out.range((row, col_power)).value = [power_row]
            sh_out.range((row, col_lag)).value = [lag_row]
    finally:
        app.status_bar = old_status

    # ---- averages row ----
    power = np.array(
        sh_out.range((first_seed_row, col_power),
                     (first_seed_row + n_rows - 1, col_power + n_cols - 1)).value,
        dtype=float
    )
    lag = np.array(
        sh_out.range((first_seed_row, col_lag),
                     (first_seed_row + n_rows - 1, col_lag + n_cols - 1)).value,
        dtype=float
    )
    mean_power = power.mean(axis=0)
    mean_lag = lag.mean(axis=0)

    avg_row = first_seed_row + n_rows
    sh_out.range((avg_row, col_seed)).value = "Average"
    sh_out.range((avg_row, col_power)).value = [mean_power.tolist()]
    sh_out.range((avg_row, col_lag)).value = [mean_lag.tolist()]

    return {
        "avg_row": avg_row,
        "n_seeds": n_seeds,
        "seed_start": seed_start,
        "degrees": degrees,
    }


# =========================
# 2) Plot chosen seed (from named ranges) -> insert into Excel
# =========================
def plot_chosen_seed_to_excel(
    sheet_name: str = "DegreeAnalysis",
    degrees_range_addr: str = "B6:H6",
    seed_cell: str = "C3",
    anchor_cell: str = "R3",
):
    """
    Requires named ranges already created:
      - Power_Selected
      - Laguerre_Selected
    Plots them and inserts a picture into Excel.
    """
    wb = xw.Book.caller()
    sh = wb.sheets[sheet_name]

    degrees = np.array(sh.range(degrees_range_addr).value, dtype=float)
    power = np.array(sh.range("Power_Selected").value, dtype=float).reshape(-1)
    lag = np.array(sh.range("Laguerre_Selected").value, dtype=float).reshape(-1)

    seed = sh.range(seed_cell).value
    seed_txt = f"{int(seed)}" if seed is not None else "?"

    y_all = np.concatenate([power, lag])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    pad = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()
    ax.plot(degrees, power, marker="o", linewidth=2, label="Power")
    ax.plot(degrees, lag, marker="s", linewidth=2, label="Laguerre")

    ax.set_title(f"LSM Price vs Degree (Seed = {seed_txt})")
    ax.set_xlabel("Regression Degree")
    ax.set_ylabel("Option Price")
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    img_path = os.path.join(tempfile.gettempdir(), "lsm_plot_seed.png")
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    pic_name = "LSM_PLOT_SEED"
    for p in sh.pictures:
        if p.name == pic_name:
            p.delete()

    anchor = sh.range(anchor_cell)
    sh.pictures.add(img_path, name=pic_name, left=anchor.left, top=anchor.top, width=520, height=320)


# =========================
# 3) Plot averages row -> insert into Excel
# =========================
def plot_average_to_excel(
    avg_row: int,
    sheet_name: str = "DegreeAnalysis",
    degrees_range_addr: str = "B6:H6",
    anchor_cell: str = "R20",
):
    wb = xw.Book.caller()
    sh = wb.sheets[sheet_name]

    degrees = np.array(sh.range(degrees_range_addr).value, dtype=float)

    mean_power = np.array(sh.range((avg_row, 2), (avg_row, 8)).value, dtype=float)   # B..H
    mean_lag = np.array(sh.range((avg_row, 10), (avg_row, 16)).value, dtype=float)   # J..P

    y_all = np.concatenate([mean_power, mean_lag])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    pad = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()
    ax.plot(degrees, mean_power, marker="o", linewidth=2, label="Power (mean)")
    ax.plot(degrees, mean_lag, marker="s", linewidth=2, label="Laguerre (mean)")

    ax.set_title("Average LSM Price vs Degree")
    ax.set_xlabel("Regression Degree")
    ax.set_ylabel("Option Price")
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    img_path = os.path.join(tempfile.gettempdir(), "lsm_plot_mean.png")
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    pic_name = "LSM_PLOT_MEAN"
    for p in sh.pictures:
        if p.name == pic_name:
            p.delete()

    anchor = sh.range(anchor_cell)
    sh.pictures.add(img_path, name=pic_name, left=anchor.left, top=anchor.top, width=520, height=320)


# =========================
# 4) Convenience: run all in one click
# =========================
def run_all(n_seeds: int = 20, seed_start: int = 1):
    info = run_degree_seed_analysis_xlwings(n_seeds=n_seeds, seed_start=seed_start)
    plot_chosen_seed_to_excel()                 # uses named ranges
    plot_average_to_excel(avg_row=info["avg_row"])