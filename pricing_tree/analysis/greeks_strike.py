import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# imports
from core_pricer import input_parameters
from analysis.greeks import compute_method_greeks
from utils.utils_bs import bs_greeks
from utils.utils_sheet import ensure_sheet
from typing import Any
# constants
HEADER_ROW = 3
DATA_START_ROW = 4
HEADER_START_CELL = f"A{HEADER_ROW}"
DATA_START_CELL = f"A{DATA_START_ROW}"


def _set_status(app: Any, text: Any) -> None:
    try:
        app.status_bar = text
    except Exception:
        pass
# Headers
def _build_headers() -> list[str]:
    return [
        "Strike",
        "Delta Tree", "Delta BS",
        "Gamma Tree", "Gamma BS",
        "Vega Tree", "Vega BS",
        "Theta Tree", "Theta BS",
        "Rho Tree", "Rho BS",
        "Vanna Tree", "Vanna BS",
        "Vomma Tree", "Vomma BS"
    ]


def _compute_data_rows(
    market: Any,option: Any,n_steps: int,exercise: str,
    optimize: bool,threshold: float,spot: float,
    rate: float,sigma: float,maturity: float,
    is_call: bool,app: Any) -> tuple[np.ndarray, list[list[Any]]]:

    K_values = np.linspace(int(0.9 * spot), int(1.1 * spot), 20)
    data = []

    for k in K_values:
        _set_status(app, f"Strike : {float(k):.4f} | N = {int(n_steps)}")
# greeks and bs values
        option.K = k
        tree_greeks = compute_method_greeks(market, option, n_steps, exercise, optimize, threshold, "backward")
        bs_vals = bs_greeks(spot, k, rate, sigma, maturity, is_call)

        _set_status(app, f"Strike : {float(k):.4f} | {int(n_steps)}/{int(n_steps)}")

        # structure intacte (15 colonnes)
        data.append([
            k,
            tree_greeks["Delta"], bs_vals["Delta"],
            tree_greeks["Gamma"], bs_vals["Gamma"],
            tree_greeks["Vega"], bs_vals["Vega"],
            tree_greeks["Theta"], bs_vals["Theta"],
            tree_greeks["Rho"], bs_vals["Rho"],
            tree_greeks["Vanna"], bs_vals["Vanna"],
            tree_greeks["Vomma"], bs_vals["Vomma"]
        ])

 
    rounded_data = [[round(x, 4) if isinstance(x, (float, np.floating)) else x for x in row] for row in data]
    return K_values, rounded_data
# Charts
def _add_charts(sh: Any, data: list[list[Any]], strike_values: np.ndarray) -> None:
    chart_specs = [
        ("Delta", 2, 3, "green", 0, 0), ("Theta", 8, 9, "red", 1, 0),
        ("Vega", 6, 7, "orange", 2, 0), ("Rho", 10, 11, "purple", 0, 1),
        ("Gamma", 4, 5, "blue", 1, 1),  ("Vomma", 14, 15, "gold", 2, 1),
        ("Vanna", 12, 13, "black", 0, 2)
    ]

    chart_width = 360; chart_height = 250
    x_spacing = 50; y_spacing = 60
    x_start = 1050; y_start = 60
# plots vs Strike
    for name, col_tree, col_bs, color, grid_x, grid_y in chart_specs:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        tree_vals = [row[col_tree - 1] for row in data]
        bs_vals = [row[col_bs - 1] for row in data]

        ax.plot(strike_values, bs_vals, color=color, linestyle="--", linewidth=1.5, label=f"{name} (BS)")
        ax.plot(strike_values, tree_vals, color=color, linewidth=1.5, label=f"{name} (Tree)")

        ax.set_title(f"{name} vs Strike (Tree vs BS)", fontsize=10)
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

        left = x_start + grid_x * (chart_width + x_spacing)
        top = y_start + grid_y * (chart_height + y_spacing)

        sh.pictures.add(fig, name=f"Chart_{name}", update=True,
                        left=left, top=top, width=chart_width, height=chart_height)
        plt.close(fig)

# Strikes tests
def strike_test():
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    app = wb.app
    headers = _build_headers()
    K_values, data = _compute_data_rows(
        market, option, N, exercise, optimize, threshold, S0, r, sigma, T, is_call, app
    )

    sheet_name = "Greeks Strike"
    sh = ensure_sheet(wb, sheet_name)
    sh.range(HEADER_START_CELL).value = headers
    sh.range(DATA_START_CELL).value = data
    # sh.range(DATA_START_CELL).expand().number_format = "0.0000"

    _add_charts(sh, data, K_values)
# Autofit
    sh.autofit()
    _set_status(app, False)

def run_test_greeks_strike():
    strike_test()

if __name__ == "__main__":
    run_test_greeks_strike()
