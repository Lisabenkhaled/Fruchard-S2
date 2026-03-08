import sys
import os
import numpy as np
from typing import Any
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# imports from project 
from core_pricer import input_parameters
from analysis.greeks import compute_method_greeks
from utils.utils_bs import bs_greeks
from utils.utils_sheet import ensure_sheet

# constants
HEADER_ROW = 3
DATA_START_ROW = 4
HEADER_START_CELL = f"A{HEADER_ROW}"
DATA_START_CELL = f"A{DATA_START_ROW}"

#build headers
def _build_headers() -> list[str]:
    return [
        "Interest Rate",
        "Delta Tree", "Delta BS",
        "Gamma Tree", "Gamma BS",
        "Vega Tree", "Vega BS",
        "Theta Tree", "Theta BS",
        "Rho Tree", "Rho BS",
        "Vanna Tree", "Vanna BS",
        "Vomma Tree", "Vomma BS",
    ]

#data rows
def _compute_data_rows(
    market: Any,option: Any,n_steps: int,exercise: str,
    optimize: bool,threshold: float,spot: float,
    strike: float, sigma: float,maturity: float,
    is_call: bool) -> tuple[np.ndarray, list[list[Any]]]:
    rate_values = np.linspace(-0.1, 0.10, 20)
    data: list[list[Any]] = []
    # boucle for on rates
    for rate_test in rate_values:
        market.r = rate_test
        tree_greeks = compute_method_greeks(market, option, n_steps, exercise, optimize, threshold, "backward")
        bs_vals = bs_greeks(spot, strike, rate_test, sigma, maturity, is_call)  
        data.append([
            rate_test,tree_greeks["Delta"], bs_vals["Delta"],
            tree_greeks["Gamma"], bs_vals["Gamma"],tree_greeks["Vega"], bs_vals["Vega"],
            tree_greeks["Theta"], bs_vals["Theta"],tree_greeks["Rho"], bs_vals["Rho"],
            tree_greeks["Vanna"], bs_vals["Vanna"],tree_greeks["Vomma"], bs_vals["Vomma"]
        ])

    #round
    rounded_data = [[round(x, 4) if isinstance(x, (float, np.floating)) else x for x in row] for row in data]
    return rate_values, rounded_data

# charts
def _add_charts(sh: Any, data: list[list[Any]], rate_values: np.ndarray) -> None:
    chart_specs = [("Delta", 2, 3, "green", 0, 0),
        ("Theta", 8, 9, "red", 1, 0),("Vega", 6, 7, "orange", 2, 0),
        ("Rho", 10, 11, "purple", 0, 1), ("Gamma", 4, 5, "blue", 1, 1),
        ("Vomma", 14, 15, "gold", 2, 1),("Vanna", 12, 13, "black", 0, 2)]

    chart_width = 360;chart_height = 250
    x_spacing = 50;y_spacing = 60
    x_start = 1050;y_start = 50

    for name, col_tree, col_bs, color, grid_x, grid_y in chart_specs:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        tree_vals = [row[col_tree - 1] for row in data]
        bs_vals = [row[col_bs - 1] for row in data]

        ax.plot(rate_values * 100, bs_vals, color=color, linestyle="--", linewidth=1.5, label=f"{name} (BS)")
        ax.plot(rate_values * 100, tree_vals, color=color, linewidth=1.5, label=f"{name} (Tree)")
        # title and labels
        ax.set_title(f"{name} vs Taux d'intérêt (Tree vs BS)", fontsize=10)
        ax.set_xlabel("Taux (%)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

        left = x_start + grid_x * (chart_width + x_spacing)
        top = y_start + grid_y * (chart_height + y_spacing)

        sh.pictures.add(fig,name=f"Chart_{name}",update=True,
            left=left,top=top, width=chart_width,height=chart_height)
        plt.close(fig)
#rate test 
def rate_test() -> None:
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    headers = _build_headers()
    rate_values, data = _compute_data_rows(
        market, option, N, exercise, optimize, threshold, S0, K, sigma, T, is_call
    )

    sheet_name = "Greeks Rate"
    sh = ensure_sheet(wb, sheet_name)
    sh.range(HEADER_START_CELL).value = headers
    sh.range(DATA_START_CELL).value = data

    _add_charts(sh, data, rate_values)
    # autofit
    sh.autofit()

def run_test_greeks_rate() -> None:
    rate_test()

if __name__ == "__main__":
    run_test_greeks_rate()
