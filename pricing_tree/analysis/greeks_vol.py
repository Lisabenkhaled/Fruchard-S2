import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from numpy.typing import NDArray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from project
from core_pricer import input_parameters
from analysis.greeks import compute_method_greeks
from utils.utils_bs import bs_greeks
from utils.utils_sheet import ensure_sheet

# volatility dataset
def _build_volatility_dataset(market: Any,option: Any,N: int,
    exercise: Any,optimize: bool,threshold: float,S0: float,
    K: float,r: float,T: float,is_call: bool) -> tuple[NDArray[np.float64], list[list[Any]]]:
    vol_values = np.linspace(0.05, 0.50, 20)
    data = []

    # boucle for on vol
    for vol in vol_values:
        market.sigma = vol
        tree_greeks = compute_method_greeks(market, option, N, exercise, optimize, threshold, "backward")
        bs_vals = bs_greeks(S0, K, r, vol, T, is_call)

        data.append([
            vol,
            tree_greeks["Delta"], bs_vals["Delta"],tree_greeks["Gamma"], bs_vals["Gamma"],
            tree_greeks["Vega"], bs_vals["Vega"],tree_greeks["Theta"], bs_vals["Theta"],
            tree_greeks["Rho"], bs_vals["Rho"],tree_greeks["Vanna"], bs_vals["Vanna"],
            tree_greeks["Vomma"], bs_vals["Vomma"]
        ])

    #round
    rounded_data = [[round(x, 4) if isinstance(x, (float, np.floating)) else x for x in row] for row in data]
    return vol_values, rounded_data

# write data for comparison
def _write_data_to_sheet(wb: Any, data: list[list[Any]]) -> Any:
    sh = ensure_sheet(wb, "Greeks Vol")
    start_row, start_col = 3, 1

    headers = ["Volatility","Delta Tree", "Delta BS",
        "Gamma Tree", "Gamma BS","Vega Tree", "Vega BS",
        "Theta Tree", "Theta BS","Rho Tree", "Rho BS",
        "Vanna Tree", "Vanna BS","Vomma Tree", "Vomma BS"]
    
    sh.range((start_row, start_col)).value = headers
    sh.range((start_row + 1, start_col)).value = data
    return sh

# charts
def _plot_greek_charts(sh: Any, vol_values: NDArray[np.float64], data: list[list[Any]]) -> None:
    chart_specs = [("Delta", 2, 3, "green", 0, 0),
        ("Theta", 8, 9, "red", 1, 0),("Vega", 6, 7, "orange", 2, 0),
        ("Rho", 10, 11, "purple", 0, 1),("Gamma", 4, 5, "blue", 1, 1),
        ("Vomma", 14, 15, "gold", 2, 1),("Vanna", 12, 13, "black", 0, 2)]

    chart_width = 360;chart_height = 250
    x_spacing = 50;y_spacing = 60
    x_start = 1050;y_start = 60

    for name, col_tree, col_bs, color, grid_x, grid_y in chart_specs:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        tree_vals = [row[col_tree - 1] for row in data]
        bs_vals = [row[col_bs - 1] for row in data]

        x_values = np.array(vol_values) * 100
        ax.plot(x_values, bs_vals, color=color, linestyle="--", linewidth=1.5, label=f"{name} (BS)")
        ax.plot(x_values, tree_vals, color=color, linewidth=1.5, label=f"{name} (Tree)")
        # title and labels
        ax.set_title(f"{name} vs Volatilité (Tree vs BS)", fontsize=10)
        ax.set_xlabel("Volatilité (%)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

        left = x_start + grid_x * (chart_width + x_spacing)
        top = y_start + grid_y * (chart_height + y_spacing)

        sh.pictures.add(fig, name=f"Chart_{name}", update=True, 
                        left=left, top=top, width=chart_width, height=chart_height)
        plt.close(fig)

# volatility test
def volatility_test() -> None:
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()
    vol_values, data = _build_volatility_dataset(
        market, option, N, exercise, optimize, threshold, S0, K, r, T, is_call
    )
    sh = _write_data_to_sheet(wb, data)
    _plot_greek_charts(sh, vol_values, data)
    # Autofit
    sh.autofit()

def run_test_greeks_volatility() -> None:
    volatility_test()

if __name__ == "__main__":
    run_test_greeks_volatility()
