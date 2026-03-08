import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from numpy.typing import NDArray

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import from projects
from utils.utils_bs import bs_price
from utils.utils_sheet import ensure_sheet
from core_pricer import (
    input_parameters,
    run_backward_pricing,
)

# build rates to test 
def _build_rate_data(market: Any,option: Any, N: int,exercise: Any,
    threshold: float,S0: float,K: float,sigma: float,T: float,
    is_call: bool) -> tuple[NDArray[np.float64], 
                            NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r_values = np.linspace(-0.1, 0.1, 30)
    bs_prices, tree_prices = [], []

    for r_test in r_values:
        market.r = r_test

        bs_prices.append(bs_price(S0, K, r_test, sigma, T, is_call))

        tree_p, _, _ = run_backward_pricing(market, option, N, exercise, optimize=False, threshold=threshold)
        tree_prices.append(tree_p)
    # difference between tree and bs
    bs_prices = np.array(bs_prices)
    tree_prices = np.array(tree_prices)
    diff = tree_prices - bs_prices
    return r_values, bs_prices, tree_prices, diff

# write rate data
def _write_rate_data(sheet_pr: Any,r_values: NDArray[np.float64],
    bs_prices: NDArray[np.float64],tree_prices: NDArray[np.float64],
    diff: NDArray[np.float64],) -> None:
    headers = ["Taux", "BS", "Tree", "Tree - BS"]
    data = np.column_stack((r_values, bs_prices, tree_prices, diff))
    #start row and col
    start_row, start_col = 4, 15
    sheet_pr.range((start_row, start_col)).value = headers
    sheet_pr.range((start_row + 1, start_col)).value = data

#rate charts
def _plot_rate_chart(sheet_pr: Any,r_values: NDArray[np.float64],
    bs_prices: NDArray[np.float64],tree_prices: NDArray[np.float64],
    diff: NDArray[np.float64],) -> None:

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    x_values = r_values * 100
    ax1.plot(x_values, bs_prices, color="green", label="BS")
    ax1.plot(x_values, tree_prices, color="gold", label="Tree")
    #labels
    ax1.set_xlabel("Taux d'intérêt(%)")
    ax1.set_ylabel("Prix d'option")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_values, diff, color="red", label="Tree - BS")
    ax2.set_ylabel("Tree - BS")
    m = float(np.max(np.abs(diff))) if diff.size else 1.0
    ax2.set_ylim(-1.1*m, 1.1*m)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    #title
    plt.title("Tree et Black-Scholes prix par rapport au taux d'intérêt")
    plt.tight_layout()

    sheet_pr.pictures.add(fig, name="Tree_vs_BS_Taux", update=True, left=1250, top=60)

    plt.close(fig)

# rate test "main"
def rate_test() -> None:
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    sheet_pr = ensure_sheet(wb, "Test Sur Param")
    r_values, bs_prices, tree_prices, diff = _build_rate_data(
        market, option, N, exercise, threshold, S0, K, sigma, T, is_call
    )
    _write_rate_data(sheet_pr, r_values, bs_prices, tree_prices, diff)
    _plot_rate_chart(sheet_pr, r_values, bs_prices, tree_prices, diff)

#run
def run_taux_test() -> None:
    rate_test()

if __name__ == "__main__":
    run_taux_test()

