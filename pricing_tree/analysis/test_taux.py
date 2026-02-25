import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils_bs import bs_price
from utils.utils_sheet import ensure_sheet
from core_pricer import (
    input_parameters,
    run_backward_pricing,
)

def rate_test():
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()
    
    # Si la feuille n'existe pas, la créer
    sheet_pr = ensure_sheet(wb, "Test Sur Param")

    # Paramètres pour le test
    r_values = np.linspace(-0.1, 0.1, 30)
    bs_prices, tree_prices = [], []

    for r_test in r_values:
        market.r = r_test

        bs_p = bs_price(S0, K, r_test, sigma, T, is_call)
        bs_prices.append(bs_p)

        tree_p, _, _ = run_backward_pricing(market, option, N, exercise, optimize=False, threshold=threshold)
        tree_prices.append(tree_p)

    bs_prices = np.array(bs_prices)
    tree_prices = np.array(tree_prices)
    diff = tree_prices - bs_prices

    headers = ["Taux", "BS", "Tree", "Tree - BS"]
    data = np.column_stack((r_values, bs_prices, tree_prices, diff))
    start_row = 4
    sheet_pr.range(f"O{start_row}").value = headers
    sheet_pr.range(f"O{start_row+1}").value = data

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(r_values * 100, bs_prices, color="green", label="BS")
    ax1.plot(r_values * 100, tree_prices, color="gold", label="Tree")
    ax1.set_xlabel("Taux d'intérêt(%)")
    ax1.set_ylabel("Prix d'option")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(r_values * 100, diff, color="red", label="Tree - BS")
    ax2.set_ylabel("Tree - BS")
    m = float(np.max(np.abs(diff))) if diff.size else 1.0
    ax2.set_ylim(-1.1*m, 1.1*m)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Tree et Black-Scholes prix par rapport au taux d'intérêt")
    plt.tight_layout()

    sheet_pr.pictures.add(fig, name="Tree_vs_BS_Taux", update=True, left=1250, top=60)

    plt.close(fig)

def run_taux_test():
    rate_test()

if __name__ == "__main__":
    run_taux_test()

