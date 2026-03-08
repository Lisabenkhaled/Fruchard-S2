import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ajout du dossier parent au path pour importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_pricer import input_parameters, run_backward_pricing
from utils.utils_bs import bs_price
from utils.utils_sheet import ensure_sheet


def compute_prices(
    market: Any,
    option: Any,
    n_steps: int,
    exercise: Any,
    optimize: Any,
    threshold: Any,
    s0: float,
    rate: float,
    sigma: float,
    maturity: float,
    is_call: bool,
    strike_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les prix BS et arbre pour chaque strike."""
    bs_prices = []
    tree_prices = []

    # boucle sur les strikes
    for strike in strike_values:
        option.K = strike

        # prix Black-Scholes
        bs_price_value = bs_price(s0, strike, rate, sigma, maturity, is_call)
        bs_prices.append(bs_price_value)

        # prix arbre
        tree_price, _, _ = run_backward_pricing(
            market,
            option,
            n_steps,
            exercise,
            optimize,
            threshold,
        )
        tree_prices.append(tree_price)

    return np.array(bs_prices), np.array(tree_prices)


def write_results(
    sheet_pr: Any,
    strike_values: np.ndarray,
    bs_prices: np.ndarray,
    tree_prices: np.ndarray,
) -> np.ndarray:
    """Écrit les résultats dans Excel et renvoie l'écart."""
    diff = tree_prices - bs_prices
    headers = ["Strike", "BS", "Tree", "Tree - BS"]
    data = np.column_stack((strike_values, bs_prices, tree_prices, diff))

    start_row = 4
    sheet_pr.range(f"A{start_row}").value = headers
    sheet_pr.range(f"A{start_row + 1}").value = data

    return diff


def add_chart(
    sheet_pr: Any,
    strike_values: np.ndarray,
    bs_prices: np.ndarray,
    tree_prices: np.ndarray,
    diff: np.ndarray,
) -> None:
    """Crée et insère le graphique dans Excel."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # courbes principales
    ax1.plot(strike_values, bs_prices, color="green", label="BS")
    ax1.plot(strike_values, tree_prices, color="gold", label="Tree")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Prix d'option")
    ax1.set_xlim(min(strike_values) - 1, max(strike_values) + 1)
    ax1.grid(True, alpha=0.3)

    # courbe d'écart sur axe secondaire
    ax2 = ax1.twinx()
    ax2.plot(strike_values, diff, color="red", label="Tree - BS")
    ax2.set_ylabel("Tree - BS")

    max_diff = float(np.max(np.abs(diff))) if diff.size else 1.0
    ax2.set_ylim(-1.1 * max_diff, 1.1 * max_diff)

    # légende combinée
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Tree et Black-Scholes prix par rapport au strike")
    plt.tight_layout()

    sheet_pr.pictures.add(
        fig,
        name="Tree_vs_BS",
        update=True,
        left=300,
        top=60,
    )
    plt.close(fig)


def strike_test() -> None:
    """Exécute le test de sensibilité au strike."""
    (
        market,
        option,
        n_steps,
        exercise,
        method,
        optimize,
        threshold,
        arbre_stock,
        arbre_proba,
        arbre_option,
        wb,
        sheet,
        s0,
        strike,
        rate,
        sigma,
        maturity,
        rho,
        lam,
        is_call,
        exdivdate,
    ) = input_parameters()

    # création de la feuille si nécessaire
    sheet_pr = ensure_sheet(wb, "Test Sur Param")

    # génération des strikes de test
    strike_values = np.linspace(strike - 5, strike + 5, 30)

    # calcul des prix
    bs_prices, tree_prices = compute_prices(
        market,
        option,
        n_steps,
        exercise,
        optimize,
        threshold,
        s0,
        rate,
        sigma,
        maturity,
        is_call,
        strike_values,
    )

    # écriture des résultats
    diff = write_results(sheet_pr, strike_values, bs_prices, tree_prices)

    # création du graphique
    add_chart(sheet_pr, strike_values, bs_prices, tree_prices, diff)


def run_strike_test() -> None:
    """Lance le test strike."""
    strike_test()


if __name__ == "__main__":
    run_strike_test()