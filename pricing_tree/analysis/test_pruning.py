# analysis/test_pruning.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import (
    input_parameters,
    run_backward_pricing
)
from utils.utils_sheet import ensure_sheet

# write results
def _write_results_table(sheet_prune: Any, seuil_values: list[float], 
                         prices: list[float], times: list[float]) -> None:
    headers: list[str] = ["Seuil", "Prix Tree", "Durée (s)"]
    table_start_col = "A"
    header_row = 2
    data_row = header_row + 1

    sheet_prune.range(f"{table_start_col}{header_row}:C{header_row}").value = headers
    sheet_prune.range(f"{table_start_col}{data_row}").value = np.column_stack((seuil_values, prices, times))

# price chart
def _add_price_chart(sheet_prune: Any,seuil_values: list[float],
    prices: list[float],baseline_price: float,left: float,
    top: float,width: int,height: int) -> None:

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(seuil_values, prices, color="gold", linewidth=2, label="Prix arbre")
    ax1.axhline(y=baseline_price, color="red", linestyle="--", linewidth=1.3,
                label=f"Prix sans pruning = {baseline_price:.6f}")
    ax1.set_xscale("log")
    
    #labels
    ax1.set_xlabel("Seuil de pruning (log scale)")
    ax1.set_ylabel("Prix de l'option")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Impact du seuil de pruning sur le prix de l'option")
    ax1.legend(loc="best")
    plt.tight_layout()
    sheet_prune.pictures.add(fig1, name="Graph_Prix", update=True, left=left, top=top, width=width, height=height)
    plt.close(fig1)

# time chart
def _add_time_chart(
    sheet_prune: Any,seuil_values: list[float],times: list[float],
    baseline_time: float,left: float,top: float,width: int,
    height: int) -> None:
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(seuil_values, times, linewidth=2, label="Durée de calcul (s)")
    ax2.axhline(y=baseline_time, color="red", linestyle="--", linewidth=1.3,
                label=f"Temps sans pruning = {baseline_time:.3f} s")
    ax2.set_xscale("log")
    ax2.set_xlabel("Seuil de pruning (log scale)")
    ax2.set_ylabel("Durée d'exécution (secondes)")
    ax2.grid(True, alpha=0.3)

    # title
    ax2.set_title("Impact du seuil de pruning sur la durée de calcul")
    ax2.legend(loc="best")
    plt.tight_layout()
    sheet_prune.pictures.add(fig2, name="Graph_Temps", update=True, left=left, top=top, width=width, height=height)
    plt.close(fig2)

# test pruning
def prune_test()->None:
    # Récupération standard des paramètres + workbook
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    sheet_prune = ensure_sheet(wb, "Test Pruning")

    # Seuils: 1e-3 -> 1e-15 inclus
    seuil_values = [10.0 ** (-k) for k in range(3, 16)]

    prices: list[float] = []
    times: list[float] = []

    app = wb.app
    app.screen_updating = False
    total = len(seuil_values)
    try:
        # Calcul du prix et temps sans pruning (baseline)
        baseline_price, baseline_time, _ = run_backward_pricing(
            market, option, N, exercise, optimize=False, threshold=0.0
        )

        # Boucle sur les seuils 
        for i, s in enumerate(seuil_values, start=1):
            # Status bar 
            try:
                app.status_bar = f"Pruning: seuil={s:.1e} | {i}/{total}"
            except Exception:
                pass

            price_t, t, _ = run_backward_pricing(
                market, option, N, exercise, optimize, threshold=s
            )
            prices.append(price_t)
            times.append(t)
    finally:
        # Toujours remettre la status bar
        try:
            app.status_bar = False
        except Exception:
            pass
        app.screen_updating = True

    # Tableau résultats 
    _write_results_table(sheet_prune, seuil_values, prices, times)
    
    # Nettoyage anciens graphiques 
    for pic in list(sheet_prune.pictures):
        if pic.name in ["Graph_Prix", "Graph_Temps"]:
            pic.delete()

    # Coordonnées depuis F3 
    chart_anchor_col = "F"
    chart_anchor_row = 3
    anchor: Any = sheet_prune.range(f"{chart_anchor_col}{chart_anchor_row}")

    left = anchor.left
    top1 = anchor.top
    width = 640
    height = 360
    gap = 20
    top2 = top1 + height + gap

    _add_price_chart(sheet_prune, seuil_values, prices, baseline_price, left, top1, width, height)
    _add_time_chart(sheet_prune, seuil_values, times, baseline_time, left, top2, width, height)
    
if __name__ == "__main__":
    prune_test()
