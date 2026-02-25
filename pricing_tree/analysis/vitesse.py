import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import (
    input_parameters,
    run_backward_pricing,
    run_recursive_pricing,
)
from utils.utils_sheet import ensure_sheet

def run_vt():
    """
    Teste le temps de calcul du modèle trinomial selon le nombre d'étapes N.
    Compare le temps pour :
    - Méthode backward / recursive
    - Avec et sans pruning
    """
    # Lecture des paramètres depuis Excel
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()
    
    # Si la feuille n'existe pas, la créer
    sheet_vt = ensure_sheet(wb, "Test Vitesse")

    # Initialisation des paramètres
    N_values = list(range(1, N+1))
    start_col_sp = "B"   # Sans pruning
    start_col_ap = "V"   # Avec pruning
    start_row = 5
    end_row = start_row + N

    # Dimensions des graphiques
    width, height = 800, 400
    top_start = 100
    left_start = 440

    # Boucle principale : mesures de temps
    for n in N_values:
        # Sans pruning
        _, temps_bp_s, _ = run_backward_pricing(market, option, n, exercise, False, threshold)
        _, temps_rec_s, _ = run_recursive_pricing(market, option, n, exercise, False, threshold)

        # Avec pruning
        _, temps_bp_av, _ = run_backward_pricing(market, option, n, exercise, True, threshold)
        _, temps_rec_av, _ = run_recursive_pricing(market, option, n, exercise, True, threshold)

        # Écriture dans Excel
        row = start_row + n
        sheet_vt.range(f"{start_col_sp}{row}").value = n
        sheet_vt.range(f"{start_col_ap}{row}").value = n

        sheet_vt.range(f"C{row}").value = temps_rec_s
        sheet_vt.range(f"W{row}").value = temps_rec_av

        sheet_vt.range(f"E{row}").value = temps_bp_s
        sheet_vt.range(f"Y{row}").value = temps_bp_av

    #Création des graphiques
    # Chart 1: Sans Pruning
    #chart1 = sheet_vt.charts.add(left=left_start, top=top_start, width=width, height=height)
    #chart1.chart_type = "xy_scatter_smooth_no_markers"
    #chart1.set_source_data(sheet_vt.range(f"{start_col_sp}{start_row}:F{end_row}"))
    #chart1.title = "Temps du calcul : Sans Pruning"

    # Chart 2: Avec Pruning
    #chart2 = sheet_vt.charts.add(left=left_start + 1300, top=top_start, width=width, height=height)
    #chart2.chart_type = "xy_scatter_smooth_no_markers"
    #chart2.set_source_data(sheet_vt.range(f"{start_col_ap}{start_row}:Z{end_row}")) 
    #chart2.title = "Temps du calcul : Avec Pruning"

    sheet_vt.autofit()