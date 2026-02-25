import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
from core_pricer import input_parameters, run_backward_pricing, run_recursive_pricing, run_black_scholes
from utils.utils_sheet import ensure_sheet
from utils.utils_tree_error import tree_error
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xlwings as xw


def outil_convergence_excel():
    """
    Crée un test de convergence entre le prix du modèle trinomial et le modèle Black-Scholes.
    Écrit les résultats dans la feuille Excel 'Test Convergence' et trace deux graphiques :
    1. Tree vs BS
    2. (Tree - BS) x Nb
    3. Tree Error
    """
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()
    
    sheet_cv = ensure_sheet(wb, "Test Convergence")

    bs_val, _ = run_black_scholes(S0, K, r, sigma, T, is_call)

    N_values = list(range(1, N + 1))

    # En-têtes et configuration
    headers = ["N", "Prix Tree", "Prix BS", "(Tree - BS) x N", "Tree Error"]
    start_col = "V"
    start_row = 5
    data_start_row = start_row + 1
    end_row = start_row + N

    sheet_cv.range(f"{start_col}{start_row}:Y{start_row}").value = headers
    sheet_cv.range(f"{start_col}{start_row}:Y{start_row}").font.bold = True

    # Calcul des prix Tree et erreurs
    data = []
    for n in N_values:
        if method == "Backward":
            price, _, _ = run_backward_pricing(market, option, n, exercise, optimize, threshold)
        else:
            price, _, _ = run_recursive_pricing(market, option, N, exercise, optimize, threshold)

        error = tree_error(S0, sigma, r, T, n)
        
        data.append([n, price, bs_val, (price - bs_val) * n, error])

    # Écriture des données
    sheet_cv.range(f"{start_col}{data_start_row}").value = data

    # Création des graphiques
    width, height = 950, 500
    top_start = 90
    left_start = 1780
    vertical_gap = 20

    left_start = sheet_cv.range("AB1").left


    # Chart 1: Tree vs BS
    chart1 = sheet_cv.charts.add(left=left_start, top=top_start, width=width, height=height)
    chart1.chart_type = "xy_scatter_smooth_no_markers"
    chart1.set_source_data(sheet_cv.range(f"{start_col}{start_row}:X{end_row}"))
    chart1.title = "Tree vs BS : Python"

    # Chart 2: (Tree - BS) x N
    helper_col = "AA"   
    helper_header = ["N", "(Tree - BS) x N"]
    sheet_cv.range(f"{helper_col}{start_row}").value = helper_header

    n_rng = sheet_cv.range(f"{start_col}{data_start_row}:{start_col}{end_row}").value
    err_rng = sheet_cv.range(f"Y{data_start_row}:Y{end_row}").value
    helper_data = [[n, e] for n, e in zip(n_rng, err_rng)]
    sheet_cv.range(f"{helper_col}{data_start_row}").value = helper_data

    chart2 = sheet_cv.charts.add(left=left_start, top=top_start + height + vertical_gap, width=width, height=height)
    chart2.chart_type = "xy_scatter_smooth_no_markers"
    chart2.set_source_data(sheet_cv.range(f"{helper_col}{start_row}:AB{end_row}")) 
    chart2.title = "(Tree - BS) x NbSteps vs N : Python"

    # Chart 3: Tree Error
    helper_cols = "AC"
    helper_headers = ["N", "Tree Error"]
    sheet_cv.range(f"{helper_cols}{start_row}").value = helper_headers

    n_rngs = sheet_cv.range(f"{start_col}{data_start_row}:{start_col}{end_row}").value
    err_rngs = sheet_cv.range(f"Z{data_start_row}:Z{end_row}").value
    helper_datas = [[n, e] for n, e in zip(n_rngs, err_rngs)]
    sheet_cv.range(f"{helper_cols}{data_start_row}").value = helper_datas

    chart3 = sheet_cv.charts.add(left=left_start, top=top_start + 2*height + 2*vertical_gap, width=width, height=height)
    chart3.chart_type = "xy_scatter_smooth_no_markers"
    chart3.set_source_data(sheet_cv.range(f"{helper_cols}{start_row}:AD{end_row}")) 
    chart3.title = "Tree Error: Python"

    sheet_cv.range(f"AA{start_row}:AD{end_row}").font.color = (255, 255, 255)
    sheet_cv.autofit()

def run_cv():
    """
    Vérifie si les conditions permettent de lancer le test de convergence.
    """
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    if (exercise == "european"):
        outil_convergence_excel()
    else: 
        pass

if __name__ == "__main__":
    run_cv()