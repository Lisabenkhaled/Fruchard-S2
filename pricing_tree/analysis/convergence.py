import sys
import os
from typing import Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings

# import from projects
from core_pricer import input_parameters, run_backward_pricing, run_recursive_pricing, run_black_scholes
from utils.utils_sheet import ensure_sheet
from utils.utils_tree_error import tree_error
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xlwings as xw

# convergence avec excel
def _compute_convergence_data(market: Any,option: Any,N: int,
    exercise: str, method: str, optimize: bool,threshold: float,
    S0: float,sigma: float,r: float,T: float,bs_val: float) -> list[list[float]]:
    data: list[list[float]] = []
    for n in range(1, N + 1):
        if method == "Backward":
            price, _, _ = run_backward_pricing(market, option, n, exercise, optimize, threshold)
        else:
            price, _, _ = run_recursive_pricing(market, option, N, exercise, optimize, threshold)
        error = tree_error(S0, sigma, r, T, n)
        data.append([n, price, bs_val, (price - bs_val) * n, error])
    return data

#convergence table 
def _write_convergence_table(sheet_cv: Any, start_col: str, start_row: int, data: list[list[float]]) -> int:
    # En-têtes et configuration
    headers: list[str] = ["N", "Prix Tree", "Prix BS", "(Tree - BS) x N", "Tree Error"]
    end_row = start_row + len(data)

    sheet_cv.range(f"{start_col}{start_row}:Y{start_row}").value = headers
    sheet_cv.range(f"{start_col}{start_row}:Y{start_row}").font.bold = True
    sheet_cv.range(f"{start_col}{start_row + 1}").value = data
    return end_row

# helper
def _build_helper_data(sheet_cv: Any, src_col: str, val_col: str, helper_col: str, start_row: int, end_row: int, headers: list[str]) -> None:
    data_start_row = start_row + 1
    sheet_cv.range(f"{helper_col}{start_row}").value = headers
    n_rng = sheet_cv.range(f"{src_col}{data_start_row}:{src_col}{end_row}").value
    err_rng = sheet_cv.range(f"{val_col}{data_start_row}:{val_col}{end_row}").value
    sheet_cv.range(f"{helper_col}{data_start_row}").value = [[n, e] for n, e in zip(n_rng, err_rng)]

def _create_charts(sheet_cv: Any, start_col: str, start_row: int, end_row: int) -> None:
    # Création des graphiques
    width, height = 950, 500
    top_start = 90
    vertical_gap = 20
    chart_anchor_col = "AB"
    chart_anchor_row = 1
    left_start = sheet_cv.range(f"{chart_anchor_col}{chart_anchor_row}").left

    # Chart 1: Tree vs BS
    chart1 = sheet_cv.charts.add(left=left_start, top=top_start, width=width, height=height)
    chart1.chart_type = "xy_scatter_smooth_no_markers"
    chart1.set_source_data(sheet_cv.range(f"{start_col}{start_row}:X{end_row}"))
    chart1.title = "Tree vs BS : Python"

    # Chart 2: (Tree - BS) x N
    helper_col = "AA"
    _build_helper_data(sheet_cv, start_col, "Y", helper_col, start_row, end_row, ["N", "(Tree - BS) x N"])

    chart2 = sheet_cv.charts.add(left=left_start, top=top_start + height + vertical_gap, width=width, height=height)
    chart2.chart_type = "xy_scatter_smooth_no_markers"
    chart2.set_source_data(sheet_cv.range(f"{helper_col}{start_row}:AB{end_row}"))
    chart2.title = "(Tree - BS) x NbSteps vs N : Python"

    # Chart 3: Tree Error
    helper_cols = "AC"
    _build_helper_data(sheet_cv, start_col, "Z", helper_cols, start_row, end_row, ["N", "Tree Error"])
    chart3 = sheet_cv.charts.add(left=left_start, top=top_start + 2 * height + 2 * vertical_gap, width=width, height=height)
    chart3.chart_type = "xy_scatter_smooth_no_markers"
    chart3.set_source_data(sheet_cv.range(f"{helper_cols}{start_row}:AD{end_row}"))
    chart3.title = "Tree Error: Python"

    sheet_cv.range(f"AA{start_row}:AD{end_row}").font.color = (255, 255, 255)
    
# convergence avec excel
def outil_convergence_excel()-> None:
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

    start_col = "V"
    start_row = 5
    data = _compute_convergence_data(
        market, option, N, exercise, method, optimize, threshold, S0, sigma, r, T, bs_val
    )
    end_row = _write_convergence_table(sheet_cv, start_col, start_row, data)
    _create_charts(sheet_cv, start_col, start_row, end_row)
    sheet_cv.autofit()
    
#run
def run_cv()->None:
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