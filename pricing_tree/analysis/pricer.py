import sys
import os
import time
from typing import Any

import xlwings as xw
from numpy import sqrt, exp, pi


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import input_parameters, run_pricer as core_run_pricer
from utils.utils_sheet import ensure_sheet
from utils.utils_tree_error import tree_error


def vertical_tree(levels: Any, attr: str, decimals: int = 6) -> list[list[Any]]:
    n = len(levels)
    matrix = [[""] * n for _ in range(2 * n + 1)]
    center_row = n

    for i, level in enumerate(levels):
        offset = len(level) // 2
        for j, node in enumerate(level):
            if node is None:
                continue
            value = getattr(node, attr, None)
            if value is None:
                continue
            matrix[center_row - (j - offset)][i] = round(value, decimals)
    return matrix


def write_tree(wb: Any, sheet_name: str, title: str, matrix: list[list[Any]]) -> None:
    """Efface et écrit la matrice dans une feuille Excel."""
    try:
        sht = ensure_sheet(wb, sheet_name)
        sht.range((1, 1), (1000, 702)).value = None
        time.sleep(0.05)
        sht.range((1, 1)).value = title
        sht.range((2, 1)).value = matrix
        try:
            sht.range((1, 1), (1000, 702)).columns.autofit()
        except Exception:
            pass
    except Exception as e:
        print(f"[display_trees] Warning: failed to write '{sheet_name}': {e}")


def display_trees(
    wb: Any,
    tree: Any,
    show_stock: bool,
    show_reach: bool,
    show_option: bool,
    threshold: float = 1e-7
) -> None:
    """
    Affiche les arbres (sous-jacent, probas d’atteinte, valeurs d’option, etc.)
    dans Excel.
    """

    if hasattr(tree, "to_levels_for_excel"):
        levels = tree.to_levels_for_excel()
    elif hasattr(tree, "tree"):
        levels = tree.tree
    else:
        raise ValueError("display_trees(): structure d’arbre non reconnue.")

    # show stock price
    if show_stock:
        matrix = vertical_tree(levels, "stock_price", 4)
        write_tree(wb, "Arbre Stock Price", "Stock Price Tree", matrix)

    # show prix option
    if show_option:
        matrix = vertical_tree(levels, "option_value", 6)
        write_tree(wb, "Arbre Option", "Option Value Tree", matrix)

    # Show reach probabilities
    if show_reach:
        matrix = vertical_tree(levels, "p_reach", 10)
        write_tree(wb, "Arbre Proba", "Reach Probability Tree", matrix)

        first = levels[0][0] if levels and levels[0] else None
        if first:
            for name in ("p_up", "p_mid", "p_down"):
                if hasattr(first, name):
                    matrix = vertical_tree(levels, name, 6)
                    write_tree(
                        wb,
                        f"Arbre {name}",
                        f"Local Probabilities ({name})",
                        matrix,
                    )


@xw.sub
def run_pricer() -> None:
    """
    Fonction principale appelée par le bouton 'PRICER' dans Excel :
    - Lit les paramètres
    - Calcule les prix
    - Affiche les résultats et les arbres
    """
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    # Execution
    results = core_run_pricer()

    # Ecrire les resultats
    sheet.range('Prix_Tree').value = results['tree_price']
    sheet.range('Prix_Tree').number_format = '0.0000'

    sheet.range('Time_Tree').value = results['tree_time']
    sheet.range('Time_Tree').number_format = '0.000000'

    sheet.range('Prix_BS').value = results['bs_price']
    sheet.range('Prix_BS').number_format = '0.0000'

    sheet.range('Time_BS').value = results['bs_time']
    sheet.range('Time_BS').number_format = '0.000000'

    sheet.range('tree_error').value = tree_error(S0, sigma, r, T, N)
    sheet.range('tree_error').number_format = '0.0000'

    # Show Tree
    display_trees(
        wb,
        results["tree"],
        show_stock=(arbre_stock == "Oui"),
        show_reach=(arbre_proba == "Oui"),
        show_option=(arbre_option == "Oui"),
        threshold=threshold
    )

    wb.save()