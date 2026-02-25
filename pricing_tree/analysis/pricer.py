import sys
import os
import time
import xlwings as xw
from numpy import sqrt, exp, pi


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import input_parameters, run_pricer as core_run_pricer
from utils.utils_sheet import ensure_sheet
from utils.utils_tree_error import tree_error

def display_trees(wb, tree, show_stock, show_reach, show_option, threshold=1e-7):
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

    def vertical_tree(levels, attr, decimals=6):
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

    def write_tree(wb, sheet_name, title, matrix):
        """Efface et écrit la matrice dans une feuille Excel."""
        try:
            sht = ensure_sheet(wb, sheet_name)
            # Effacement sans clear_contents() pour éviter bug AppleScript
            sht.range("A:ZZ").value = None
            time.sleep(0.05)
            sht.range("A1").value = title
            sht.range("A2").value = matrix
            try:
                sht.range("A:ZZ").columns.autofit()
            except Exception:
                pass  # Ignore Mac Excel autofit issues
        except Exception as e:
            print(f"[display_trees] Warning: failed to write '{sheet_name}': {e}")

    if show_stock:
        matrix = vertical_tree(levels, "stock_price", 4)
        write_tree(wb, "Arbre Stock Price", "Stock Price Tree", matrix)

    if show_option:
        matrix = vertical_tree(levels, "option_value", 6)
        write_tree(wb, "Arbre Option", "Option Value Tree", matrix)

    if show_reach:
        matrix = vertical_tree(levels, "p_reach", 10)
        write_tree(wb, "Arbre Proba", "Reach Probability Tree", matrix)

        # Probabilités locales (p_up, p_mid, p_down)
        first = levels[0][0] if levels and levels[0] else None
        if first:
            for name in ("p_up", "p_mid", "p_down"):
                if hasattr(first, name):
                    matrix = vertical_tree(levels, name, 6)
                    write_tree(wb, f"Arbre {name}", f"Local Probabilities ({name})", matrix)

@xw.sub
def run_pricer():
    """
    Fonction principale appelée par le bouton 'PRICER' dans Excel :
    - Lit les paramètres
    - Calcule les prix
    - Affiche les résultats et les arbres
    """
    # Lecture des paramètres depuis Excel
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    results = core_run_pricer()

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

    display_trees(
        wb,
        results["tree"],
        show_stock=(arbre_stock == "Oui"),
        show_reach=(arbre_proba == "Oui"),
        show_option=(arbre_option == "Oui"),
        threshold=threshold
    )

    wb.save()
