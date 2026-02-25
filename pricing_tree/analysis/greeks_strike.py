import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import input_parameters
from analysis.greeks import compute_method_greeks
from utils.utils_bs import bs_greeks
from utils.utils_sheet import ensure_sheet


def strike_test():
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    K_values = np.linspace(int(0.9 * S0), int(1.1 * S0), 20)
    data = []

    # -- Status bar (réf. au même App Excel)
    app = wb.app

    for k in K_values:
        # Début du pricing pour ce strike : 0/N
        try:
            app.status_bar = f"Strike : {float(k):.4f} | N = {int(N)}"
        except Exception:
            pass

        option.K = k
        tree_greeks = compute_method_greeks(market, option, N, exercise, optimize, threshold, "backward")
        bs_vals = bs_greeks(S0, k, r, sigma, T, is_call)

        # Fin du pricing pour ce strike : N/N
        try:
            app.status_bar = f"Strike : {float(k):.4f} | {int(N)}/{int(N)}"
        except Exception:
            pass

        # >>> NE RIEN CHANGER ICI : structure intacte (15 colonnes)
        data.append([
            k,
            tree_greeks["Delta"], bs_vals["Delta"],
            tree_greeks["Gamma"], bs_vals["Gamma"],
            tree_greeks["Vega"], bs_vals["Vega"],
            tree_greeks["Theta"], bs_vals["Theta"],
            tree_greeks["Rho"], bs_vals["Rho"],
            tree_greeks["Vanna"], bs_vals["Vanna"],
            tree_greeks["Vomma"], bs_vals["Vomma"]
        ])

    # Réinitialise la barre d’état
    try:
        app.status_bar = False
    except Exception:
        pass


    data = [[round(x, 4) if isinstance(x, (float, np.floating)) else x for x in row] for row in data]

    sheet_name = "Greeks Strike"
    sh = ensure_sheet(wb, sheet_name)

    headers = [
        "Strike",
        "Delta Tree", "Delta BS",
        "Gamma Tree", "Gamma BS",
        "Vega Tree", "Vega BS",
        "Theta Tree", "Theta BS",
        "Rho Tree", "Rho BS",
        "Vanna Tree", "Vanna BS",
        "Vomma Tree", "Vomma BS"
    ]

    sh.range("A3").value = headers
    sh.range("A4").value = data
    #sh.range("A4").expand().number_format = "0.0000"

    chart_specs = [
        ("Delta", 2, 3, "green", 0, 0),
        ("Theta", 8, 9, "red", 1, 0),
        ("Vega", 6, 7, "orange", 2, 0),
        ("Rho", 10, 11, "purple", 0, 1),
        ("Gamma", 4, 5, "blue", 1, 1),
        ("Vomma", 14, 15, "gold", 2, 1),
        ("Vanna", 12, 13, "black", 0, 2)
    ]

    chart_width = 360     
    chart_height = 250
    x_spacing = 50
    y_spacing = 60
    x_start = 1050
    y_start = 60

    for name, col_tree, col_bs, color, grid_x, grid_y in chart_specs:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        tree_vals = [row[col_tree - 1] for row in data]
        bs_vals = [row[col_bs - 1] for row in data]

        ax.plot(K_values, bs_vals, color=color, linestyle="--", linewidth=1.5, label=f"{name} (BS)")
        ax.plot(K_values, tree_vals, color=color, linewidth=1.5, label=f"{name} (Tree)")

        ax.set_title(f"{name} vs Strike (Tree vs BS)", fontsize=10)
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

        left = x_start + grid_x * (chart_width + x_spacing)
        top = y_start + grid_y * (chart_height + y_spacing)

        sh.pictures.add(fig, name=f"Chart_{name}", update=True,
                        left=left, top=top, width=chart_width, height=chart_height)
        plt.close(fig)

    sh.autofit()
    
    try:
        wb.app.status_bar = False
    except Exception:
        pass


def run_test_greeks_strike():
    strike_test()


if __name__ == "__main__":
    run_test_greeks_strike()
