import sys
import os
import copy
import numpy as np

# -------------------------------------------------------------------------
# Import des modules internes
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_pricer import (
    input_parameters,
    run_backward_pricing,
    run_recursive_pricing,
)
from utils.utils_bs import bs_greeks
from utils.utils_grecs import OneDimDerivative


# -------------------------------------------------------------------------
# 1. Fonction générique : récupération du prix selon la méthode
# -------------------------------------------------------------------------
def get_price(market, option, N, exercise, optimize, threshold, method):
    """Retourne le prix de l’option selon la méthode choisie."""
    pricing_fn = run_backward_pricing if method.lower() == "backward" else run_recursive_pricing
    price, _, _ = pricing_fn(market, option, N, exercise, optimize, threshold)
    return float(price)


# -------------------------------------------------------------------------
# 2. Wrapper générique pour dérivées numériques
# -------------------------------------------------------------------------
def greek_wrapper(params, x: float) -> float:
    """
    Modifie un paramètre de marché pour recalculer le prix.
    Utilisé pour la dérivation numérique.
    """
    market, option, N, exercise, optimize, threshold, target, method = params
    m = copy.deepcopy(market)
    setattr(m, target, x)
    return get_price(m, option, N, exercise, optimize, threshold, method)


# -------------------------------------------------------------------------
# 3. Dérivées croisées : Vanna & Vomma
# -------------------------------------------------------------------------
def finite_diff_2d(market, option, N, exercise, optimize, threshold, method, base_price):
    """
    Calcule Vanna (∂²V / ∂S∂σ) et Vomma (∂²V / ∂σ²) par différences finies centrées.
    """
    S0, sigma0 = market.S0, market.sigma
    hS = max(1e-5, 0.01 * S0)
    hSigma = max(1e-5, 0.005)

    def price_shift(dS=0.0, dSigma=0.0):
        m = copy.deepcopy(market)
        m.S0 = S0 + dS
        m.sigma = max(1e-6, sigma0 + dSigma)
        return get_price(m, option, N, exercise, optimize, threshold, method)

    # --- Calcul des dérivées croisées
    p_up_up     = price_shift(+hS, +hSigma)
    p_up_down   = price_shift(+hS, -hSigma)
    p_down_up   = price_shift(-hS, +hSigma)
    p_down_down = price_shift(-hS, -hSigma)

    vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hSigma)

    # --- Calcul de Vomma
    p_sig_up   = price_shift(0.0, +hSigma)
    p_sig_down = price_shift(0.0, -hSigma)
    vomma = (p_sig_up - 2 * base_price + p_sig_down) / (hSigma ** 2)

    # Sécurité numérique
    if not np.isfinite(vanna): vanna = 0.0
    if not np.isfinite(vomma): vomma = 0.0

    return float(vanna), float(vomma)


# -------------------------------------------------------------------------
# 4. Calcul complet des greeks selon une méthode
# -------------------------------------------------------------------------
def compute_method_greeks(market, option, N, exercise, optimize, threshold, method):
    """
    Calcule les principaux grecs pour une méthode donnée.
    """
    base_price = get_price(market, option, N, exercise, optimize, threshold, method)

    # Petits incréments pour dérivées numériques
    hS = max(1e-5, 0.005 * market.S0)
    hSigma = max(1e-5, 0.005)
    hR = 1e-4
    hT = 1.0 / 365.0  # un jour

    # --- Dérivées simples
    dS = OneDimDerivative(greek_wrapper, (market, option, N, exercise, optimize, threshold, "S0", method), shift=hS)
    dSigma = OneDimDerivative(greek_wrapper, (market, option, N, exercise, optimize, threshold, "sigma", method), shift=hSigma)
    dR = OneDimDerivative(greek_wrapper, (market, option, N, exercise, optimize, threshold, "r", method), shift=hR)
    dT = OneDimDerivative(greek_wrapper, (market, option, N, exercise, optimize, threshold, "T", method), shift=hT)

    # --- Calcul des grecs
    Delta = dS.first(market.S0)
    Gamma = dS.second(market.S0)
    Vega  = dSigma.first(market.sigma)
    Rho   = dR.first(market.r)
    Theta = -dT.first(market.T)

    # --- Dérivées croisées
    Vanna, Vomma = finite_diff_2d(market, option, N, exercise, optimize, threshold, method, base_price)

    return {
        "Delta": Delta,
        "Gamma": Gamma,
        "Vega": Vega,
        "Theta": Theta,
        "Rho": Rho,
        "Vanna": Vanna,
        "Vomma": Vomma,
    }


# -------------------------------------------------------------------------
# 5. Fonction principale (intégrée à Excel)
# -------------------------------------------------------------------------
def compute_greeks():
    """
    Calcule les grecs pour les deux méthodes (Backward / Recursive)
    et les compare à Black-Scholes si applicable.
    """
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, is_call, exdivdate, *_) = input_parameters()

    ws = wb.sheets["Greeks"]

    # Nettoyage de la feuille

    ws.range("C6:ZZ1048576").clear_contents()

    # Vérifie si on peut comparer à BS
    can_use_bs = (not exdivdate) and (exercise.lower() == "european")
    bs_results = bs_greeks(S0, K, r, sigma, T, is_call) if can_use_bs else None

    # Calcul des grecs pour les deux méthodes
    results = {
        "recursive": compute_method_greeks(market, option, N, exercise, optimize, threshold, "recursive"),
        "backward": compute_method_greeks(market, option, N, exercise, optimize, threshold, "backward"),
    }

    # --- Écriture des résultats dans Excel
    labels = ["Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"]
    start_row = 6
    col_bs, col_rec, col_bw = "C", "F", "I"

    for i, key in enumerate(labels):
        ws.range(f"{col_rec}{start_row + i}").value = results["recursive"].get(key)
        ws.range(f"{col_bw}{start_row + i}").value = results["backward"].get(key)
        #ws.range(f"{col_rec}{start_row + i}:{col_bw}{start_row + i}").number_format = "0.000000"

        if can_use_bs:
            ws.range(f"{col_bs}{start_row + i}").value = bs_results.get(key)
        else:
            ws.range(f"{col_bs}{start_row + i}").value = None

    return results


# -------------------------------------------------------------------------
# 6. Exécution directe
# -------------------------------------------------------------------------
if __name__ == "__main__":
    compute_greeks()