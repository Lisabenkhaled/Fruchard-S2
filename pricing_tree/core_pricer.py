import time
import numpy as np
import os
import xlwings as xw
import datetime as dt

from models.market import Market
from models.option_trade import Option
from models.tree import TrinomialTree
from utils.utils_bs import bs_price
from utils.utils_date import datetime_to_years
from models.backward_pricing import price_backward
from models.recursive_pricing import price_recursive, clear_recursive_cache  # 👈 added import


# -------------------------------------------------------------------------
# 1. Lecture des paramètres dans Excel
# -------------------------------------------------------------------------
def input_parameters():
    """
    Lit les paramètres du pricer depuis Excel.
    Retourne les objets nécessaires pour le pricing.
    """

    # Paramètres de marché 
    S0 = 100
    r = 0.05
    sigma = 0.3
    rho = 0.0
    lam = 0.0
    exdiv_raw = None

    # Paramètres de l’option
    K = 102
    pricing_date = dt.date(2026, 2, 18)
    maturity_date = dt.date(2027, 2, 18)
    is_call = True
    exercise = "american"

    # Paramètres de l’arbre 
    # Valeur par défaut plus petite pour tests rapides. Augmentez si vous voulez plus de précision.
    N = 10000
    method = "Backward"
    optimize = "non"
    threshold = 0.00000000000001

    # Options d’affichage
    arbre_stock = False
    arbre_proba = False
    arbre_option = False

    # Conversion des dates
    T = datetime_to_years(maturity_date, pricing_date)
    exdivdate = datetime_to_years(exdiv_raw, pricing_date)

    # Création des objets Market et Option
    market = Market(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        exdivdate=exdivdate,
        pricing_date=pricing_date,
        rho=rho,
        lam=lam
    )
    option = Option(K=K, is_call=is_call)

    return (market, option, N, exercise, method, optimize, threshold,
            arbre_stock, arbre_proba, arbre_option, None, None,
            S0, K, r, sigma, T, rho, lam, is_call, exdivdate)


# -------------------------------------------------------------------------
# 2. Backward pricing
# -------------------------------------------------------------------------
def run_backward_pricing(market, option, N, exercise, optimize, threshold):
    """Calcule le prix de l’option via la méthode backward."""
    start = time.time()

    tree = TrinomialTree(market, option, N, exercise)
    tree.build_tree()
    tree.compute_reach_probabilities()

    if optimize == "Oui":
        tree.prune_tree(threshold)

    price = price_backward(tree)
    elapsed = time.time() - start
    return price, elapsed, tree


# -------------------------------------------------------------------------
# 3. Recursive pricing (with cache clearing)
# -------------------------------------------------------------------------
def run_recursive_pricing(market, option, N, exercise, optimize, threshold):
    """
    Calcule le prix de l’option via la méthode récursive.
    Nettoie le cache après le pricing pour éviter les interférences
    avec les appels successifs (utilisés pour les Greeks).
    """
    start = time.time()

    tree = TrinomialTree(market, option, N, exercise)
    tree.build_tree()
    tree.compute_reach_probabilities()

    if optimize == "Oui":
        tree.prune_tree(threshold)

    price = price_recursive(tree)
    elapsed = time.time() - start

    # Clear recursive cache for this tree
    clear_recursive_cache(tree)

    return price, elapsed, tree


# -------------------------------------------------------------------------
# 4. Black-Scholes reference
# -------------------------------------------------------------------------
def run_black_scholes(S0, K, r, sigma, T, is_call):
    """Calcule le prix Black-Scholes (sans dividende explicite ici)."""
    start = time.time()
    price = bs_price(S0, K, r, sigma, T, is_call)
    elapsed = time.time() - start
    return price, elapsed


# -------------------------------------------------------------------------
# 5. Main pricer
# -------------------------------------------------------------------------
def run_pricer():
    """
    Exécute le pricer complet selon la méthode choisie (Backward ou Recursive)
    et compare au modèle de Black-Scholes si applicable.
    """
    (market, option, N, exercise, method, optimize, threshold,
     arbre_stock, arbre_proba, arbre_option, wb, sheet,
     S0, K, r, sigma, T, rho, lam, is_call, exdivdate) = input_parameters()

    # Defensive check: ensure time-to-maturity is positive to avoid math domain errors
    if T is None or T <= 0:
        raise ValueError(f"Temps à maturité invalide T={T}. Vérifiez que la date de maturité est postérieure à la date d'évaluation.")

    # Choix de la méthode d’arbre
    if method == "Backward":
        price, elapsed, tree = run_backward_pricing(market, option, N, exercise, optimize, threshold)
    else:
        price, elapsed, tree = run_recursive_pricing(market, option, N, exercise, optimize, threshold)

    # Black–Scholes
    if exercise == "european":
        bs_val, bs_time = run_black_scholes(S0, K, r, sigma, T, is_call)
    else:
        bs_val, bs_time = None, None

    return {
        "tree_price": price,
        "tree_time": elapsed,
        "bs_price": bs_val,
        "bs_time": bs_time,
        "tree": tree
    }

# example usage:
if __name__ == "__main__":
    results = run_pricer()
    print(f"Trinomial Tree Price: {results['tree_price']} (computed in {results['tree_time']:.4f} seconds)")
    if results['bs_price'] is not None:
        print(f"Black-Scholes Price: {results['bs_price']} (computed in {results['bs_time']:.4f} seconds)")
