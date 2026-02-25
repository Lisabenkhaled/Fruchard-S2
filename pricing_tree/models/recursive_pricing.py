import functools
import sys
sys.setrecursionlimit(3000)  

import functools

# cache global, isolé par arbre
_GLOBAL_RECURSIVE_CACHE = {}

def recursive_cache():
    """
    Décorateur avec cache séparé par arbre.
    Chaque arbre (TrinomialTree) dispose de son propre cache,
    qui est automatiquement créé puis supprimé après le pricing.
    Cela évite les dérivées nulles lors des tests de Greeks.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(tree, i=0, k=0, cache=None):
            tree_id = id(tree)

            # Crée un cache spécifique à cet arbre
            if tree_id not in _GLOBAL_RECURSIVE_CACHE:
                _GLOBAL_RECURSIVE_CACHE[tree_id] = {}

            local_cache = _GLOBAL_RECURSIVE_CACHE[tree_id]
            key = (i, k)

            if key in local_cache:
                return local_cache[key]

            result = func(tree, i, k, local_cache)
            local_cache[key] = result
            return result
        return wrapper
    return decorator


def clear_recursive_cache(tree=None):
    """
    Supprime le cache global ou celui d’un arbre spécifique.
    À appeler à la fin de chaque pricing pour éviter les interférences
    entre les appels successifs (ex: calculs de Greeks).
    """
    if tree is None:
        _GLOBAL_RECURSIVE_CACHE.clear()
    else:
        _GLOBAL_RECURSIVE_CACHE.pop(id(tree), None)

@recursive_cache()
def price_recursive(tree, i=0, k=0, cache=None):
    """
    Calcule le prix d’une option par la méthode récursive sur un arbre trinomial.

    Paramètres
    ----------
    tree : TrinomialTree
        Arbre trinomial déjà construit.
    i : int
        Indice de l’étape temporelle (0 = racine).
    k : int
        Décalage relatif au nœud central (indice horizontal).
    cache : dict
        Cache partagé entre les appels récursifs (par arbre).

    Retour
    -------
    float
        Valeur de l’option au nœud (i, k).
    """

    # Vérifie si le résultat est déjà en cache
    key = (i, k)
    if cache is not None and key in cache:
        return cache[key]

    # --- Cas terminal : maturité ---
    if i >= tree.N:
        idx = i + k
        node = tree.tree[i][idx] if 0 <= idx < len(tree.tree[i]) else None
        value = 0.0 if node is None else tree.option.payoff(node.stock_price)
        if cache is not None:
            cache[key] = value
        return value

    # --- Nœud courant ---
    idx = i + k
    if idx < 0 or idx >= len(tree.tree[i]):
        if cache is not None:
            cache[key] = 0.0
        return 0.0

    node = tree.tree[i][idx]
    if node is None:
        if cache is not None:
            cache[key] = 0.0
        return 0.0

    # --- Données locales ---
    S = node.stock_price
    pD, pM, pU = node.p_down, node.p_mid, node.p_up
    df = tree.df

    # --- Appels récursifs ---
    v_down = price_recursive(tree, i + 1, k - 1)
    v_mid  = price_recursive(tree, i + 1, k)
    v_up   = price_recursive(tree, i + 1, k + 1)

    continuation = df * (pD * v_down + pM * v_mid + pU * v_up)

    # --- Cas américain : comparaison avec exercice anticipé ---
    if tree.exercise == "american":
        exercise_val = tree.option.payoff(S)
        value = max(exercise_val, continuation)
    else:
        value = continuation

    # --- Sauvegarde dans le cache ---
    if cache is not None:
        cache[key] = value

    return value
