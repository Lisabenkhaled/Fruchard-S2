import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _backward_kernel(V_next, pD, pM, pU, df, exer, is_american):
    """
    Numba pour la récurrence arrière dans un arbre trinomial.

    Paramètres
    ----------
    V_next : np.ndarray
        Valeurs de l’option au niveau i+1 (niveau futur)
    pD, pM, pU : np.ndarray
        Probabilités locales (down, mid, up) au niveau i
    df : float
        Facteur d’actualisation exp(-r * dt)
    exer : np.ndarray
        Valeur d’exercice immédiate (payoff)
    is_american : bool
        True si option américaine (exercice anticipé possible)

    Retour
    ------
    V_new : np.ndarray
        Valeurs de l’option au niveau i (niveau courant)
    """

    N = len(pD)
    V_new = np.zeros(N)

    for j in range(N):
        Vd = V_next[j] if j < len(V_next) else 0.0
        Vm = V_next[j + 1] if j + 1 < len(V_next) else 0.0
        Vu = V_next[j + 2] if j + 2 < len(V_next) else 0.0

        hold = df * (pD[j] * Vd + pM[j] * Vm + pU[j] * Vu)

        V_new[j] = max(hold, exer[j]) if is_american else hold

    return V_new


def price_backward(tree):
    """
    Calcule le prix d'une option via la méthode de récurrence arrière
    """

    option = tree.option
    df = tree.df
    is_american = (tree.exercise == "american")
    N = tree.N
    tree_nodes = tree.tree

    # Payoff à maturité)
    last_level = tree_nodes[-1]
    V = np.array(
        [option.payoff(node.stock_price) if node is not None else 0.0
         for node in last_level],
        dtype=np.float64
    )

    # Boucle de récurrence arrière
    for i in range(N - 1, -1, -1): 
        level = tree_nodes[i]
        n = len(level)

        pD = np.empty(n, dtype=np.float64)
        pM = np.empty(n, dtype=np.float64)
        pU = np.empty(n, dtype=np.float64)
        exer = np.empty(n, dtype=np.float64)

        for j, node in enumerate(level):
            if node is None:
                pD[j] = pM[j] = pU[j] = 0.0
                exer[j] = 0.0
            else:
                pD[j], pM[j], pU[j] = node.p_down, node.p_mid, node.p_up
                exer[j] = option.payoff(node.stock_price)

        V = _backward_kernel(V, pD, pM, pU, df, exer, is_american)

    return float(V[0])
