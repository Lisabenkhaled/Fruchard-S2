import math

def compute_reach_probabilities(tree):
    """
    Calcule et stocke la probabilité d’atteinte (p_reach) de chaque nœud
    dans un arbre trinomial utilisant des Nodes statiques.

    Hypothèses :
      - tree.tree est une liste de niveaux [[Node, Node, ...], ...]
      - Chaque Node possède : p_down, p_mid, p_up
      - Aucun Node ne référence directement tree
    """

    # --- Étape 1 : réinitialisation ---
    for level in tree.tree:
        for node in level:
            if node is not None:
                node.p_reach = 0.0

    # Racine
    root = tree.tree[0][0]
    root.p_reach = 1.0

    # --- Étape 2 : propagation des probabilités vers l’avant ---
    for i in range(tree.N):
        current_level = tree.tree[i]
        next_level = tree.tree[i + 1]
        n_next = len(next_level)

        for j, node in enumerate(current_level):
            if node is None or node.p_reach <= 0.0:
                continue

            pD, pM, pU = node.p_down, node.p_mid, node.p_up
            if pD == 0 and pM == 0 and pU == 0:
                continue

            k_rel = j - i  # position relative du nœud dans le niveau i

            # ---- Propagation individuelle ----
            if pD > 0:
                idx_d = k_rel - 1 + (i + 1)
                if 0 <= idx_d < n_next and next_level[idx_d] is not None:
                    next_level[idx_d].p_reach += node.p_reach * pD

            if pM > 0:
                idx_m = k_rel + (i + 1)
                if 0 <= idx_m < n_next and next_level[idx_m] is not None:
                    next_level[idx_m].p_reach += node.p_reach * pM

            if pU > 0:
                idx_u = k_rel + 1 + (i + 1)
                if 0 <= idx_u < n_next and next_level[idx_u] is not None:
                    next_level[idx_u].p_reach += node.p_reach * pU

    # --- Étape 3 : normalisation (pour éviter dérives d’arrondi) ---
    total = sum(node.p_reach for node in tree.tree[-1] if node)
    if total > 0:
        inv_total = 1.0 / total
        for level in tree.tree:
            for node in level:
                if node is not None:
                    node.p_reach *= inv_total


def prune_tree(tree, threshold=1e-7):
    """
    Supprime (met à None) les nœuds dont la probabilité d’atteinte p_reach
    est inférieure à un seuil donné.

    Paramètres
    ----------
    tree : TrinomialTree
        Arbre trinomial avec les p_reach déjà calculées.
    threshold : float
        Seuil minimal sous lequel un nœud est supprimé.
    verbose : bool
        Si True, affiche le nombre de nœuds supprimés.
    """
    for level in tree.tree:
        for j, node in enumerate(level):
            if node is not None and node.p_reach < threshold:
                level[j] = None

