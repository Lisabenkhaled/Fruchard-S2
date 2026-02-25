import math
from models.node import Node
from models.option_trade import Option
from models.pruning import compute_reach_probabilities, prune_tree
from utils.utils_dividends import get_dividend_on_step
from utils.utils_constants import MIN_P
from models.probabilities import local_probabilities


class TrinomialTree:
    """
    Classe principale pour la construction et la gestion d’un arbre trinomial.
    Chaque noeud de l’arbre est une instance de Node

    L’arbre est utilisé pour :
      - le calcul des prix d’options par récurrence (backward/recursive)
      - le calcul des probabilités d’atteinte (p_reach)
      - l’affichage des niveaux dans Excel
    """

    def __init__(self, market, option: Option, N: int, exercise="european"):
        """
        Initialise les paramètres du modèle trinomial.

        Paramètres
        ----------
        market : Market
            Marché contenant S0, r, sigma, T, etc.
        option : Option
            Objet représentant les caractéristiques de l’option (K, type, etc.).
        N : int
            Nombre d’étapes temporelles de l’arbre.
        exercise : str
            Type d’option ("european" ou "american").
        """
        self.market = market
        self.option = option
        self.N = N
        self.exercise = exercise.lower()

        # Paramètres du marché 
        self.dt = market.T / N             
        self.r = market.r                   
        self.sigma = market.sigma           
        self.df = math.exp(-self.r * self.dt)  

        #  Paramètres du modèle trinomial
        self.alpha = math.exp(self.sigma * math.sqrt(3.0 * self.dt))  # facteur de hausse
        self.exp_sig2_dt = math.exp(self.sigma ** 2 * self.dt)
        self.log_alpha = math.log(self.alpha)
        self.exp_r_dt = math.exp(self.r * self.dt)

        # Structures internes
        self.tree = []         # Arbre complet (liste de listes de Node)
        self.proba_tree = []   # Liste des probabilités locales pour chaque niveau
        self.trunk = [0.0] * (self.N + 1)  # Prix médian par étape

    def build_tree(self):
        """
        Construit l’arbre trinomial en 3 étapes :
          1. Création du noeud racine.
          2. Construction progressive des niveaux de prix.
          3. Calcul des probabilités locales p_down, p_mid, p_up.
        """
        S0 = self.market.S0
        dt, r = self.dt, self.r
        exp_r_dt = self.exp_r_dt
        alpha, log_alpha = self.alpha, self.log_alpha
        exp_sig2_dt = self.exp_sig2_dt
        N = self.N

        # Création du noeud racine
        root = Node.create(0, S0)
        self.tree = [[root]]
        self.trunk[0] = S0

        # Construction des niveaux suivants
        for i in range(1, N + 1):
            prev_mid = self.trunk[i - 1]
            t_i, t_ip1 = (i - 1) * dt, i * dt

            div, has_dividend = get_dividend_on_step(self.market, t_i, t_ip1, prev_mid)

            # Calcul du prix médian du niveau suivant
            mid_i = prev_mid * exp_r_dt - div
            if mid_i < MIN_P:
                mid_i = MIN_P
            self.trunk[i] = mid_i

            # Création des noeuds du niveau i
            level_nodes = []
            for k in range(-i, i + 1):
                S = mid_i * math.exp(log_alpha * k)
                node = Node.create(i, S)
                level_nodes.append(node)
            self.tree.append(level_nodes)

        # Calcul des probabilités locales
        self.proba_tree = []
        for i, level in enumerate(self.tree[:-1]): 
            t_i, t_ip1 = i * dt, (i + 1) * dt
            level_proba = []

            # Référence médiane du niveau
            mid_ref = self.trunk[i]
            div, has_dividend = get_dividend_on_step(self.market, t_i, t_ip1, mid_ref)
            trunk_next = self.trunk[i + 1]

            # Calcul des probabilités locales pour chaque noeud
            for node in level:
                S = node.stock_price
                pD, pM, pU, kprime = local_probabilities(
                    S_i_k=S,
                    i=i,
                    dt=dt,
                    r=r,
                    a=alpha,
                    exp_sig2_dt=exp_sig2_dt,
                    trunk_next=trunk_next,
                    div=div,
                    has_dividend=has_dividend,
                )
                # Sauvegarde des probabilités dans le nœud
                node.p_down, node.p_mid, node.p_up = pD, pM, pU
                level_proba.append((pD, pM, pU, kprime))

            self.proba_tree.append(level_proba)

    def compute_reach_probabilities(self):
        """
        Calcule les probabilités d’atteinte p_reach pour chaque noeud
        en appelant la fonction de propagation dédiée.
        """
        compute_reach_probabilities(self)

    def prune_tree(self, threshold=1e-7):
        """
        Supprime les noeuds dont la probabilité d’atteinte p_reach
        est inférieure à un seuil donné.
        """
        prune_tree(self, threshold)

    def to_levels_for_excel(self):
        """
        Renvoie l’arbre sous une forme compatible avec display_trees(),
        pour affichage dans Excel.
        """
        return self.tree
