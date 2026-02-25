class Node:
    """
    Classe pour la construction rapide d’un arbre trinomial.

    Chaque noeud contient uniquement les informations essentielles :
      - l’indice temporel (time_index)
      - le prix du sous-jacent (stock_price)
      - la valeur de l’option (option_value)
      - les probabilités locales (p_down, p_mid, p_up)
      - la probabilité d’atteinte (p_reach)
    """

    @staticmethod
    def create(time_index: int, stock_price: float):
        """
        Crée un noeud minimal sans initialisation lourde.
        """
        n = Node.__new__(Node) 
        n.time_index = time_index
        n.stock_price = stock_price
        n.option_value = 0.0
        n.p_down = 0.0
        n.p_mid = 0.0
        n.p_up = 0.0
        n.p_reach = 0.0
        return n

    