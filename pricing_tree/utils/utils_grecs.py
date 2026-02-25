from typing import Callable

class OneDimDerivative:
    """
    Classe utilitaire pour calculer les dérivées numériques (1re et 2e)
    d’une fonction à une variable, en utilisant la méthode des différences finies
    """

    def __init__(self, function: Callable[[object, float], float], other_parameters: object, shift: float = 1e-4):
        self.f = function               # Fonction à dériver
        self.param = other_parameters   # Paramètres additionnels fixes
        self.shift = shift              # Pas de variation utilisé pour les différences finies

    def first(self, x: float) -> float:
        """
        Calcule la dérivée première
        """
        return (self.f(self.param, x + self.shift) - self.f(self.param, x - self.shift)) / (2 * self.shift)
    
    def second(self, x: float) -> float:
        """
        Calcule la dérivée seconde
        """
        return (self.f(self.param, x + self.shift) - 2 * self.f(self.param, x) + self.f(self.param, x - self.shift)) / (self.shift ** 2)
    