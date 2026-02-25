import math

class DividendPolicy:
    """
    Modèle de dividende discret basé sur une décroissance exponentielle.
    """

    def __init__(self, rho: float, lam: float, t0: float = 0.0):
        self.rho = rho
        self.lam = lam
        self.t0 = float(t0)

    def amount(self, t: float, S: float, S0: float) -> float:
        """
        Calcule le montant du dividende à la date t.
        """
        return self.rho * (S0 * math.exp(-self.lam * (t - self.t0)) + S * (1 - math.exp(-self.lam * (t - self.t0))))
