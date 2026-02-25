import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dividend import DividendPolicy


class Market:
    """
    Classe Market : contient les paramètres de marché nécessaires
    à la construction et à l’évaluation des arbres binomiaux ou trinômiaux.

    Attributs principaux :
    ----------------------
    S0 : float
        Prix initial du sous-jacent.
    r : float
        Taux sans risque (en continu).
    sigma : float
        Volatilité du sous-jacent.
    T : float
        Maturité (en années).
    rho, lam : float
        Paramètres du modèle de dividende (si applicable).
    pricing_date : float ou None
        Date actuelle (t = 0 par défaut).
    dividends : list
        Liste contenant les politiques de dividende (peut être vide).
    """

    def __init__(self, S0: float, r: float, sigma: float, T: float,
                 exdivdate=None, pricing_date=None,
                 rho: float = 0.0, lam: float = 0.0):
        """
        Initialise les paramètres du marché.
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.rho = rho
        self.lam = lam
        self.pricing_date = pricing_date

        self.dividends = []
        if exdivdate is not None:
            policy = DividendPolicy(rho, lam, t0=0.0)
            self.dividends.append((float(exdivdate), policy))

    def has_dividend(self) -> bool:
        """
        Retourne True si un dividende est défini dans le marché.
        """
        return bool(self.dividends)

