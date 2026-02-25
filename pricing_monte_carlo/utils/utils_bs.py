from math import exp, sqrt, log
from scipy.stats import norm


def d1(S, K, r, sigma, T):
    """
    Calcule le paramètre d1 du modèle de Black-Scholes (sans dividendes).
    """
    return (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d2(S, K, r, sigma, T):
    """
    Calcule le paramètre d2 = d1 - sigma * sqrt(T).
    """
    return d1(S, K, r, sigma, T) - sigma * sqrt(T)


def bs_price(S, K, r, sigma, T, is_call=True):
    """
    Prix d'une option européenne selon le modèle de Black-Scholes (sans dividendes).

    Paramètres
    ----------
    S : float
        Prix spot du sous-jacent
    K : float
        Prix d'exercice
    r : float
        Taux sans risque
    sigma : float
        Volatilité du sous-jacent
    T : float
        Temps jusqu'à maturité (en années)
    is_call : bool
        True pour un call, False pour un put
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    df_r = exp(-r * T)  # facteur d’actualisation

    if is_call:
        price = S * norm.cdf(d_1) - K * df_r * norm.cdf(d_2)
    else:
        price = K * df_r * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    return price