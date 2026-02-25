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


def bs_greeks(S, K, r, sigma, T, is_call=True):
    """
    Calcule les principaux Greeks du modèle Black-Scholes (sans dividendes).

    Retourne un dictionnaire : Delta, Gamma, Vega, Theta, Rho, Vanna, Vomma
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)

    pdf_d1 = norm.pdf(d_1)
    Nd1, Nd2 = norm.cdf(d_1), norm.cdf(d_2)
    Nmd1, Nmd2 = norm.cdf(-d_1), norm.cdf(-d_2)

    df_r = exp(-r * T)

    # Delta 
    delta = Nd1 if is_call else -Nmd1

    # Gamma et Vega 
    gamma = pdf_d1 / (S * sigma * sqrt(T))
    vega = S * pdf_d1 * sqrt(T)

    # Theta
    term1 = -(S * pdf_d1 * sigma) / (2 * sqrt(T))
    if is_call:
        theta = term1 - r * K * df_r * Nd2
        rho_val = K * T * df_r * Nd2
    else:
        theta = term1 + r * K * df_r * Nmd2
        rho_val = -K * T * df_r * Nmd2

    # Second-order Greeks
    vomma = vega * d_1 * d_2 / sigma
    vanna = vega * (1 - d_1 / (sigma * sqrt(T))) / S

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho_val,
        "Vanna": vanna,
        "Vomma": vomma,
    }
