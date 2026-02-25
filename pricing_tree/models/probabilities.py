import math
import numpy as np
from numba import njit
from utils.utils_constants import clip_and_normalize, MIN_P, EPS


@njit(fastmath=True, cache=True)
def local_probabilities(
    S_i_k: float,
    i: int,
    dt: float,
    r: float,
    a: float,
    exp_sig2_dt: float,
    trunk_next: float,
    div: float,
    has_dividend: bool,
) -> tuple:
    """
    Calcule les probabilités locales d’un arbre trinomial à chaque nœud.

    Paramètres
    ----------
    S_i_k : float
        Prix du sous-jacent au nœud (i, k).
    i : int
        Indice temporel courant.
    dt : float
        Taille du pas de temps.
    r : float
        Taux sans risque.
    a : float
        Facteur multiplicatif du mouvement (S_up = S * a).
    exp_sig2_dt : float
        exp(sigma² * dt) pré-calculé pour la stabilité numérique.
    trunk_next : float
        Prix médian du niveau suivant.
    div : float
        Dividende versé pendant l’intervalle (si applicable).
    has_dividend : bool
        Indique si un dividende est pris en compte à cette étape.

    Retourne
    --------
    tuple : (p_down, p_mid, p_up, kprime)
        Probabilités locales et position centrale k′.
    """

    exp_r_dt = math.exp(r * dt)
    a2 = a * a
    exp2r = exp_r_dt * exp_r_dt
    loga = math.log(a)

    E = S_i_k * exp_r_dt - div if has_dividend else S_i_k * exp_r_dt
    V = (S_i_k * S_i_k) * exp2r * (exp_sig2_dt - 1.0)

    if V < 1e-18:
        base_next = trunk_next
        if base_next < MIN_P:
            base_next = max(MIN_P, S_i_k * exp_r_dt)
        denom = loga if abs(loga) > EPS else EPS
        kprime = int(round(math.log(max(E, MIN_P) / base_next) / denom))
        S_mid = base_next * (a ** kprime)

        # Recherche du voisin le plus proche de E
        dE1 = abs(E - S_mid / a)
        dE2 = abs(E - S_mid)
        dE3 = abs(E - S_mid * a)

        if dE1 <= dE2 and dE1 <= dE3:
            return 1.0, 0.0, 0.0, kprime
        elif dE2 <= dE1 and dE2 <= dE3:
            return 0.0, 1.0, 0.0, kprime
        else:
            return 0.0, 0.0, 1.0, kprime

    base_next = max(MIN_P, trunk_next)
    denom = loga if abs(loga) > EPS else EPS
    kprime = int(round(math.log(max(E, MIN_P) / base_next) / denom))

    # Limitation de k′ pour rester dans les bornes de l’arbre
    if kprime > (i + 1):
        kprime = i + 1
    elif kprime < -(i + 1):
        kprime = -(i + 1)

    S_mid = base_next * (a ** kprime)

    # Recentrage de la position centrale si l’espérance sort des bornes
    S_up = S_mid * a
    S_down = S_mid / a
    lower = 0.5 * (S_mid + S_down)
    upper = 0.5 * (S_mid + S_up)
    shifts = 0

    while (E > upper or E < lower) and shifts < 10:
        if E > upper:
            kprime += 1
            S_mid *= a
        else:
            kprime -= 1
            S_mid /= a
        S_up = S_mid * a
        S_down = S_mid / a
        lower = 0.5 * (S_mid + S_down)
        upper = 0.5 * (S_mid + S_up)
        shifts += 1

    m1 = E / S_mid
    m2 = (V + E * E) / (S_mid * S_mid)
    den = (1.0 - a) * ((1.0 / a2) - 1.0)
    if abs(den) < EPS:
        return 0.0, 1.0, 0.0, kprime

    if not has_dividend:
        # Cas sans dividende : formule simplifiée
        p_down = (exp_sig2_dt - 1.0) / den
        p_up = p_down / a
        p_mid = 1.0 - p_up - p_down
    else:
        # Cas avec dividende : ajustement des moments
        num = (m2 - 1.0) - (a + 1.0) * (m1 - 1.0)
        p_down = num / den
        p_up = (m1 - 1.0 - ((1.0 / a) - 1.0) * p_down) / (a - 1.0)
        p_mid = 1.0 - p_up - p_down
        p_down, p_mid, p_up = clip_and_normalize(p_down, p_mid, p_up)

    return p_down, p_mid, p_up, kprime
