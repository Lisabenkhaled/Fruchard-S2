import math
import numpy as np
from numba import njit
from utils.utils_constants import clip_and_normalize, MIN_P, EPS


@njit(fastmath=True, cache=True)

# compute variance
def _compute_variance(S_i_k: float, exp_r_dt: float, exp_sig2_dt: float) -> float:
    exp2r = exp_r_dt * exp_r_dt
    return (S_i_k * S_i_k) * exp2r * (exp_sig2_dt - 1.0)

@njit(fastmath=True, cache=True)
def _compute_kprime(E: float, trunk_next: float, loga: float) -> int:
    base_next = max(MIN_P, trunk_next)
    denom = loga if abs(loga) > EPS else EPS
    return int(round(math.log(max(E, MIN_P) / base_next) / denom))

@njit(fastmath=True, cache=True)
# probabilities
def _degenerate_probabilities(S_i_k: float, E: float, exp_r_dt: float, 
                              trunk_next: float, a: float, loga: float) -> tuple[float, float, float, int]:
    base_next = trunk_next
    if base_next < MIN_P:
        base_next = max(MIN_P, S_i_k * exp_r_dt)

    denom = loga if abs(loga) > EPS else EPS
    kprime = int(round(math.log(max(E, MIN_P) / base_next) / denom))
    S_mid = base_next * (a ** kprime)

    dE1 = abs(E - S_mid / a)
    dE2 = abs(E - S_mid)
    dE3 = abs(E - S_mid * a)

    # Gestion des valeurs 
    if dE1 <= dE2 and dE1 <= dE3:
        return 1.0, 0.0, 0.0, kprime
    if dE2 <= dE1 and dE2 <= dE3:
        return 0.0, 1.0, 0.0, kprime
    return 0.0, 0.0, 1.0, kprime

@njit(fastmath=True, cache=True)
# recentrer kprime
def _recenter_kprime(E: float, i: int, kprime: int, a: float, base_next: float) -> tuple[float, int]:
    max_shift = i + 1
    if kprime > max_shift:
        kprime = max_shift
    elif kprime < -max_shift:
        kprime = -max_shift

    # Recentrage de la position centrale si l’espérance sort des bornes
    S_mid = base_next * (a ** kprime)
    S_up, S_down = S_mid * a, S_mid / a
    lower, upper = 0.5 * (S_mid + S_down), 0.5 * (S_mid + S_up)
    shifts = 0

    # boucle while
    while (E > upper or E < lower) and shifts < 10:
        if E > upper:
            kprime += 1
            S_mid *= a
        else:
            kprime -= 1
            S_mid /= a
        S_up, S_down = S_mid * a, S_mid / a
        lower, upper = 0.5 * (S_mid + S_down), 0.5 * (S_mid + S_up)
        shifts += 1
    return S_mid, kprime

@njit(fastmath=True, cache=True)
# moment proba
def _moment_probabilities(E: float, V: float, S_mid: float, 
                          a: float, a2: float, exp_sig2_dt: float, 
                          has_dividend: bool) -> tuple[float, float, float]:
    m1 = E / S_mid
    m2 = (V + E * E) / (S_mid * S_mid)
    den = (1.0 - a) * ((1.0 / a2) - 1.0)
    if abs(den) < EPS:
        return 0.0, 1.0, 0.0

    if not has_dividend:
        # Cas sans dividende : formule simplifiée
        p_down = (exp_sig2_dt - 1.0) / den
        p_up = p_down / a
        p_mid = 1.0 - p_up - p_down
        return p_down, p_mid, p_up

    num = (m2 - 1.0) - (a + 1.0) * (m1 - 1.0)
    p_down = num / den
    p_up = (m1 - 1.0 - ((1.0 / a) - 1.0) * p_down) / (a - 1.0)
    p_mid = 1.0 - p_up - p_down
    return clip_and_normalize(p_down, p_mid, p_up)


@njit(fastmath=True, cache=True)
# Local proba
def local_probabilities(
    S_i_k: float,i: int,dt: float,r: float,
    a: float,exp_sig2_dt: float,trunk_next: float,
    div: float,has_dividend: bool) -> tuple[float, float, float, int]:
    """
    Calcule les probabilités locales d’un arbre trinomial à chaque nœud.
    """
    exp_r_dt = math.exp(r * dt)
    E = S_i_k * exp_r_dt - div if has_dividend else S_i_k * exp_r_dt
    V = _compute_variance(S_i_k, exp_r_dt, exp_sig2_dt)

    a2 = a * a
    loga = math.log(a)

    # Handling if V very low
    if V < 1e-18:
        return _degenerate_probabilities(S_i_k, E, exp_r_dt, trunk_next, a, loga)

    base_next = max(MIN_P, trunk_next)
    kprime = _compute_kprime(E, base_next, loga)
    S_mid, kprime = _recenter_kprime(E, i, kprime, a, base_next)

    p_down, p_mid, p_up = _moment_probabilities(E, V, S_mid, a, a2, exp_sig2_dt, has_dividend)

    return p_down, p_mid, p_up, kprime
