from numba import njit

EPS = 1e-14
MIN_P = 1e-12

@njit(fastmath=True, cache=True)
def clip_and_normalize(pD, pM, pU):
    """
    Nettoie et renormalise les probabilités locales.
    Compatible Numba, logique identique à la version Python.
    """
    # Nettoyage
    if pD < -MIN_P:
        pD = 0.0
    elif pD < 0.0:
        pD = 0.0

    if pM < -MIN_P:
        pM = 0.0
    elif pM < 0.0:
        pM = 0.0

    if pU < -MIN_P:
        pU = 0.0
    elif pU < 0.0:
        pU = 0.0

    # Normalisation
    s = pD + pM + pU
    if s < EPS:
        return 0.0, 1.0, 0.0  
    return pD / s, pM / s, pU / s
