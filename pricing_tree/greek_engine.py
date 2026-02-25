# pricing_tree/greeks_engine.py

import copy
import numpy as np

from .utils.utils_grecs import OneDimDerivative
from .pricer import price_tree_backward_direct


# -------------------------------------------------------
# 1️⃣ Price getter (Backward only)
# -------------------------------------------------------
def _get_price(market, option, N, exercise, optimize, threshold):
    """
    Pure tree backward pricing call.
    No Excel. No input_parameters.
    """
    out = price_tree_backward_direct(
        S0=market.S0,
        r=market.r,
        sigma=market.sigma,
        K=option.K,
        is_call=option.is_call,
        exercise=exercise,
        pricing_date=market.pricing_date,
        maturity_date=market.T,  # already embedded via T in market
        N=N,
        ex_div_date=None if not market.dividends else market.dividends[0][0],
        div_amount=0.0 if not market.dividends else market.rho * market.S0,
        optimize=optimize,
        threshold=threshold,
        return_tree=False
    )
    return float(out["tree_price"])


# -------------------------------------------------------
# 2️⃣ Generic wrapper for 1D derivatives
# -------------------------------------------------------
def _greek_wrapper(params, x: float) -> float:
    market, option, N, exercise, optimize, threshold, target = params
    m = copy.deepcopy(market)
    setattr(m, target, x)
    return _get_price(m, option, N, exercise, optimize, threshold)


# -------------------------------------------------------
# 3️⃣ Cross derivatives (Vanna, Vomma)
# -------------------------------------------------------
def _finite_diff_2d(market, option, N, exercise, optimize, threshold, base_price):

    S0, sigma0 = market.S0, market.sigma
    hS = max(1e-5, 0.01 * S0)
    hSigma = max(1e-5, 0.005)

    def price_shift(dS=0.0, dSigma=0.0):
        m = copy.deepcopy(market)
        m.S0 = S0 + dS
        m.sigma = max(1e-6, sigma0 + dSigma)
        return _get_price(m, option, N, exercise, optimize, threshold)

    p_up_up     = price_shift(+hS, +hSigma)
    p_up_down   = price_shift(+hS, -hSigma)
    p_down_up   = price_shift(-hS, +hSigma)
    p_down_down = price_shift(-hS, -hSigma)

    vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hSigma)

    p_sig_up   = price_shift(0.0, +hSigma)
    p_sig_down = price_shift(0.0, -hSigma)
    vomma = (p_sig_up - 2 * base_price + p_sig_down) / (hSigma ** 2)

    if not np.isfinite(vanna): vanna = 0.0
    if not np.isfinite(vomma): vomma = 0.0

    return float(vanna), float(vomma)


# -------------------------------------------------------
# 4️⃣ Public reusable greeks engine
# -------------------------------------------------------
def compute_tree_greeks_engine(
    market,
    option,
    N,
    exercise,
    optimize=False,
    threshold=1e-14,
):
    """
    Pure Tree Greeks engine.
    No Excel.
    No input_parameters.
    Fully reusable.
    """

    base_price = _get_price(market, option, N, exercise, optimize, threshold)

    hS = max(1e-5, 0.005 * market.S0)
    hSigma = max(1e-5, 0.005)
    hR = 1e-4
    hT = 1.0 / 365.0

    dS = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "S0"), shift=hS)
    dSigma = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "sigma"), shift=hSigma)
    dR = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "r"), shift=hR)
    dT = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "T"), shift=hT)

    Delta = dS.first(market.S0)
    Gamma = dS.second(market.S0)
    Vega  = dSigma.first(market.sigma)
    Rho   = dR.first(market.r)
    Theta = -dT.first(market.T)

    Vanna, Vomma = _finite_diff_2d(
        market, option, N, exercise, optimize, threshold, base_price
    )

    return {
        "Price": base_price,
        "Delta": Delta,
        "Gamma": Gamma,
        "Vega": Vega,
        "Theta": Theta,
        "Rho": Rho,
        "Vanna": Vanna,
        "Vomma": Vomma,
    }