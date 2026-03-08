# pricing_tree/greeks_engine.py
from __future__ import annotations
import copy
import numpy as np
from typing import Any
from .utils.utils_grecs import OneDimDerivative
from .pricer import price_tree_backward_direct
from .models.market import Market
from .models.option_trade import Option
from .utils.utils_grecs import OneDimDerivative

GreekParams = tuple[Market, Option, int, str, bool, float, str]
# get price function
def _get_price(
    market: Market,option: Option,
    N: int,exercise: str,
    optimize: bool,threshold: float) -> float:
    out = price_tree_backward_direct(S0=market.S0,
        r=market.r,sigma=market.sigma,K=option.K,
        is_call=option.is_call,exercise=exercise,
        pricing_date=market.pricing_date,maturity_date=market.T,
        N=N,
        ex_div_date=None if not market.dividends else market.dividends[0][0],
        div_amount=0.0 if not market.dividends else market.rho * market.S0,
        optimize=optimize,threshold=threshold, return_tree=False    )
    return float(out["tree_price"])

# greek wrapper
def _greek_wrapper(params: GreekParams, x: float) -> float:
    market, option, N, exercise, optimize, threshold, target = params
    m = copy.deepcopy(market)
    setattr(m, target, x)
    return _get_price(m, option, N, exercise, optimize, threshold)

# finite differences
def _finite_diff_2d(
    market: Market,option: Option,
    N: int,exercise: str,optimize: bool,
    threshold: float, base_price: float) -> tuple[float, float]:

    S0, sigma0 = market.S0, market.sigma
    hS = 0.01
    hSigma = 0.01
    
    # shift
    def price_shift(dS: float = 0.0, dSigma: float = 0.0) -> float:
        m = copy.deepcopy(market)
        m.S0 = S0 + dS
        m.sigma = max(1e-6, sigma0 + dSigma)
        return _get_price(m, option, N, exercise, optimize, threshold)

    p_up_up     = price_shift(+hS, +hSigma)
    p_up_down   = price_shift(+hS, -hSigma)
    p_down_up   = price_shift(-hS, +hSigma)
    p_down_down = price_shift(-hS, -hSigma)

    # vanna
    vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hSigma)

    p_sig_up   = price_shift(0.0, +hSigma)
    p_sig_down = price_shift(0.0, -hSigma)
    vomma = (p_sig_up - 2 * base_price + p_sig_down) / (hSigma ** 2)

    if not np.isfinite(vanna): vanna = 0.0
    if not np.isfinite(vomma): vomma = 0.0

    return float(vanna) / 100.0, float(vomma) / 10000.0

# tree greeks
def compute_tree_greeks_engine(
    market: Market,option: Option,
    N: int,exercise: str,
    optimize: bool = False,threshold: float = 1e-14) -> dict[str, Any]:
    base_price = _get_price(market, option, N, exercise, optimize, threshold)

    hS = max(1e-4, 0.01 * market.S0);hSigma = 0.01
    hR = 1e-4;hT = 1.0 / 365.0

    # les dérivées
    dS = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "S0"), shift=hS)
    dSigma = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "sigma"), shift=hSigma)
    dR = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "r"), shift=hR)
    dT = OneDimDerivative(_greek_wrapper,
        (market, option, N, exercise, optimize, threshold, "T"), shift=hT)

    # greeks calcul
    Delta = dS.first(market.S0)
    Gamma = dS.second(market.S0)
    Vega  = dSigma.first(market.sigma) / 100.0
    Rho   = dR.first(market.r)
    Theta = -dT.first(market.T)

    Vanna, Vomma = _finite_diff_2d(
        market, option, N, exercise, optimize, threshold, base_price
    )

    return {"Delta": Delta, "Gamma": Gamma,"Vega": Vega,
        "Theta": Theta,"Rho": Rho,"Vanna": Vanna,"Vomma": Vomma}