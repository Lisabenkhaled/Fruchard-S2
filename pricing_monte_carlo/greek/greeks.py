from typing import Callable, Dict, Tuple
import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams
from utils.utils_grecs import OneDimDerivative

PriceFn = Callable[[Market, OptionTrade, CorePricingParams], float]

def _price_spot(params: Tuple[PriceFn, Market, OptionTrade, CorePricingParams], S: float) -> float:
    price_fn, market, trade, core = params
    m = Market(S0=S, r=market.r, sigma=market.sigma)
    return float(price_fn(m, trade, core))

def _price_vol(params: Tuple[PriceFn, Market, OptionTrade, CorePricingParams], sigma: float) -> float:
    price_fn, market, trade, core = params
    m = Market(S0=market.S0, r=market.r, sigma=sigma)
    return float(price_fn(m, trade, core))

def _price_rate(params: Tuple[PriceFn, Market, OptionTrade, CorePricingParams], r: float) -> float:
    price_fn, market, trade, core = params
    m = Market(S0=market.S0, r=r, sigma=market.sigma)
    return float(price_fn(m, trade, core))


# Theta: bump maturity date
def _price_time_days(params: Tuple[PriceFn, Market, OptionTrade, CorePricingParams, int], days: float) -> float:
    price_fn, market, trade, core, base_days = params
    d = int(round(days))

    new_trade = OptionTrade(
        strike=trade.strike,
        is_call=trade.is_call,
        exercise=trade.exercise,
        pricing_date=trade.pricing_date,
        maturity_date=trade.maturity_date + dt.timedelta(days=(d - base_days)),
        ex_div_date=getattr(trade, "ex_div_date", None),
        div_amount=getattr(trade, "div_amount", 0.0),
    )

    return float(price_fn(market, new_trade, core))


# Cross derivative helper (Vanna)
def _cross_second(f: Callable[[float, float], float], x: float, y: float, dx: float, dy: float) -> float:
    return (
        f(x + dx, y + dy)
        - f(x + dx, y - dy)
        - f(x - dx, y + dy)
        + f(x - dx, y - dy)
    ) / (4.0 * dx * dy)

def compute_greeks_vector(price_fn: PriceFn,market: Market,
    trade: OptionTrade,
    params: CorePricingParams,
    shift_spot: float = 0.01,
    shift_vol: float = 0.01,
    shift_rate: float = 1e-4,
    shift_days: int = 1
) -> Dict[str, float]:
    if params.method != "vector":
        raise ValueError("This function is for vector pricers only.")

    # Delta, Gamma
    spot_params = (price_fn, market, trade, params)
    dS = OneDimDerivative(function=_price_spot, other_parameters=spot_params, shift=shift_spot)
    delta = dS.first(market.S0)
    gamma = dS.second(market.S0)

    # Vega, Vomma
    vol_params = (price_fn, market, trade, params)
    dV = OneDimDerivative(function=_price_vol, other_parameters=vol_params, shift=shift_vol)
    vega = dV.first(market.sigma) / 100.0
    vomma = dV.second(market.sigma) / 10000.0

    # Rho
    rate_params = (price_fn, market, trade, params)
    dR = OneDimDerivative(function=_price_rate, other_parameters=rate_params, shift=shift_rate)
    rho = dR.first(market.r)

    # Vanna
    def price_S_sigma(S: float, sigma: float) -> float:
        m = Market(S0=S, r=market.r, sigma=sigma)
        return float(price_fn(m, trade, params))
    
    vanna = _cross_second(price_S_sigma, market.S0, market.sigma, shift_spot, shift_vol) / 100.0

    # Theta
    base_days = 0
    time_params = (price_fn, market, trade, params, base_days)
    dT = OneDimDerivative(function=_price_time_days, other_parameters=time_params, shift=float(shift_days))
    dP_dDays = dT.first(0.0)
    theta = -dP_dDays * 365.0

    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega": float(vega),
        "Theta": float(theta),
        "Rho": float(rho),
        "Vanna": float(vanna),
        "Vomma": float(vomma),
    }