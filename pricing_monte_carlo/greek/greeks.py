from typing import Callable
from dataclasses import replace
from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams

# Type d'une fonction de pricing:
# elle prend (market, trade, params) et renvoie un float (le prix)
PriceFn = Callable[[Market, OptionTrade, CorePricingParams], float]


def compute_greeks_vector(
    price_fn: PriceFn,
    market: Market,
    trade: OptionTrade,
    params: CorePricingParams,
    eps_spot: float = 0.01,
    eps_vol: float = 0.01,
) -> dict[str, float]:

    if params.method != "vector":
        raise ValueError("This function is for vector pricers only.")

    # Prix de base
    price_0 = float(price_fn(market, trade, params))

    # =========================
    # Delta & Gamma (spot bump)
    # =========================
    m_up = Market(S0=market.S0 + eps_spot, r=market.r, sigma=market.sigma)
    m_dn = Market(S0=market.S0 - eps_spot, r=market.r, sigma=market.sigma)

    price_up = float(price_fn(m_up, trade, params))
    price_dn = float(price_fn(m_dn, trade, params))

    delta = (price_up - price_dn) / (2.0 * eps_spot)
    gamma = (price_up - 2.0 * price_0 + price_dn) / (eps_spot ** 2)

    # =========================
    # Vega & Vomma (vol bump)
    # =========================
    mv_up = Market(S0=market.S0, r=market.r, sigma=market.sigma + eps_vol)
    mv_dn = Market(S0=market.S0, r=market.r, sigma=market.sigma - eps_vol)

    price_v_up = float(price_fn(mv_up, trade, params))
    price_v_dn = float(price_fn(mv_dn, trade, params))

    vega = ((price_v_up - price_v_dn) / (2.0 * eps_vol)) / 100
    vomma = ((price_v_up - 2.0 * price_0 + price_v_dn) / (eps_vol ** 2)) / 10000

    # =========================
    # Vanna (spot+vol croisé)
    # =========================
    m_up_v_up = Market(S0=market.S0 + eps_spot, r=market.r, sigma=market.sigma + eps_vol)
    m_up_v_dn = Market(S0=market.S0 + eps_spot, r=market.r, sigma=market.sigma - eps_vol)
    m_dn_v_up = Market(S0=market.S0 - eps_spot, r=market.r, sigma=market.sigma + eps_vol)
    m_dn_v_dn = Market(S0=market.S0 - eps_spot, r=market.r, sigma=market.sigma - eps_vol)

    p_u_u = float(price_fn(m_up_v_up, trade, params))
    p_u_d = float(price_fn(m_up_v_dn, trade, params))
    p_d_u = float(price_fn(m_dn_v_up, trade, params))
    p_d_d = float(price_fn(m_dn_v_dn, trade, params))

    vanna = ((p_u_u - p_u_d - p_d_u + p_d_d) / (4.0 * eps_spot * eps_vol))/100

    return {
        "price": price_0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "vomma": vomma,
    }