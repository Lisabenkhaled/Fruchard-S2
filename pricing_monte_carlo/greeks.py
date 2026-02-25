import datetime as dt
from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price


def compute_greeks_vector(
    market: Market,
    trade: OptionTrade,
    params: CorePricingParams,
    eps_spot: float = 0.5,
    eps_vol: float = 0.01,
    eps_time: float = 1/365,
):
    """
    Greeks by central finite differences.
    Works for:
        - European vector
        - American LS vector
    """

    if params.method != "vector":
        raise ValueError("This function is for vector pricers only.")

    # =========================
    # Base price
    # =========================
    price_0, _, _, _ = core_price(market, trade, params)

    # =========================
    # DELTA & GAMMA (spot bump)
    # =========================
    market_up = Market(
        S0=market.S0 + eps_spot,
        r=market.r,
        sigma=market.sigma
    )

    market_down = Market(
        S0=market.S0 - eps_spot,
        r=market.r,
        sigma=market.sigma
    )

    price_up, _, _, _ = core_price(market_up, trade, params)
    price_down, _, _, _ = core_price(market_down, trade, params)

    delta = (price_up - price_down) / (2 * eps_spot)
    gamma = (price_up - 2 * price_0 + price_down) / (eps_spot ** 2)

    # =========================
    # VEGA (vol bump)
    # =========================
    market_vol_up = Market(
        S0=market.S0,
        r=market.r,
        sigma=market.sigma + eps_vol
    )

    market_vol_down = Market(
        S0=market.S0,
        r=market.r,
        sigma=market.sigma - eps_vol
    )

    price_vol_up, _, _, _ = core_price(market_vol_up, trade, params)
    price_vol_down, _, _, _ = core_price(market_vol_down, trade, params)

    vega = (price_vol_up - price_vol_down) / (2 * eps_vol)

    # =========================
    # THETA (1 day decay)
    # =========================
    new_maturity = trade.maturity_date - dt.timedelta(days=1)

    trade_shorter = OptionTrade(
        strike=trade.strike,
        is_call=trade.is_call,
        exercise=trade.exercise,
        pricing_date=trade.pricing_date,
        maturity_date=new_maturity,
        q=trade.q,
        ex_div_date=trade.ex_div_date,
        div_amount=trade.div_amount,
    )

    price_shorter, _, _, _ = core_price(market, trade_shorter, params)

    theta = (price_shorter - price_0) / eps_time

    return {
        "price": price_0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }