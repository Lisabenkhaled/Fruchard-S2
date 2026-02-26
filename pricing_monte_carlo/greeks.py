from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price


def compute_greeks_vector(
    market: Market,
    trade: OptionTrade,
    params: CorePricingParams,
    eps_spot: float = 0.5,
    eps_vol: float = 0.01,
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
    # VEGA & VOMMA (vol bump)
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
    vomma = (price_vol_up - 2 * price_0 + price_vol_down) / (eps_vol ** 2)

    # =========================
    # VANNA (mixed bump)
    # =========================
    market_up_vol_up = Market(
        S0=market.S0 + eps_spot,
        r=market.r,
        sigma=market.sigma + eps_vol
    )

    market_up_vol_down = Market(
        S0=market.S0 + eps_spot,
        r=market.r,
        sigma=market.sigma - eps_vol
    )

    market_down_vol_up = Market(
        S0=market.S0 - eps_spot,
        r=market.r,
        sigma=market.sigma + eps_vol
    )

    market_down_vol_down = Market(
        S0=market.S0 - eps_spot,
        r=market.r,
        sigma=market.sigma - eps_vol
    )

    price_up_vol_up, _, _, _ = core_price(market_up_vol_up, trade, params)
    price_up_vol_down, _, _, _ = core_price(market_up_vol_down, trade, params)
    price_down_vol_up, _, _, _ = core_price(market_down_vol_up, trade, params)
    price_down_vol_down, _, _, _ = core_price(market_down_vol_down, trade, params)

    vanna = (
        price_up_vol_up
        - price_up_vol_down
        - price_down_vol_up
        + price_down_vol_down
    ) / (4 * eps_spot * eps_vol)

    return {
        "price": price_0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "vomma": vomma,
    }