from .models.market import Market as TreeMarket
from .models.option_trade import Option as TreeOption
from .greek_engine import compute_tree_greeks_engine


def tree_greeks_from_mc(mc_market, mc_trade, N=10_000):

    tree_market = TreeMarket(
        S0=mc_market.S0,
        r=mc_market.r,
        sigma=mc_market.sigma,
        T=mc_trade.T,
        exdivdate=mc_trade.ex_div_time(),
        pricing_date=mc_trade.pricing_date,
        rho=mc_trade.div_amount / mc_market.S0 if mc_trade.div_amount else 0.0,
        lam=0.0,
    )

    tree_option = TreeOption(
        K=mc_trade.strike,
        is_call=mc_trade.is_call,
    )

    return compute_tree_greeks_engine(
        market=tree_market,
        option=tree_option,
        N=N,
        exercise=mc_trade.exercise,
    )