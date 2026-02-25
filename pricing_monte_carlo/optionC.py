import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price


if __name__ == "__main__":

    pricing_date = dt.date(2026, 2, 26)
    maturity_date = dt.date(2027, 4, 26)

    market = Market(S0=100.0, r=0.04, sigma=0.25)

    trade = OptionTrade(
        strike=99.0,  # not used directly but keep consistent
        is_call=False,
        exercise="digital_american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 6, 21),
        div_amount=3.0,
    )

    params = CorePricingParams(
        n_paths=200_000,
        n_steps=365,
        seed=127,
        antithetic=False,
        method="vector",
        basis="laguerre",
        degree=2,
        digital_strike=99.0,
        digital_payout=1.0,
    )

    price, std, se, elapsed = core_price(market, trade, params)

    print("\n=== OPTION C: American Digital K=99 ===")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("95% CI:", price - 1.96 * se, "to", price + 1.96 * se)
    print("Time:", elapsed)