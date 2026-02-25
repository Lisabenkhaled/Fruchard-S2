import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price


def main():
    pricing_date = dt.date(2026, 2, 26)
    maturity_date = dt.date(2027, 4, 26)

    market = Market(S0=100.0, r=0.04, sigma=0.25)

    trade = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 6, 21),
        div_amount=3.0,
    )

    params = CorePricingParams(
        n_paths=2000,
        n_steps=200,
        seed=127,
        antithetic=False,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2,
    )

    price, std, se, elapsed = core_price(market, trade, params)

    print("=== OPTION A: American Put K=100 ===")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("95% CI:", price - 1.96 * se, "to", price + 1.96 * se)
    print("Time:", elapsed)


if __name__ == "__main__":
    main()