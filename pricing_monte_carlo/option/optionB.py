import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams, core_price


def bermudan_exercise_steps(pricing_date, maturity_date, n_steps):
    dates = []
    cur = pricing_date

    # 26 of each month
    while cur <= maturity_date:
        d = dt.date(cur.year, cur.month, 26)
        if pricing_date <= d <= maturity_date:
            dates.append(d)

        # next month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

    if maturity_date not in dates:
        dates.append(maturity_date)

    total_days = (maturity_date - pricing_date).days
    steps = []

    for d in dates:
        tau = (d - pricing_date).days / total_days
        idx = int(round(tau * n_steps))
        steps.append(min(max(idx, 0), n_steps))

    return sorted(set(steps))


if __name__ == "__main__":

    pricing_date = dt.date(2026, 2, 26)
    maturity_date = dt.date(2027, 4, 26)

    market = Market(S0=100.0, r=0.04, sigma=0.25)

    trade = OptionTrade(
        strike=102.0,
        is_call=False,
        exercise="bermudan",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 6, 21),
        div_amount=3.0,
    )

    n_steps = 365
    exercise_steps = bermudan_exercise_steps(pricing_date, maturity_date, n_steps)

    params = CorePricingParams(
        n_paths=2000,
        n_steps=n_steps,
        seed=127,
        antithetic=False,
        method="vector",
        basis="laguerre",
        degree=2,
        exercise_steps=exercise_steps,
    )

    price, std, se, elapsed = core_price(market, trade, params)

    print("\n=== OPTION B: Bermudan Put K=102 ===")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("95% CI:", price - 1.96 * se, "to", price + 1.96 * se)
    print("Time:", elapsed)