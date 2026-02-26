import time
from dataclasses import dataclass, field
from typing import Literal, Tuple, Sequence

from model.market import Market
from model.option import OptionTrade
from utils.utils_stats import sample_std, standard_error, sample_std_anti, standard_error_anti

from model.mc_pricer import (
    # European
    price_european_naive_mc_vector,
    price_european_naive_mc_scalar,
    # American naive
    price_american_naive_mc_vector,
    price_american_naive_mc_scalar,
    # American Longstaff–Schwartz
    price_american_ls_vector,
    price_american_ls_scalar,
)

from model.mc_pricer_exotics import (
    price_bermudan_put_ls_vector,
    price_bermudan_put_ls_scalar,
    price_american_digital_ls_vector,
    price_american_digital_ls_scalar,
)

from model.mc_pricer_digital import (
    price_american_digital_first_hit_vector,
    price_american_digital_first_hit_scalar,
)

Method = Literal["vector", "scalar"]
AmericanAlgo = Literal["naive", "ls"]
Basis = Literal["power", "laguerre"]

@dataclass(frozen=True)
class CorePricingParams:
    n_paths: int
    n_steps: int
    seed: int = 0
    antithetic: bool = False
    method: Method = "vector"
    american_algo: AmericanAlgo = "ls"
    basis: Basis = "laguerre"
    degree: int = 2

    # EXOTICS
    exercise_steps: Sequence[int] = field(default_factory=tuple)  # for bermudan
    digital_strike: float | None = None                           # for digital
    digital_payout: float = 1.0


def core_price(
    market: Market,
    trade: OptionTrade,
    p: CorePricingParams
) -> Tuple[float, float, float, float]:
    """
    Returns:
        price,
        std (Monte Carlo standard deviation),
        std_error,
        elapsed_time
    """

    if p.n_paths <= 0 or p.n_steps <= 0:
        raise ValueError("n_paths and n_steps must be >= 1")

    ex_style = trade.exercise.lower()

    # Exotics
    if ex_style == "bermudan":
        pricer = price_bermudan_put_ls_vector if p.method == "vector" else price_bermudan_put_ls_scalar
        kwargs = {"basis": p.basis, "degree": p.degree, "exercise_steps": p.exercise_steps}

    #elif ex_style == "digital_american":
        #if p.digital_strike is None:
            #raise ValueError("digital_strike must be provided for digital_american")
        #pricer = price_american_digital_ls_vector if p.method == "vector" else price_american_digital_ls_scalar
        #kwargs = {"basis": p.basis, "degree": p.degree, "digital_strike": p.digital_strike, "payout": p.digital_payout}

    elif ex_style == "digital_american":
        if p.digital_strike is None:
            raise ValueError("digital_strike must be provided for digital_american")

        pricer = (
            price_american_digital_first_hit_vector
            if p.method == "vector"
            else price_american_digital_first_hit_scalar
        )
        kwargs = {"digital_strike": p.digital_strike, "payout": p.digital_payout}

    # Vanilla options
    elif ex_style == "european":
        pricer = (
            price_european_naive_mc_vector
            if p.method == "vector"
            else price_european_naive_mc_scalar
        )
        kwargs = {}

    else:  # american
        if p.american_algo == "naive":
            pricer = (
                price_american_naive_mc_vector
                if p.method == "vector"
                else price_american_naive_mc_scalar
            )
            kwargs = {}
        else:
            pricer = (
                price_american_ls_vector
                if p.method == "vector"
                else price_american_ls_scalar
            )
            kwargs = {"basis": p.basis, "degree": p.degree}

    # Timing
    start_time = time.perf_counter()

    result = pricer(
        market,
        trade,
        p.n_paths,
        p.n_steps,
        seed=p.seed,
        antithetic=p.antithetic,
        **kwargs,
    )

    elapsed = time.perf_counter() - start_time

    price, discounted_payoffs = result[0], result[1]

    if p.antithetic:
        std = sample_std_anti(discounted_payoffs)
        se  = standard_error_anti(discounted_payoffs)
    else:
        std = sample_std(discounted_payoffs)
        se  = standard_error(discounted_payoffs)

    return price, std, se, elapsed

# Example usage:
if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 26)

    market = Market(S0=100, r=0.10, sigma=0.20)

    trade = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 5, 29),
        div_amount=3.0,
    )

    params = CorePricingParams(
        n_paths=100_000,
        n_steps=100,
        seed=1,
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2,
    )

    price, std, se, elapsed = core_price(market, trade, params)

    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)