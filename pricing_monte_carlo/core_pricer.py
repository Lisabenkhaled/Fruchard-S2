import time
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

from model.market import Market
from model.option import OptionTrade

from utils.utils_stats import (
    sample_std, standard_error,
    sample_std_anti, standard_error_anti,
)

from model.mc_pricer import (
    # European vanilla
    price_european_naive_mc_vector,
    price_european_naive_mc_scalar,
    # American vanilla - naive
    price_american_naive_mc_vector,
    price_american_naive_mc_scalar,
    # American vanilla - Longstaff–Schwartz
    price_american_ls_vector,
    price_american_ls_scalar,
)

# Digital American
from model.mc_pricer_digital import (
    price_american_digital_vector,
    price_american_digital_scalar,
)

Method = Literal["vector", "scalar"]
AmericanAlgo = Literal["naive", "ls"]
Basis = Literal["power", "laguerre"]
PayoffType = Literal["vanilla", "digital"]


@dataclass(frozen=True)
class CorePricingParams:
    n_paths: int
    n_steps: int
    seed: int = 0
    antithetic: bool = False
    method: Method = "vector"

    # Vanilla American only
    american_algo: AmericanAlgo = "ls"
    basis: Basis = "laguerre"
    degree: int = 2

    # Payoff selection
    payoff: PayoffType = "vanilla"

    # Digital params (only used if payoff="digital")
    digital_strike: Optional[float] = None
    payout: float = 1.0


def _pick_pricer(trade: OptionTrade, p: CorePricingParams):
    ex_style = trade.exercise.lower()

    # -------------------------
    # DIGITAL
    # -------------------------
    if p.payoff == "digital":
        if ex_style != "american":
            raise ValueError("Digital option implemented only for exercise='american' (first-hit).")
        if p.digital_strike is None:
            raise ValueError("digital_strike must be provided when payoff='digital'.")

        pricer = (
            price_american_digital_vector
            if p.method == "vector"
            else price_american_digital_scalar
        )
        kwargs = {"digital_strike": float(p.digital_strike), "payout": float(p.payout)}
        return pricer, kwargs

    # -------------------------
    # VANILLA
    # -------------------------
    if ex_style == "european":
        pricer = (
            price_european_naive_mc_vector
            if p.method == "vector"
            else price_european_naive_mc_scalar
        )
        kwargs = {}
        return pricer, kwargs

    if ex_style == "american":
        if p.american_algo == "naive":
            pricer = (
                price_american_naive_mc_vector
                if p.method == "vector"
                else price_american_naive_mc_scalar
            )
            kwargs = {}
            return pricer, kwargs

        pricer = (
            price_american_ls_vector
            if p.method == "vector"
            else price_american_ls_scalar
        )
        kwargs = {"basis": p.basis, "degree": int(p.degree)}
        return pricer, kwargs

    raise ValueError(f"Unknown exercise style: {trade.exercise!r}")


def core_price(
    market: Market,
    trade: OptionTrade,
    p: CorePricingParams
) -> Tuple[float, float, float, float]:
    """
    Returns:
      price, std, std_error, elapsed_seconds
    """
    if p.n_paths <= 0 or p.n_steps <= 0:
        raise ValueError("n_paths and n_steps must be >= 1")

    pricer, kwargs = _pick_pricer(trade, p)

    start_time = time.perf_counter()

    price, discounted_payoffs = pricer(
        market,
        trade,
        p.n_paths,
        p.n_steps,
        seed=p.seed,
        antithetic=p.antithetic,
        **kwargs,
    )

    elapsed = time.perf_counter() - start_time

    if p.antithetic:
        std = sample_std_anti(discounted_payoffs)
        se = standard_error_anti(discounted_payoffs)
    else:
        std = sample_std(discounted_payoffs)
        se = standard_error(discounted_payoffs)

    return float(price), float(std), float(se), float(elapsed)


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 2, 26)
    maturity_date = dt.date(2027, 4, 26)

    market = Market(S0=100, r=0.04, sigma=0.25)

    # -------------------------
    # Example 1: Vanilla American (LS)
    # -------------------------
    trade_vanilla = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 5, 29),
        div_amount=3.0,
    )

    params_vanilla = CorePricingParams(
        n_paths=100_000,
        n_steps=100,
        seed=1,
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2,
        payoff="vanilla",
    )

    price, std, se, elapsed = core_price(market, trade_vanilla, params_vanilla)
    print("\n[VANILLA AMERICAN LS]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)

    # -------------------------
    # Example 2: Digital American (first-hit)
    # -------------------------
    trade_digital = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 6, 21),
        div_amount=3.0,
    )

    params_digital = CorePricingParams(
        n_paths=100_000,
        n_steps=100,
        seed=42,
        antithetic=True,
        method="vector",
        payoff="digital",
        digital_strike=90.0,
        payout=1.0,
    )

    price, std, se, elapsed = core_price(market, trade_digital, params_digital)
    print("\n[DIGITAL AMERICAN FIRST-HIT]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)

    # -------------------------
    # Example 3: Bermudan
    # -------------------------
    trade_vanilla = OptionTrade(
        strike=100.0,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 6, 21),
        div_amount=3.0,
    )

    params_vanilla = CorePricingParams(
        n_paths=100_000,
        n_steps=14,
        seed=42,
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="power",
        degree=2,
        payoff="vanilla",
    )

    price, std, se, elapsed = core_price(market, trade_vanilla, params_vanilla)
    print("\n[BERMUDAN]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)