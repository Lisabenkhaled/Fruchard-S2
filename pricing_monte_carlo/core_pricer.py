import time
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any

from model.market import Market
from model.option import OptionTrade

# Statistics utilities
from utils.utils_stats import (
    sample_std,
    standard_error,
    sample_std_anti,
    standard_error_anti,
)

# Vanilla option pricers
from model.mc_pricer import (
    price_european_naive_mc_vector,
    price_european_naive_mc_scalar,
    price_american_naive_mc_vector,
    price_american_naive_mc_scalar,
    price_american_ls_vector,
    price_american_ls_scalar,
)

# Digital American pricers
from model.mc_pricer_digital import (
    price_american_digital_vector,
    price_american_digital_scalar,
)

# Type definitions
Method = Literal["vector", "scalar"]
AmericanAlgo = Literal["naive", "ls"]
Basis = Literal["power", "laguerre"]
PayoffType = Literal["vanilla", "digital"]

# Parameter container for the core pricer
@dataclass(frozen=True)
class CorePricingParams:
    n_paths: int
    n_steps: int
    seed: int = 0
    antithetic: bool = False
    method: Method = "vector"

    # Vanilla American configuration
    american_algo: AmericanAlgo = "ls"
    basis: Basis = "laguerre"
    degree: int = 2

    # Payoff type
    payoff: PayoffType = "vanilla"

    # Digital parameters (used only if payoff="digital")
    digital_strike: Optional[float] = None
    payout: float = 1.0

# Select the appropriate pricing function
def _pick_pricer(trade: OptionTrade, p: CorePricingParams) -> Tuple[Any, Dict[str, Any]]:
    ex_style = trade.exercise.lower()

    # DIGITAL OPTION CASE
    if p.payoff == "digital":
        if ex_style != "american":
            raise ValueError("Digital option implemented only for american exercise.")
        if p.digital_strike is None:
            raise ValueError("digital_strike must be provided.")

        pricer = (
            price_american_digital_vector
            if p.method == "vector"
            else price_american_digital_scalar
        )

        kwargs = {
            "digital_strike": float(p.digital_strike),
            "payout": float(p.payout),
        }

        return pricer, kwargs
    
    # EUROPEAN VANILLA
    if ex_style == "european":
        pricer = (
            price_european_naive_mc_vector
            if p.method == "vector"
            else price_european_naive_mc_scalar
        )
        return pricer, {}

    # AMERICAN VANILLA
    if ex_style == "american":
        if p.american_algo == "naive":
            pricer = (
                price_american_naive_mc_vector
                if p.method == "vector"
                else price_american_naive_mc_scalar
            )
            return pricer, {}

        # Longstaff-Schwartz algorithm
        pricer = (
            price_american_ls_vector
            if p.method == "vector"
            else price_american_ls_scalar
        )

        kwargs = {
            "basis": p.basis,
            "degree": int(p.degree),
        }

        return pricer, kwargs

    raise ValueError(f"Unknown exercise style: {trade.exercise}")

# Core pricing engine
def core_price(market: Market, trade: OptionTrade, p: CorePricingParams) -> Tuple[float, float, float, float]:
    # Select the appropriate pricer
    pricer, kwargs = _pick_pricer(trade, p)
    start_time = time.perf_counter()

    # Run the Monte Carlo pricer
    price, discounted_payoffs = pricer(
        market,
        trade,
        p.n_paths,
        p.n_steps,
        seed=p.seed,
        antithetic=p.antithetic,
        **kwargs,
    )
    # Stop timer
    elapsed = time.perf_counter() - start_time

    # Compute statistics
    if p.antithetic:
        std = sample_std_anti(discounted_payoffs)
        se = standard_error_anti(discounted_payoffs)
    else:
        std = sample_std(discounted_payoffs)
        se = standard_error(discounted_payoffs)

    # Return results
    return float(price), float(std), float(se), float(elapsed)

# Exemple d'usage
if __name__ == "__main__":
    import datetime as dt

    pricing_date = dt.date(2026, 3, 1)
    maturity_date = dt.date(2026, 12, 25)

    market = Market(S0=100, r=0.10, sigma=0.20)

    # Example 1: Vanilla American (LS)
    # Option
    trade_vanilla = OptionTrade(
        strike=100.0,
        is_call=True,
        exercise="european",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=dt.date(2026, 11, 30),
        div_amount=0.0,
    )

    # Parameters
    params_vanilla = CorePricingParams(
        n_paths=100,
        n_steps=10,
        seed=1,
        antithetic=True,
        method="vector",
        american_algo="ls",
        basis="laguerre",
        degree=2,
        payoff="vanilla",
    )
    
    # Calculation
    price, std, se, elapsed = core_price(market, trade_vanilla, params_vanilla)
    print("\n[VANILLA AMERICAN LS]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)

    # Example 2: Digital American 
    # Option
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

    # Parameters
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
    
    # Calculation
    price, std, se, elapsed = core_price(market, trade_digital, params_digital)
    print("\n[DIGITAL AMERICAN FIRST-HIT]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)

    # Example 3: Bermudan
    # Option
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

    # Parameters
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

    # Calculation
    price, std, se, elapsed = core_price(market, trade_vanilla, params_vanilla)
    print("\n[BERMUDAN]")
    print("Price:", price)
    print("Std:", std)
    print("Std Error:", se)
    print("Time:", elapsed)