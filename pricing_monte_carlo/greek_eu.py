import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams
from greeks import compute_greeks_vector

# ✅ Tree Greeks adapter (Option 1)
from pricing_tree.greek_adaptateur import tree_greeks_from_mc


pricing_date = dt.date(2026, 2, 26)
maturity_date = dt.date(2027, 3, 1)

market = Market(S0=100, r=0.03, sigma=0.2)

trade_eu = OptionTrade(
    strike=100,
    is_call=True,
    exercise="european",
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    ex_div_date=None,
    div_amount=0.0
)

params_eu = CorePricingParams(
    n_paths=100_000,
    n_steps=300,
    seed=42,
    antithetic=True,
    method="vector",
    # if your CorePricingParams includes american_algo/basis/degree, you can leave defaults
)

greeks_eu = compute_greeks_vector(market, trade_eu, params_eu)

print("EU (MC VECTOR)")
for k, v in greeks_eu.items():
    try:
        print(f"{k}: {float(v):.6f}")
    except Exception:
        print(f"{k}: {v}")

tree_greeks = tree_greeks_from_mc(
    mc_market=market,
    mc_trade=trade_eu,
    N=300,        # tree steps
)

print("\nEU (TREE BACKWARD)")
for k, v in tree_greeks.items():
    try:
        print(f"{k}: {float(v):.6f}")
    except Exception:
        print(f"{k}: {v}")
