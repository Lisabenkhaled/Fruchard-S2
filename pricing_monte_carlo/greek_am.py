import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams
from greeks import compute_greeks_vector

from pricing_tree.greek_adaptateur import tree_greeks_from_mc


pricing_date = dt.date(2026, 2, 26)
maturity_date = dt.date(2027, 4, 26)

market = Market(
    S0=100,
    r=0.04,
    sigma=0.25
)

trade_am = OptionTrade(
    strike=100,
    is_call=False,
    exercise="american",
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    q=0.0,
    ex_div_date=dt.date(2026, 6, 21),
    div_amount=3.0
)

params_am_ls = CorePricingParams(
    n_paths=10_000,
    n_steps=365,
    seed=1,
    antithetic=True,
    method="vector",
    american_algo="ls",
    basis="laguerre",
    degree=2
)

greeks_am_mc = compute_greeks_vector(market, trade_am, params_am_ls)

print("AM LS (MC VECTOR)")
for k, v in greeks_am_mc.items():
    try:
        print(f"{k}: {float(v):.6f}")
    except Exception:
        print(f"{k}: {v}")

tree_greeks_am = tree_greeks_from_mc(
    mc_market=market,
    mc_trade=trade_am,
    N=1000
)

print("\nAM (TREE)")
for k, v in tree_greeks_am.items():
    try:
        print(f"{k}: {float(v):.6f}")
    except Exception:
        print(f"{k}: {v}")
