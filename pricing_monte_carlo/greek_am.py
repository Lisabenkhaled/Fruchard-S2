import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import datetime as dt

from model.market import Market
from model.option import OptionTrade
from core_pricer import CorePricingParams
from greeks import compute_greeks_vector

# ✅ Tree Greeks adapter
from pricing_tree.greek_adaptateur import tree_greeks_from_mc


pricing_date = dt.date(2026, 3, 1)
maturity_date = dt.date(2026, 12, 25)

market = Market(
    S0=100,
    r=0.05,
    sigma=0.20
)

trade_am = OptionTrade(
    strike=100,
    is_call=True,
    exercise="american",
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    q=0.0,
    ex_div_date=dt.date(2026, 11, 30),
    div_amount=3.0
)

params_am_ls = CorePricingParams(
    n_paths=10_000,   # ↑ increase for more stable Greeks
    n_steps=250,
    seed=2,
    antithetic=True,
    method="vector",
    american_algo="ls",
    basis="laguerre",
    degree=2
)

# -----------------------
# MC Greeks (LS)
# -----------------------
greeks_am_mc = compute_greeks_vector(market, trade_am, params_am_ls)

print("AM LS (MC VECTOR)")
for k, v in greeks_am_mc.items():
    try:
        print(f"{k}: {float(v):.6f}")
    except Exception:
        print(f"{k}: {v}")

# -----------------------
# Tree Greeks
# -----------------------
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
