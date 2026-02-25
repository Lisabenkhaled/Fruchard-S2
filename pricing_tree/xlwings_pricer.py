# xlwings_pricer.py — pont Excel <-> Option-Pricing-main

from __future__ import annotations

import os, sys
# -- bootstrap chemins relatifs: dossier courant + sous-dossiers du projet --
_BASE = os.path.abspath(os.path.dirname(__file__) or os.getcwd())
sys.path[:0] = [
    _BASE,
    os.path.join(_BASE, "analysis"),
    os.path.join(_BASE, "models"),
    os.path.join(_BASE, "utils"),
]
# ---------------------------------------------------------------------------

import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Literal

import xlwings as xw

# --- imports métier explicites (conformes à l'arbo) ---
from models.tree import TrinomialTree
from models.market import Market
from models.option_trade import Option
from utils.utils_bs import bs_price
from utils.utils_tree_error import tree_error





# =============================== Utils ===============================

def _like_prefix(s: str, prefix: str) -> bool:
    return str(s).strip().lower().startswith(prefix.lower().rstrip("*"))

def _as_bool(x) -> bool:
    if isinstance(x, str):
        x = x.strip().lower()
        if x in {"oui", "yes", "true"}: return True
        if x in {"non", "no", "false"}: return False
    return bool(x)

def _norm_cp(x: str) -> Literal["Call", "Put"]:
    s = str(x).strip().lower()
    return "Call" if s.startswith("c") else "Put"

def _norm_ex(x: str) -> Literal["EU", "US"]:
    s = str(x).strip().lower()
    return "EU" if s.startswith("e") else "US"

def _norm_method(x: str) -> Literal["rec", "back"]:
    s = str(x).strip().lower()
    if _like_prefix(s, "r*"): return "rec"
    if _like_prefix(s, "b*"): return "back"
    return "back"

def _norm_lang(x: str) -> Literal["py", "vb"]:
    s = str(x).strip().lower()
    if _like_prefix(s, "py*"): return "py"
    if _like_prefix(s, "vb*"): return "vb"
    return "py"

def _to_datetime(val) -> dt.datetime:
    """Convertit proprement ce qui vient d'Excel en datetime."""
    if isinstance(val, dt.datetime):
        return val
    if isinstance(val, dt.date):
        return dt.datetime(val.year, val.month, val.day)
    # Numéro de série Excel (Windows) -> base 1899-12-30
    if isinstance(val, (int, float)):
        base = dt.datetime(1899, 12, 30)
        return base + dt.timedelta(days=float(val))
    if isinstance(val, str):
        txt = val.strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                d = dt.datetime.strptime(txt, fmt)
                return d
            except ValueError:
                pass
        # dernier recours: essayer fromisoformat (peut lever ValueError)
        try:
            return dt.datetime.fromisoformat(txt)
        except Exception:
            pass
    # défaut: maintenant, pour éviter None
    return dt.datetime.now()


# =============================== Structures & mapping Excel ===============================

@dataclass
class Inputs:
    # Marché
    spot: float
    vol: float
    rate: float
    div: float
    div_ex_date: dt.datetime

    # Option
    strike: float
    maturity: dt.datetime
    cp: Literal["Call", "Put"]
    exercise: Literal["EU", "US"]

    # Arbre
    pricing_date: dt.datetime
    n_steps: int
    dt_step: float
    alpha: float

    # Affichages/optim (non utilisés par TrinomialTree mais on les lit pour compat)
    show_tree: bool
    pruning: bool
    prune_thresh: float

    # Sélection
    method: Literal["rec", "back"]
    lang: Literal["py", "vb"]


NR = {
    # === Marché ===
    "Spot": "Spot",
    "Vol": "Vol",
    "Rate": "Taux",
    "Div": "Dividende",
    "DivDate": "ExDivDate_Dividende",
    "Rho": "Rho",
    "Lambda": "Lambda",

    # === Option ===
    "Strike": "Strike",
    "Mat": "Maturity",
    "CP": "Call_Put",
    "Ex": "Exercice",

    # === Arbre ===
    "PricingDate": "date_pricing",
    "N": "N",
    "DeltaT": "Delta_t",
    "Alpha": "Alpha",
    "Show": "AffichageStock",       # affichage du sous-jacent
    "ShowOption": "AffichageOption",
    "ShowProba": "AffichageProba",
    "Pruning": "Pruning",
    "Thresh": "SeuilPruning",

    # === Méthode & Langage ===
    "Method": "Methode_Pricing",
    "Lang": "Langage",

    # === Sorties ===
    "PriceTree": "Prix_Tree",
    "TimeTree": "Time_Tree",
    "PriceBS": "Prix_BS",
    "TimeBS": "Time_BS",

    # === Divers / Tests ===
    "TreeError": "tree_error",
    "ParamTestSpeed": "Param_TestVitesse",
}


# =============================== IO Excel ===============================

def read_inputs(sht: xw.Sheet) -> Inputs:
    n = sht.book.names
    v = lambda key: n[NR[key]].refers_to_range.value

    return Inputs(
        # Marché
        spot=float(v("Spot")),
        vol=float(v("Vol")),
        rate=float(v("Rate")),
        div=float(v("Div")),
        div_ex_date=_to_datetime(v("DivDate")),

        # Option
        strike=float(v("Strike")),
        maturity=_to_datetime(v("Mat")),
        cp=_norm_cp(v("CP")),
        exercise=_norm_ex(v("Ex")),

        # Arbre
        pricing_date=_to_datetime(v("PricingDate")),
        n_steps=int(v("N")),
        dt_step=float(v("DeltaT")),
        alpha=float(v("Alpha")),

        # Flags
        show_tree=_as_bool(v("Show")),
        pruning=_as_bool(v("Pruning")),
        prune_thresh=float(v("Thresh")),

        # Sélection
        method=_norm_method(v("Method")),
        lang=_norm_lang(v("Lang")),
    )


def write_outputs(sht: xw.Sheet, price_tree: float, time_tree: float,
                  price_bs: Optional[float], time_bs: Optional[float],
                  inp: Optional[Inputs] = None) -> None:
    n = sht.book.names
    n[NR["PriceTree"]].refers_to_range.value = float(price_tree)
    n[NR["TimeTree"]].refers_to_range.value = f"{time_tree:.4f} s"

    if price_bs is not None:
        n[NR["PriceBS"]].refers_to_range.value = float(price_bs)
        if time_bs is not None:
            n[NR["TimeBS"]].refers_to_range.value = f"{time_bs:.4f} s"

    # ==== Ajout : Tree Error théorique ====
    try:
        if inp is not None:
            err_val = tree_error(
                S0=inp.spot,
                sigma=inp.vol,
                r=inp.rate,
                T=max((inp.maturity - inp.pricing_date).days, 0) / 365.0,
                N=max(int(inp.n_steps), 1),
            )
            n["tree_error"].refers_to_range.value = err_val
        else:
            n["tree_error"].refers_to_range.value = ""
    except Exception:
        n["tree_error"].refers_to_range.value = ""

# =============================== Constructions métier ===============================

def build_market(inp: Inputs) -> Market:
    """Market(S0, r, sigma, T, ...)"""
    # Sécurise T > 0
    days = (inp.maturity - inp.pricing_date).days
    T_years = max(days, 0) / 365.0

    # Essaie d'appeler Market sans l'argument 'dividends'
    try:
        return Market(S0=inp.spot, r=inp.rate, sigma=inp.vol, T=T_years, dividends=None)
    except TypeError:
        # fallback si 'dividends' n'existe pas dans la signature
        return Market(S0=inp.spot, r=inp.rate, sigma=inp.vol, T=T_years)

def build_option(inp: Inputs) -> Option:
    """Option(K, is_call=True|False)"""
    return Option(K=inp.strike, is_call=(inp.cp == "Call"))

def build_tree(inp: Inputs, mkt: Market, opt: Option) -> TrinomialTree:
    """TrinomialTree(market, option, N, exercise="european"/"american")"""
    exercise = "european" if inp.exercise == "EU" else "american"

    # Sécurise N >= 1
    N = max(int(inp.n_steps), 1)

    tree = TrinomialTree(market=mkt, option=opt, N=N, exercise=exercise)

    # Certaines versions exigent une étape de construction explicite
    for method in ("build_tree", "build", "construct_tree", "construct", "init_tree", "initialize"):
        if hasattr(tree, method):
            try:
                getattr(tree, method)()              # sans args
            except TypeError:
                try:
                    getattr(tree, method)(mkt, opt)  # avec args
                except TypeError:
                    pass
            break

    return tree


# =============================== Pricers ===============================

from models.backward_pricing import price_backward
from models.recursive_pricing import price_recursive
from utils.utils_bs import bs_price


def run_tree_pricer(inp: Inputs) -> float:
    """Calcule le prix d'une option via l'arbre trinomial (backward ou recursive)."""
    mkt = build_market(inp)
    opt = build_option(inp)
    tree = build_tree(inp, mkt, opt)

    # Choix de la méthode selon la sélection Excel
    if inp.method == "rec":
        return price_recursive(tree)
    else:
        return price_backward(tree)


def run_bs_pricer(inp: Inputs) -> Optional[float]:
    """Calcule le prix Black-Scholes si l'option est européenne."""
    if inp.exercise != "EU":
        return None
    T_years = max((inp.maturity - inp.pricing_date).days, 0) / 365.0
    return bs_price(
        S=inp.spot,
        K=inp.strike,
        r=inp.rate,
        sigma=inp.vol,
        T=T_years,
        is_call=(inp.cp == "Call"),
    )

# =============================== Macro principale (bouton) ===============================

@xw.sub
def RunPricer():
    wb = xw.Book.caller()
    sht = wb.sheets["Param"]

    inp = read_inputs(sht)

    # Si Langage = vb*, on rend la main à VBA
    if inp.lang == "vb":
        return

    t0 = time.perf_counter()
    price_tree = run_tree_pricer(inp)
    t1 = time.perf_counter()

    price_bs = None
    t_bs = None
    try:
        t2 = time.perf_counter()
        price_bs = run_bs_pricer(inp)
        t3 = time.perf_counter()
        if price_bs is not None:
            t_bs = t3 - t2
    except Exception:
        price_bs, t_bs = None, None

    write_outputs(
    sht,
    price_tree=price_tree,
    time_tree=(t1 - t0),
    price_bs=price_bs,
    time_bs=t_bs,
    inp=inp,        # <--- ajouté ici
)



# =============================== UDF (facultatif) ===============================

@xw.func
def xlw_price_tree():
    wb = xw.Book.caller()
    sht = wb.sheets["Param"]
    inp = read_inputs(sht)
    if inp.lang == "vb":
        return ""
    return float(run_tree_pricer(inp))

@xw.func
def xlw_price_bs():
    wb = xw.Book.caller()
    sht = wb.sheets["Param"]
    inp = read_inputs(sht)
    if inp.lang == "vb":
        return ""
    p = run_bs_pricer(inp)
    return "" if p is None else float(p)


if __name__ == "__main__":
    print("Appeler ce module depuis Excel via xlwings (macro RunPricer).")
