import streamlit as st
import numpy as np
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from models.market import Market
from models.option_trade import Option
from analysis.greeks import compute_method_greeks
from core_pricer import run_backward_pricing, run_recursive_pricing, run_black_scholes
from utils.utils_date import datetime_to_years


st.set_page_config(page_title="EU vs US Option Pricer", page_icon="📈", layout="wide")
st.title("📊 Arbre Trinominal : Option Européenne vs Américaine ")

st.markdown("""
Cette application interactive permet de **calculer le prix d’une option financière** en utilisant un **Arbre Trinominal**.

- Backward Pricing
- Récursive Pricing           
""")

st.sidebar.title("⚙️ Paramètres du modèle")
st.markdown("""
<style>
label[for="pricing_method"] {
    font-size: 25px !important;
    font-weight: 600 !important;
    color: white !important;
}
div[data-baseweb="select"] > div {
    height: 37px !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.subheader("📊 Méthode de Pricing")
method = st.sidebar.selectbox(
    "Choisir la méthode",
    ("Trinomial – Backward", "Trinomial – Récursive"),
    key="pricing_method"
)

st.sidebar.markdown("---")

st.sidebar.subheader("💰 Marché")
S0 = st.sidebar.number_input("Spot (S0)", value=100.0)
r = st.sidebar.number_input("Taux sans risque", value=0.2, step=0.01)
sigma = st.sidebar.number_input("Volatilité", value=0.2, step=0.01)
pricing_date = st.sidebar.date_input("Date de Pricing", value=datetime.date.today() + datetime.timedelta(days=90))

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Option")
option_type = st.sidebar.radio("Type d’option", ["Call", "Put"], horizontal=True)
K = st.sidebar.number_input("Strike (K)", value=100.0)
maturity_date = st.sidebar.date_input("Maturité", value=datetime.date.today() + datetime.timedelta(days=90))

st.sidebar.markdown("---")

st.sidebar.subheader("💵 Dividende")
has_div = st.sidebar.checkbox("Inclure un dividende ?", value=False)
if has_div:
    exdiv_raw = st.sidebar.date_input("Date Ex-Div", value=datetime.date.today() + datetime.timedelta(days=90))
    rho = st.sidebar.number_input("Taux de rendement du dividende (ρ)", value=0.02, step=0.01)
    lam = st.sidebar.number_input("Facteur de décroissance (λ)", value=0.0, step=0.01)
else:
    exdiv_raw, rho, lam = None, 0.0, 0.0

st.sidebar.markdown("---")

st.sidebar.subheader("🌲 Arbre")
N = st.sidebar.number_input("Nombre de pas", value=100, step = 10)
optimize = st.sidebar.radio("Pruning ?", ["Oui", "Non"], horizontal=True)

if optimize == "Oui":
    threshold = st.sidebar.text_input(
        "Seuil d’élagage (probabilité minimale)",
        value="1e-7"
    )

    try:
        threshold = float(threshold)
        if not (1e-20 <= threshold <= 1e-2):
            st.sidebar.warning("⚠️ Le seuil doit être compris entre 1e-20 et 1e-2.")
    except ValueError:
        st.sidebar.error("❌ Veuillez entrer une valeur numérique valide (ex: 1e-7).")
        threshold = None
else:
    threshold = None

if threshold == None:
    threshold = 0.0
else:
    threshold = threshold

if maturity_date and pricing_date:
    T = datetime_to_years(maturity_date, pricing_date)
else:
    T = 1.0

if exdiv_raw is not None:
    exdivdate = datetime_to_years(exdiv_raw, pricing_date)
else:
    exdivdate = None

button = st.sidebar.button("🧮 Calculer Prix & Greeks")

tab_result, tab_prix = st.tabs([
    "📊 Résultats du Pricing et Greeks",
    "🧠 Analyse de Sensibilité"
])

market = Market(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        exdivdate=exdivdate,
        pricing_date=pricing_date,
        rho=rho,
        lam=lam
    )

is_call = (option_type == "Call")
option = Option(K=K, is_call=is_call)

with tab_result:
    if button:

        if method == "Trinomial – Backward":
            option_eu, time_eu, _ = run_backward_pricing(market, option, N, exercise = "european", optimize=optimize, threshold=threshold)
            option_us, time_us, _ = run_backward_pricing(market, option, N, exercise = "american", optimize=optimize, threshold=threshold)
            greeks_eu = compute_method_greeks(market, option, N, exercise = "european", optimize=optimize, threshold=threshold, method="backward" )
            greeks_us = compute_method_greeks(market, option, N, exercise = "american", optimize=optimize, threshold=threshold, method="backward" )
        else:
            option_eu, time_eu, _ = run_recursive_pricing(market, option, N, exercise = "european", optimize=optimize, threshold=threshold)
            option_us, time_us, _ = run_recursive_pricing(market, option, N, exercise = "american", optimize=optimize, threshold=threshold)
            greeks_eu = compute_method_greeks(market, option, N, exercise = "european", optimize=optimize, threshold=threshold, method="recursive" )
            greeks_us = compute_method_greeks(market, option, N, exercise = "american", optimize=optimize, threshold=threshold, method="recursive" )

        st.markdown("""
        <style>
        .price-card {
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.4rem;
            font-weight: 700;
            color: white;
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .eu-card {
            background-color: #1E88E5; /* Blue */
        }
        .us-card {
            background-color: #43A047; /* Green */
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
            f"<div class='price-card eu-card'>{option_type} EU : {option_eu:.2f} </div>",
            unsafe_allow_html=True
            )

        with col2:
            st.markdown(
            f"<div class='price-card us-card'>{option_type} US : {option_us:.2f} </div>",
            unsafe_allow_html=True
            )

        greeks_data = {
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Vomma"],
        "Européen": [
            round(greeks_eu.get("Delta"), 4),
            round(greeks_eu.get("Gamma"), 6),
            round(greeks_eu.get("Vega"), 4),
            round(greeks_eu.get("Theta"), 4),
            round(greeks_eu.get("Rho"), 4),
            round(greeks_eu.get("Vanna"), 4),
            round(greeks_eu.get("Vomma"), 4),
            ],
        "Américain": [
            round(greeks_us.get("Delta"), 4),
            round(greeks_us.get("Gamma"), 6),
            round(greeks_us.get("Vega"), 4),
            round(greeks_us.get("Theta"), 4),
            round(greeks_us.get("Rho"), 4),
            round(greeks_us.get("Vanna"), 4),
            round(greeks_us.get("Vomma"), 4),
            ]
        }
        st.markdown("<h3 style='text-align: center;'>📐 Greeks</h3>", unsafe_allow_html=True)
        df_greeks = pd.DataFrame(greeks_data)
        st.table(df_greeks.set_index("Greek"))

    else:
        st.info("👈 Choisissez vos paramètres puis cliquez sur un bouton pour lancer le calcul.")


with tab_prix:
    st.markdown("<h3 style='text-align:center;'>🧠 Analyse de Sensibilité – Prix</h3>", unsafe_allow_html=True)

    st.write("Cet onglet permet d’étudier comment le **prix des options Européennes et Américaines** varie lorsqu’on modifie un paramètre du modèle.")
    variable_prix = st.selectbox(
        "📊 Choisir la variable à faire varier :",
        ["Volatilité (σ)", 
         "Taux sans risque (r)", 
         "Maturité (T)", 
         "Prix d'exercice (K)"],
        index=0
    )

    st.markdown("<h3 style='text-align:center;'>⚙️ Définir la plage d'étude</h3>", unsafe_allow_html=True)
    if variable_prix == "Volatilité (σ)":
        min_val = st.number_input("Volatilité minimale (σ min)", value=0.05, step=0.01)
        max_val = st.number_input("Volatilité maximale (σ max)", value=0.6, step=0.01)
    
    elif variable_prix == "Taux sans risque (r)":
        min_val = st.number_input("Taux sans risque minimal (r min)", value=0.0, step=0.01)
        max_val = st.number_input("Taux sans risque maximal (r max)", value=0.1, step=0.01)
    
    elif variable_prix == "Maturité (T)":
        st.markdown("#### 📅 Choisissez les bornes de date de maturité")
        min_date = st.date_input("Date de maturité minimale", value=pricing_date)
        max_date = st.date_input("Date de maturité maximale", value=pricing_date + datetime.timedelta(days=365))
        if min_date < pricing_date:
            st.warning("⚠️ La date minimale doit être postérieure à la date de pricing.")
            min_date = pricing_date
        min_val = datetime_to_years(min_date, pricing_date)
        max_val = datetime_to_years(max_date, pricing_date)
    
    elif variable_prix == "Prix d'exercice (K)":
        min_val = st.number_input("Strike minimal (K min)", value=S0 * 0.5, step=1.0)
        max_val = st.number_input("Strike maximal (K max)", value=S0 * 1.5, step=1.0)

    n_points = st.number_input("Nombre de points étudiés", min_value=5, max_value=100, value=15, step=1)

    st.markdown("---")
    run_sensitivity = st.button("▶️ Lancer l’analyse de sensibilité", use_container_width=True)
    
    if run_sensitivity:
        variable_range = np.linspace(min_val, max_val, int(n_points))
        EU_prices, US_prices, Diff = [], [], []

        st.markdown(f"### 📈 Impact de {variable_prix}")

        for val in variable_range:
            if variable_prix == "Volatilité (σ)":
                market.sigma = val
            elif variable_prix == "Taux sans risque (r)":
                market.r = val
            elif variable_prix == "Maturité (T)":
                market.T = val
            elif variable_prix == "Prix d'exercice (K)":
                option.K = val

            if method == "Trinomial – Backward":
                eu, _, _ = run_backward_pricing(market, option, N, exercise="european", optimize=optimize, threshold=threshold)
                us, _, _ = run_backward_pricing(market, option, N, exercise="american", optimize=optimize, threshold=threshold)
            else:
                eu, _, _ = run_recursive_pricing(market, option, N, exercise="european", optimize=optimize, threshold=threshold)
                us, _, _ = run_recursive_pricing(market, option, N, exercise="american", optimize=optimize, threshold=threshold)

            EU_prices.append(eu)
            US_prices.append(us)
            Diff.append(us - eu)

        df_sensitivity = pd.DataFrame({
        "Variable": variable_range,
        "Prix Européen": EU_prices,
        "Prix Américain": US_prices,
        "Différence (US–EU)": Diff
        })

        st.line_chart(df_sensitivity, x="Variable", y=["Prix Européen", "Prix Américain", "Différence (US–EU)"])







