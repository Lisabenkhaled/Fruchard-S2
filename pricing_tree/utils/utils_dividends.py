def get_dividend_on_step(market, t_i: float, t_ip1: float, S: float):
    """
    Vérifie s’il existe un dividende sur l’intervalle [t_i, t_{i+1})
    et retourne son montant ainsi qu’un indicateur booléen.
    """
    div = 0.0
    has_dividend = False

    if market.has_dividend():
        t_div, policy = market.dividends[0]  # Unique ex-div dans ce projet
        if t_i < t_div < t_ip1:
            div = policy.amount(t_div, S, market.S0)
            has_dividend = True

    return div, has_dividend
