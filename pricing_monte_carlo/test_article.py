import numpy as np
import datetime as dt

from model.option import OptionTrade
from model.regression import design_matrix


def test_ls_article_example_power_basis():
    """
    Reproduce the Longstaff-Schwartz numerical example 
    """

    K = 1.10
    df = 0.94176

    # dummy dates 
    pricing_date = dt.date(2026, 1, 1)
    maturity_date = dt.date(2029, 1, 1)

    trade = OptionTrade(
        strike=K,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        q=0.0,
        ex_div_date=None,
        div_amount=0.0
    )

    # Paths from article 
    paths = np.array([
        [1.00, 1.09, 1.08, 1.34],
        [1.00, 1.16, 1.26, 1.54],
        [1.00, 1.22, 1.07, 1.03],
        [1.00, 0.93, 0.97, 0.92],
        [1.00, 1.11, 1.56, 1.52],
        [1.00, 0.76, 0.77, 0.90],
        [1.00, 0.92, 0.84, 1.01],
        [1.00, 0.88, 1.22, 1.34],
    ], dtype=float)

    V3 = trade.payoff_vector(paths[:, 3])
    expected_V3 = np.array([0.00, 0.00, 0.07, 0.18, 0.00, 0.20, 0.09, 0.00])
    assert np.allclose(V3, expected_V3, atol=1e-12), f"V3 mismatch: {V3}"

    S2 = paths[:, 2]
    ex2 = trade.payoff_vector(S2)
    itm2 = ex2 > 0.0

    Y2 = df * V3[itm2]

    X2 = design_matrix(S2[itm2], degree=2, basis="power")
    beta2, *_ = np.linalg.lstsq(X2, Y2, rcond=None)
    cont2 = X2 @ beta2

    expected_beta2 = np.array([-1.06998, 2.98340, -1.81358])
    assert np.allclose(beta2, expected_beta2, atol=5e-4), f"beta2 mismatch: {beta2}"

    ex_now2_itm = ex2[itm2] > cont2
    exercised2 = np.where(itm2)[0][ex_now2_itm]

    assert np.array_equal(exercised2, np.array([3, 5, 6])), f"exercised2 mismatch: {exercised2}"

    V2 = df * V3
    V2[exercised2] = ex2[exercised2]

    S1 = paths[:, 1]
    ex1 = trade.payoff_vector(S1)
    itm1 = ex1 > 0.0

    Y1 = df * V2[itm1]

    X1 = design_matrix(S1[itm1], degree=2, basis="power")
    beta1, *_ = np.linalg.lstsq(X1, Y1, rcond=None)
    cont1 = X1 @ beta1

    expected_beta1 = np.array([2.0375, -3.3354, 1.3565])
    assert np.allclose(beta1, expected_beta1, atol=5e-4), f"beta1 mismatch: {beta1}"

    ex_now1_itm = ex1[itm1] > cont1
    exercised1 = np.where(itm1)[0][ex_now1_itm]

    assert np.array_equal(exercised1, np.array([3, 5, 6, 7])), f"exercised1 mismatch: {exercised1}"

    V1 = df * V2
    V1[exercised1] = ex1[exercised1]

    price0 = float(df * V1.mean())
    assert abs(price0 - 0.1144) < 5e-4, f"price mismatch: {price0}"

    print("Article example reproduced (POWER basis, degree=2)")
    print(f"beta(t=2) [c,b,a] = {beta2}")
    print(f"beta(t=1) [c,b,a] = {beta1}")
    print(f"Price t=0         = {price0:.6f}")


def run_same_example_laguerre_for_comparison():
    """
    Same paths, but using Laguerre basis (degree=2) with scale=K, to show how the choice of basis affects the results.
    """
    K = 1.10
    df = 0.94176

    pricing_date = dt.date(2026, 1, 1)
    maturity_date = dt.date(2029, 1, 1)

    trade = OptionTrade(
        strike=K,
        is_call=False,
        exercise="american",
        pricing_date=pricing_date,
        maturity_date=maturity_date,
    )

    paths = np.array([
        [1.00, 1.09, 1.08, 1.34],
        [1.00, 1.16, 1.26, 1.54],
        [1.00, 1.22, 1.07, 1.03],
        [1.00, 0.93, 0.97, 0.92],
        [1.00, 1.11, 1.56, 1.52],
        [1.00, 0.76, 0.77, 0.90],
        [1.00, 0.92, 0.84, 1.01],
        [1.00, 0.88, 1.22, 1.34],
    ], dtype=float)

    V3 = trade.payoff_vector(paths[:, 3])

    S2 = paths[:, 2]
    ex2 = trade.payoff_vector(S2)
    itm2 = ex2 > 0.0
    Y2 = df * V3[itm2]
    X2 = design_matrix(S2[itm2], degree=2, basis="laguerre", scale=trade.strike)
    beta2, *_ = np.linalg.lstsq(X2, Y2, rcond=None)
    cont2 = X2 @ beta2
    exercised2 = np.where(itm2)[0][(ex2[itm2] > cont2)]

    V2 = df * V3
    V2[exercised2] = ex2[exercised2]

    S1 = paths[:, 1]
    ex1 = trade.payoff_vector(S1)
    itm1 = ex1 > 0.0
    Y1 = df * V2[itm1]
    X1 = design_matrix(S1[itm1], degree=2, basis="laguerre", scale=trade.strike)
    beta1, *_ = np.linalg.lstsq(X1, Y1, rcond=None)
    cont1 = X1 @ beta1
    exercised1 = np.where(itm1)[0][(ex1[itm1] > cont1)]

    V1 = df * V2
    V1[exercised1] = ex1[exercised1]

    price0 = float(df * V1.mean())

    print("\n--- Laguerre comparison (degree=2, scale=K) ---")
    print(f"beta(t=2) = {beta2}")
    print(f"beta(t=1) = {beta1}")
    print(f"Price t=0 = {price0:.6f}")


if __name__ == "__main__":
    test_ls_article_example_power_basis()
    run_same_example_laguerre_for_comparison()
