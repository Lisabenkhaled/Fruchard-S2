from numpy import sqrt, exp, pi

def tree_error(S0, sigma, r, T, N):
    return ((3 * S0) / (8 * sqrt(2 * pi))) * (((exp(sigma**2 * (T/N)) - 1)*exp(2*r*(T/N))) / sqrt(exp(sigma**2*T)-1))