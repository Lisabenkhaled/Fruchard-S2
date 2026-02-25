from dataclasses import dataclass

@dataclass(frozen=True)
class Market:
    S0: float
    r: float
    sigma: float
