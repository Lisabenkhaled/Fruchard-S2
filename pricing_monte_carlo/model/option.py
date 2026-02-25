from dataclasses import dataclass
import datetime as dt
import numpy as np
from utils.utils_date import datetime_to_years

@dataclass(frozen=True)
class OptionTrade:
    strike: float
    is_call: bool
    exercise: str  # "european" ou "american"
    pricing_date: dt.date
    maturity_date: dt.date

    # dividends
    q: float = 0.0                       # continuous dividend yield
    ex_div_date: dt.date | None = None   # single discrete dividend ex-date
    div_amount: float = 0.0              # amount paid at ex-date

    @property
    def T(self) -> float:
        return float(datetime_to_years(self.maturity_date, self.pricing_date))

    def ex_div_time(self) -> float | None:
        if self.ex_div_date is None:
            return None
        return float(datetime_to_years(self.ex_div_date, self.pricing_date))

    # payoff scalar
    def payoff_scalar(self, S: float) -> float:
        K = self.strike
        if self.is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # payoff vector
    def payoff_vector(self, S: np.ndarray) -> np.ndarray:
        K = self.strike
        if self.is_call:
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)
