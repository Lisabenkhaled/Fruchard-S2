from dataclasses import dataclass
from functools import cached_property
import datetime as dt
import numpy as np
from utils.utils_date import datetime_to_years


@dataclass(frozen=True)
class OptionTrade:
    strike: float
    is_call: bool
    exercise: str  # "european" or "american"
    pricing_date: dt.date
    maturity_date: dt.date

    # dividends
    q: float = 0.0
    ex_div_date: dt.date | None = None
    div_amount: float = 0.0

    @cached_property
    def T(self) -> float:
        return float(datetime_to_years(self.maturity_date, self.pricing_date))

    @cached_property
    def ex_div_t(self) -> float | None:
        if self.ex_div_date is None:
            return None
        return float(datetime_to_years(self.ex_div_date, self.pricing_date))

    def ex_div_time(self) -> float | None:
        return self.ex_div_t

    # Scalar Payoff
    def payoff_scalar(self, S: float) -> float:
        K = self.strike
        if self.is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # Vectorized Payoff
    def payoff_vector(self, S: np.ndarray) -> np.ndarray:
        K = self.strike
        if self.is_call:
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)