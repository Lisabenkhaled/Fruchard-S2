from dataclasses import dataclass

@dataclass
class Option:
    """
    Classe de données représentant une option vanille (call ou put)
    """
    K: float             # Strike de l'option
    is_call: bool = True # Type de l'option: Call = True, Put = False

    def payoff(self, S: float) -> float:
        """
        Calcule le payoff de l’option à l’échéance en fonction du prix du sous-jacent.
        """
        if self.is_call:
            return max(S - self.K, 0)
        else:
            return max(self.K - S, 0)