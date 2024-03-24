from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from utils.type import OptionType

class OptionBase(ABC):
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        maturity: float,
        rate: float,
        volatility: float,
        option_type: OptionType,
        dividend: Optional[float] = None,
    ) -> None:
        self._spot_price = spot_price
        self._strike_price = strike_price
        self._maturity = maturity
        self._rate = rate
        self._volatility = volatility
        self._option_type = option_type
        self._dividend = dividend if dividend is not None else 0.0
        self._d1 = self.__d1_func()
        self._d2 = self.__d2_func()

    def __d1_func(self) -> float:
        """Compute d1 of the Black-Scholes formula.

        Returns:
            float: The value of d1.
        """
        return (
            np.log(self._spot_price / self._strike_price)
            + (
                (self._rate - self._dividend)
                + 0.5 * self._volatility ** 2
            )
            * self._maturity
        ) / (
            self._volatility
            * np.sqrt(self._maturity)
        )

    def __d2_func(self) -> float:
        """Compute d2 of the Black-Scholes formula.

        Returns:
            float: The value of d2.
        """

        return (
            np.log(self._spot_price / self._strike_price)
            + (
                (self._rate- self._dividend)
                + 0.5 * self._volatility ** 2
            )
            * self._maturity
        ) / (
            self._volatility
            * np.sqrt(self._maturity)
        ) - self._volatility * np.sqrt(
            self._maturity
        )

    @property
    def d1(self) -> float:
        return self._d1

    @property
    def d2(self) -> float:
        return self._d2

    @abstractmethod
    def compute_price(self):
        pass

    @abstractmethod
    def compute_greeks(self):
        pass