import numpy as np
from datetime import datetime
from typing import Dict, Any
from typing import Optional
from src.abstract import _Data, _Strategies
from utils.check import check_configs
from abc import ABC
from utils import portfolio_tools as pf_t


class Position(ABC):
    def __init__(self, capital: float, weights: np.ndarray, date: datetime, horizon: datetime):
        """
        Initializes a new instance of the Position class.

        Args:
            capital (float): Initial capital amount, must be greater than zero.
            weights (np.ndarray): Asset allocation weights, must sum to 1.
            date (datetime): Effective date of this position.
            horizon (int, optional): Investment horizon in days.

        Raises:
            ValueError: If the initial conditions for capital or weights are not met.
        """
        if capital <= 0:
            raise ValueError("Capital must be greater than zero.")
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must sum to 1.")

        self._capital = capital
        self._weights = weights
        self._date = date
        self._next_weights: np.ndarray = None
        self._horizon = horizon

    @property
    def capital(self) -> float:
        """Returns the capital amount of the position."""
        return self._capital

    @property
    def weights(self) -> np.ndarray:
        """Returns the current asset allocation weights."""
        return self._weights

    @property
    def date(self) -> datetime:
        """Returns the effective date of this position."""
        return self._date

    @property
    def next_weights(self) -> Optional[np.ndarray]:
        """Returns the next planned weights for asset allocation."""
        return self._next_weights

    @property
    def horizon(self) -> datetime:
        """Returns the investment horizon in days."""
        return self._horizon

    @property
    def returns(self) -> np.ndarray:
        """Returns the investment horizon in days."""
        return self.returns

    def update_nweights(self, next_weights: np.ndarray):
        """
        Updates both the next weights and the investment horizon. Validates both before updating.

        Args:
            next_weights (np.ndarray): New next weights, must sum to 1.

        Raises:
            ValueError: If the next weights do not sum to 1 or if the horizon is not a positive integer.
        """
        if not np.isclose(next_weights.sum(), 1):
            raise ValueError("Next weights must sum to 1.")

        self._next_weights = next_weights


class Portfolio(Position):
    def __init__(self, pf_config: Dict[str, Any], capital: float, weights: np.ndarray,
                 date: datetime, horizon: datetime):
        """
        Initializes a new instance of the Portfolio class, which manages positions and configurations.

        Args:
            pf_config (dict): Configuration of the portfolio, must include at least a 'symbols' key.
        """
        super().__init__(capital, weights, date, horizon)
        if len(pf_config["symbols"]) != weights.shape[0]:
            raise ValueError("weights does not match the number of portfolio assets.")
        self._pf_config: Dict[str, Any] = pf_config
        self._strategies: dict = {}
        self._metrics: Dict[str, Any] = {}
        self._returns: np.ndarray = None

    @property
    def pf_config(self) -> Dict[str, Any]:
        return self._pf_config

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics

    @property
    def strategies(self) -> Dict[str, Any]:
        return self._strategies

    @property
    def returns(self) -> Dict[str, Any]:
        return self._returns

    def update_metrics(self, data: _Data) -> None:
        """
        Updates the portfolio metrics based on the data.
        """

        check_configs(portfolio=self, data=data)
        if "model" not in data.metrics:
            raise ValueError("The provided data is not yet fitted by model")
        self._metrics = data.metrics

    def fit_strat(self, strategies: _Strategies):
        if "model" not in self.metrics:
            raise ValueError("The portfolio metrics is empty, please update with trained data")
        strategies.fit(self)
        self._strategies = strategies.strat_config  # à ajuster en cas de plusieur strategies

    def observed_returns(self, data: _Data):
        check_configs(portfolio=self, data=data, check_date=False)
        if not (self.date >= data.data_config["start_date"] and self.horizon <= data.data_config["end_date"]):
            raise ValueError("Data do not cover the portfolio zone")
        self._returns = data.window_returns(self.date, self.horizon)

    def forward(self, new_horizon: datetime):
        if "fee_rate" not in self.strategies:
            raise ValueError("The portfolio is not yet fitted by strategies")
        if self._returns is None:
            raise ValueError("The portfolio's observed returns is not yet updated by observed data:_Data")

        past_capital = pf_t.capital_fw(self.next_weights, self.weights, self.strategies["fee_rate"], self.capital)
        self._capital = pf_t.fw_portfolio_value(self.next_weights, self.returns, past_capital, self.metrics["scale"])
        self._weights = self.next_weights
        self._date = self.horizon
        self._horizon = new_horizon
        self._metrics = {}
        self._strategies = {}
        self._next_weights = None
        self._returns = None
