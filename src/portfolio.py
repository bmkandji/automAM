import numpy as np
from datetime import datetime
from typing import Dict, Any
from typing import Optional
from src.abstract import _Data, _Strategies
from utils.check import check_configs, checks_weight
from abc import ABC
from utils import portfolio_tools as pf_t


class Position(ABC):
    def __init__(self, capital: float, weights: np.ndarray, date: datetime):
        """
        Initializes a new instance of the Position class with initial capital, asset weights, and dates.

        Args:
            capital (float): Initial capital amount. Must be greater than zero.
            weights (np.ndarray): Asset allocation weights, which must sum to 1.
            date (datetime): Effective date of this position.

        Raises:
            ValueError: If the initial conditions for capital or weights are not met.
        """
        if capital <= 0:
            raise ValueError("Capital must be greater than zero.")
        checks_weight(weights)  # Ensures weights sum to 1

        self._capital: float = capital
        self._weights: np.ndarray = weights
        self._date: datetime = date
        self._opti_next_weights: np.ndarray = None

    @property
    def capital(self) -> float:
        """Returns the current capital amount of the position."""
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
    def opti_next_weights(self) -> Optional[np.ndarray]:
        """Returns the planned next weights for asset allocation, if available."""
        return self._opti_next_weights


class Portfolio(Position):
    def __init__(self, pf_config: Dict[str, Any], capital: float,
                 weights: np.ndarray, date: datetime):
        """
        Initializes a new instance of the Portfolio class, extending the Position with portfolio configurations.

        Args:
            pf_config (dict): Configuration of the portfolio including asset symbols and reference portfolios.
            capital (float): Initial capital amount.
            weights (np.ndarray): Asset allocation weights.
            date (datetime): Effective date of the portfolio.

        Raises:
            ValueError: If the number of weights does not match the number of symbols.
        """
        super().__init__(capital, weights, date)
        if len(pf_config["symbols"]) + 1 != weights.shape[0]:
            raise ValueError("The number of weights does not match the number of portfolio assets.")

        if "ref_portfolios" not in pf_config:
            pf_config["ref_portfolios"] = {}

        pf_config["ref_portfolios"] = {key: np.array(value) for key, value in
                                       pf_config['ref_portfolios'].items()}  # Convert lists to arrays

        self._pf_config: Dict[str, Any] = pf_config
        self._strategies: _Strategies = None
        self._metrics: Dict[str, Any] = {}
        self._refAsset_capital: Dict[str, float] = {key: capital for key in pf_config["ref_portfolios"].keys()}

        # add a fictive portfolio reference representing the theoretical optimal weights
        self._refAsset_capital["Theoretical_pf"] = self.capital
        self._pf_config["ref_portfolios"]["Theoretical_pf"] = self.weights

    @property
    def pf_config(self) -> Dict[str, Any]:
        """Returns the portfolio configuration dictionary."""
        return self._pf_config

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the current metrics of the portfolio."""
        return self._metrics

    @property
    def strategies(self) -> Dict[str, Any]:
        """Returns the strategy configuration applied to the portfolio."""
        return self._strategies

    @property
    def refAsset_capital(self) -> Dict[str, float]:
        """Returns the reference asset capital amounts, indexed by asset key."""
        return self._refAsset_capital

    def update_metrics(self, data: _Data) -> None:
        """
        Updates the portfolio metrics based on provided data.

        Args:
            data (_Data): Data containing necessary metrics for updating the portfolio.

        Raises:
            ValueError: If the data lacks a model fit necessary for the portfolio metrics.
        """
        check_configs(portfolio=self, data=data)
        if "model" not in data.metrics:
            raise ValueError("The provided data lacks a necessary model fit.")
        self._metrics = data.metrics  # Update the portfolio's metrics

    def fit_strat(self, strategies: _Strategies):
        """
        Fits strategies to the portfolio, updating the strategy configurations.

        Args:
            strategies (_Strategies): Strategies to be fitted to the portfolio.

        Raises:
            ValueError: If the portfolio metrics are empty or incomplete.
        """
        if "model" not in self.metrics:
            raise ValueError("The portfolio metrics are empty or incomplete, please update with trained data.")
        opti_next_weights = strategies.fit(self)
        checks_weight(opti_next_weights)  # Check sum to 1
        self._opti_next_weights = opti_next_weights
        self._strategies = strategies  # Adjusted to handle multiple strategies

    def forward(self, data: _Data, right_next_weight: np.ndarray = None, update_ref_pf: bool = True,
                fit_strat = False, strategies: _Strategies = None):
        """
        Advances the portfolio, updating weights and returns.

        Args:
            data:
            right_next_weight (np.ndarray, optional): Corrected next weights to apply.
            update_ref_pf (bool): Flag to determine if reference portfolio weights should be updated.

        Raises:
            ValueError: For invalid dates, missing strategies, or unupdated returns.
        """
        print(data.data_config["start_date"])
        check_configs(portfolio=self, data=data, check_date=False)
        if not (data.data_config["start_date"] <= self.date < data.data_config["end_date"]):
            raise ValueError("The data does not cover the required period for the portfolio.")
        returns = data.window_returns(self.date,
                                      data.data_config["end_date"])  # Compute and store returns
        print(returns)
        # Check if necessary strategies are present in the portfolio
        if "fee_rate" not in self.strategies.strat_config:
            raise ValueError("The portfolio lacks necessary strategies.")
        print(self.opti_next_weights,
              self.pf_config["ref_portfolios"]["Theoretical_pf"],
              self.strategies.strat_config["fee_rate"], self.refAsset_capital["Theoretical_pf"])
        # Evaluate the capital after fee of the fictive portfolio and update the weights
        self._refAsset_capital["Theoretical_pf"] = pf_t.capital_fw(self.opti_next_weights,
                                                                   self.pf_config["ref_portfolios"]["Theoretical_pf"],
                                                                   self.strategies.strat_config["fee_rate"],
                                                                   self.refAsset_capital["Theoretical_pf"])
        print(self._refAsset_capital["Theoretical_pf"])

        self._pf_config["ref_portfolios"]["Theoretical_pf"] = self.opti_next_weights

        # If corrected weights are provided, validate them and update the optimal next weights
        if right_next_weight:
            checks_weight(right_next_weight)
            self._opti_next_weights = right_next_weight

        # Calculate the portfolio value at t+
        past_capital = pf_t.capital_fw(self.opti_next_weights, self.weights,
                                       self.strategies.strat_config["fee_rate"], self.capital)
        self._capital = pf_t.fw_portfolio_value(self.opti_next_weights, returns,
                                                past_capital, self.metrics["scale"])

        # Update portfolio attributes
        self._weights = self.opti_next_weights
        self._date = data.data_config["end_date"]

        # Update reference portfolio values if required
        if not update_ref_pf:
            self._refAsset_capital = {}

        elif update_ref_pf and self._refAsset_capital:
            print(self._refAsset_capital.items())
            ref_capitalNetfee = {
                key: pf_t.capital_fw(
                    self.pf_config["ref_portfolios"][key],
                    self.pf_config["ref_portfolios"][key],
                    self.strategies.strat_config["fee_rate"],
                    capital)
                for key, capital in self._refAsset_capital.items()
                if key != "Theoretical_pf"
            }

            ref_capitalNetfee["Theoretical_pf"] = self._refAsset_capital["Theoretical_pf"]
            print(ref_capitalNetfee)

            self._refAsset_capital = {
                key: pf_t.fw_portfolio_value(
                    self.pf_config["ref_portfolios"][key],
                    returns,
                    capital,
                    self.metrics["scale"]
                )
                for key, capital in ref_capitalNetfee.items()
                if key in self.pf_config["ref_portfolios"]
            }
        else:
            raise ValueError("The references portfolios are note updated since many period.")

        # Reset internal attributes
        self._metrics = {}
        self._strategies = {}
        self._opti_next_weights = None
