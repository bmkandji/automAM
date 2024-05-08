import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from typing import Optional
from src.abstract import _Data, _Strategies
from utils.check import check_configs, checks_weight
from abc import ABC
from utils import portfolio_tools as pf_t


class Position(ABC):
    def __init__(self, capital: float, weights: np.ndarray, date: datetime, horizon: datetime):
        """
        Initializes a new instance of the Position class with initial capital, asset weights, and dates.

        Args:
            capital (float): Initial capital amount. Must be greater than zero.
            weights (np.ndarray): Asset allocation weights, which must sum to 1.
            date (datetime): Effective date of this position.
            horizon (datetime): Investment horizon represented as a datetime.

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
        self._horizon: datetime = horizon

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

    @property
    def horizon(self) -> datetime:
        """Returns the investment horizon as a datetime object."""
        return self._horizon

    @property
    def returns(self) -> np.ndarray:
        """Placeholder property to provide structure; actual implementation should follow."""
        # This needs proper implementation to function correctly
        return self.returns  # This could lead to a recursive call if not implemented properly

    def update_opti_nweights(self, opti_next_weights: np.ndarray):
        """
        Updates the next weights after validating that they sum to 1.

        Args:
            opti_next_weights (np.ndarray): New planned weights, must sum to 1.

        Raises:
            ValueError: If the next weights do not sum to 1.
        """
        checks_weight(opti_next_weights)  # Check sum to 1
        self._opti_next_weights = opti_next_weights  # Update the weights


class Portfolio(Position):
    def __init__(self, pf_config: Dict[str, Any], capital: float, weights: np.ndarray, date: datetime,
                 horizon: datetime):
        """
        Initializes a new instance of the Portfolio class, extending the Position with portfolio configurations.

        Args:
            pf_config (dict): Configuration of the portfolio including asset symbols and reference portfolios.
            capital (float): Initial capital amount.
            weights (np.ndarray): Asset allocation weights.
            date (datetime): Effective date of the portfolio.
            horizon (datetime): Investment horizon represented as a datetime.

        Raises:
            ValueError: If the number of weights does not match the number of symbols.
        """
        super().__init__(capital, weights, date, horizon)
        if len(pf_config["symbols"])+1 != weights.shape[0]:
            raise ValueError("The number of weights does not match the number of portfolio assets.")

        if "ref_portfolios" not in pf_config:
            pf_config["ref_portfolios"] = {}

        pf_config["ref_portfolios"] = {key: np.array(value) for key, value in
                                       pf_config['ref_portfolios'].items()}  # Convert lists to arrays

        self._pf_config: Dict[str, Any] = pf_config
        self._strategies: _Strategies = {}
        self._metrics: Dict[str, Any] = {}
        self._returns: pd.Series = None
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
    def returns(self) -> np.ndarray:
        """Returns the computed returns for the portfolio."""
        return self._returns.values

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
        strategies.fit(self)
        self._strategies = strategies.strat_config  # Adjusted to handle multiple strategies

    def observed_returns(self, data: _Data):
        """
        Updates portfolio with returns observed over the specified period.

        Args:
            data (_Data): Data containing the returns over a specified period.

        Raises:
            ValueError: If the data does not cover the required portfolio period.
        """
        check_configs(portfolio=self, data=data, check_date=False)
        if not (self.date >= data.data_config["start_date"] and self.horizon <= data.data_config["end_date"]):
            raise ValueError("The data does not cover the required period for the portfolio.")
        self._returns = data.window_returns(self.date, self.horizon)  # Compute and store returns

    def forward(self, new_horizon: datetime, right_next_weight: np.ndarray = None, update_ref_pf: bool = True):
        """
        Advances the portfolio to a new horizon, updating weights and returns.

        Args:
            new_horizon (datetime): The new horizon date for the portfolio.
            right_next_weight (np.ndarray, optional): Corrected next weights to apply.
            update_ref_pf (bool): Flag to determine if reference portfolio weights should be updated.

        Raises:
            ValueError: For invalid dates, missing strategies, or unupdated returns.
        """
        # Check if the new horizon is later than the current one
        if new_horizon <= self.horizon:
            raise ValueError("The new horizon must be later than the current horizon.")

        # Check if necessary strategies are present in the portfolio
        if "fee_rate" not in self.strategies:
            raise ValueError("The portfolio lacks necessary strategies.")

        # Check if the portfolio's returns have been updated
        if self._returns is None:
            raise ValueError("The portfolio's returns have not been updated.")

        # Evaluate the capital after fee of the fictive portfolio and update the weights
        self._refAsset_capital["Theoretical_pf"] = pf_t.capital_fw(self.opti_next_weights,
                                                                   self.pf_config["ref_portfolios"]["Theoretical_pf"],
                                                                   self.strategies["fee_rate"],
                                                                   self.refAsset_capital["Theoretical_pf"])

        self._pf_config["ref_portfolios"]["Theoretical_pf"] = self.opti_next_weights

        # If corrected weights are provided, validate them and update the optimal next weights
        if right_next_weight:
            checks_weight(right_next_weight)
            self._opti_next_weights = right_next_weight

        # Calculate the portfolio value at the new horizon
        past_capital = pf_t.capital_fw(self.opti_next_weights, self.weights,
                                       self.strategies["fee_rate"], self.capital)
        self._capital = pf_t.fw_portfolio_value(self.opti_next_weights, self.returns,
                                                past_capital, self.metrics["scale"])

        # Update portfolio attributes
        self._weights = self.opti_next_weights
        self._date = self.horizon
        self._horizon = new_horizon

        # Update reference portfolio values if required
        if not update_ref_pf:
            self._refAsset_capital = {}
        elif update_ref_pf and self._refAsset_capital:
            print(self._refAsset_capital.items())
            ref_capitalNetfee = {
                key: pf_t.capital_fw(
                    self.pf_config["ref_portfolios"][key],
                    self.pf_config["ref_portfolios"][key],
                    self.strategies["fee_rate"],
                    capital)
            for key, capital in self._refAsset_capital.items()
            }
            print(ref_capitalNetfee)

            self._refAsset_capital = {
                key: pf_t.fw_portfolio_value(
                    self.pf_config["ref_portfolios"][key],
                    self.returns,
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
        self._returns = None
