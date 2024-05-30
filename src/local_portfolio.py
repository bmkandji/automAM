import numpy as np
from datetime import datetime
from typing import Dict, Any
from typing import Optional
from src.abstract import _Data, _Strategies
from utils.check import check_configs, checks_weights
from abc import ABC
from utils import portfolio_tools as pf_t
from src.remote_portfolio import RemotePortfolio
from src.common import get_last_trading_day


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
        checks_weights(weights)  # Ensures weights sum to 1

        self._capital: float = capital
        self._weights: np.ndarray = weights
        self._date: datetime = date

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

        pf_config["ref_portfolios"] = pf_config["ref_portfolios"] = {
            key: {
                "capital": capital,
                "weights": np.array(pf_config["ref_portfolios"][key]["next_weights"]),
                "next_weights": np.array(pf_config["ref_portfolios"][key]["next_weights"])
            }
            for key in pf_config["ref_portfolios"].keys()
        }

        self._pf_config: Dict[str, Any] = pf_config
        self._strategy: _Strategies = None
        self._metrics: Dict[str, Any] = {}
        self._next_weights: np.ndarray = weights

    @property
    def pf_config(self) -> Dict[str, Any]:
        """Returns the portfolio configuration dictionary."""
        return self._pf_config

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the current metrics of the portfolio."""
        return self._metrics

    @property
    def strategy(self) -> _Strategies:
        """Returns the strategy configuration applied to the portfolio."""
        return self._strategy

    @property
    def next_weights(self) -> Optional[np.ndarray]:
        """Returns the planned next weights for asset allocation, if available."""
        return self._next_weights

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

    def update_weights(self, strategies: _Strategies):
        """
        Fits strategies to the portfolio, updating the strategy configurations.

        Args:
            strategies (_Strategies): Strategies to be fitted to the portfolio.

        Raises:
            ValueError: If the portfolio metrics are empty or incomplete.
        """
        if "model" not in self.metrics:
            raise ValueError("The portfolio metrics are empty or incomplete, please update with trained data.")
        self.update_cash_weight()
        next_weights = strategies.fit(self)
        checks_weights(next_weights)  # Check sum to 1
        self._next_weights = next_weights
        self._strategy = strategies  # Adjusted to handle multiple strategies

    def forward(self, data: _Data,
                update_weights=True,
                strategies: _Strategies = None,
                rem_portfolio: RemotePortfolio = None,
                update_ref_pf: bool = True):
        """
        Advances the portfolio, updating weights and returns.

        Args:
            update_weights:
            strategies:
            rem_portfolio:
            data:
            update_ref_pf (bool): Flag to determine if reference portfolio weights should be updated.

        Raises:
            ValueError: For invalid dates, missing strategies, or un-updated returns.
        """

        check_configs(portfolio=self, data=data,
                      rportfolio=rem_portfolio, check_date=False,
                      check_scale=True, check_fit_date=False)

        if not (data.data_config["start_date"] <= self.date < data.data_config["end_date"]):
            raise ValueError("The data does not cover the required period for the portfolio.")

        returns = data.window_returns(self.date, data.data_config["end_date"])
        print(f"observed returns: {returns}\n")
        if rem_portfolio is not None:

            capital_weights = rem_portfolio.weights()
            observed_weights = np.array([capital_weights["weights"][asset]
                                         for asset in
                                         ["cash"] + self.pf_config["symbols"]])
            if get_last_trading_day(capital_weights["date"], self.pf_config["market"]) != data.data_config["end_date"]:
                raise ValueError("The date of data and remote portfolio are not coherent")

            checks_weights(observed_weights)

            (self._capital, self._weights) = (capital_weights["capital"],
                                              observed_weights)

        else:
            past_capital = pf_t.capital_fw(self.next_weights, self.weights,
                                           self.strategy.strat_config["fee_rate"], self.capital)

            self._capital, self._weights = pf_t.fw_portfolio_value(self.next_weights,
                                                                   returns, past_capital,
                                                                   self.metrics["scale"])
            print(f"capital: {self.capital}\n, weight: {self.weights}\n")

        self._date = data.data_config["end_date"]

        if not update_ref_pf:
            self._pf_config["ref_portfolios"] = {}

        elif update_ref_pf and self.pf_config["ref_portfolios"]:
            for key in self._pf_config["ref_portfolios"]:
                next_weights = self.pf_config["ref_portfolios"][key]["next_weights"]
                current_weights = self.pf_config["ref_portfolios"][key]["weights"]
                fee_rate = self.strategy.strat_config["fee_rate"]
                current_capital = self.pf_config["ref_portfolios"][key]["capital"]

                past_capital_ref = pf_t.capital_fw(next_weights,
                                                   current_weights,
                                                   fee_rate,
                                                   current_capital)
                (self._pf_config["ref_portfolios"][key]["capital"],
                 self._pf_config["ref_portfolios"][key]["weights"]) = \
                    pf_t.fw_portfolio_value(
                        next_weights,
                        returns,
                        past_capital_ref,
                        self.metrics["scale"])
        else:
            raise ValueError("The references portfolios are note updated since many period.")
        print(update_weights)
        if update_weights:
            self.update_cash_weight()
            self.update_metrics(data)
            strategies = self.strategy if strategies is None else strategies
            self.update_weights(strategies)

        else:
            self._metrics = {}
            self._next_weights = self.weights

    def update_cash_weight(self):
        """
        Update the cash weight in the portfolio configuration if the capital is above the minimum threshold.
        Adjusts the cash weight within the specified min/max bounds based on the current capital.
        Raises an error if the capital is below the minimum threshold.
        """
        # Check if the capital is above the minimum defined
        if self.capital > self.pf_config["fixed_weights"]["minmax_cash"][0]:
            # Calculate the new weight for cash
            new_cash_weight = min(
                max(
                    self.pf_config["fixed_weights"]["minmax_cash"][0] / self.capital,
                    self.pf_config["fixed_weights"]["target_cash_weights"]
                ),
                self.pf_config["fixed_weights"]["minmax_cash"][1] / self.capital
            )
            # Update the portfolio configuration
            self.pf_config["fixed_weights"]["index_And_weights"][1][0] = new_cash_weight
        else:
            # Raise an error if the capital is not sufficient
            raise ValueError("Insufficient capital for investment.")
