from utils.portfolio_tools import mean_variance_portfolio, max_return, tracking_error
import numpy as np
from src.local_portfolio import Portfolio
from src.abstract import _Strategies


class Strategies(_Strategies):
    def __init__(self, strat_config: dict):
        """
        Initializes an instance of the Strategies class with the given parameters.

        Parameters:
        strat_config (dict) : Configuration dictionary for the strategy that includes details such as
                              risk aversion, fee rate, asset weight bounds, etc.

        This method loads the specified strategy configuration from the configuration dictionary
        and initializes the inherited _Strategies instance.
        """
        super().__init__(strat_config)

    def fit(self, portfolio: Portfolio) -> np.ndarray:
        """
        Adjusts the portfolio position using an optimization strategy specified in strat_config.

        Parameters:
        portfolio (Portfolio) : Portfolio object containing current market data (means, covariances, etc.)
                                and the current state of the portfolio (weights, capital, etc.).

        Returns:
        np.ndarray : The new portfolio weights after optimization.

        This method selects the optimization algorithm based on the specified strategy ('mean_var',
         'max_return', or 'tracking_error') and calculates the optimal new weights for the portfolio.
        """

        if self.strat_config["strategy"] == "mean_var":
            arg = [portfolio.metrics["mean"], portfolio.metrics["covariance"],
                   self.strat_config["aversion"], self.strat_config["fee_rate"],
                   self.strat_config["bounds"], portfolio.weights, portfolio.capital,
                   portfolio.metrics["scale"], np.array(portfolio.pf_config["fixed_weights"]["index_And_weights"]),
                   False, False]
            return mean_variance_portfolio(*arg)

        elif self.strat_config["strategy"] == "max_return":
            arg = [portfolio.metrics["mean"], portfolio.metrics["covariance"],
                   self.strat_config["max_vol"], self.strat_config["fee_rate"],
                   self.strat_config["bounds"], portfolio.weights, portfolio.capital,
                   portfolio.metrics["scale"], np.array(portfolio.pf_config["fixed_weights"]["index_And_weights"]),
                   False, False]
            return max_return(*arg)

        elif self.strat_config["strategy"] == "tracking_error":
            arg = [portfolio.metrics["mean"], portfolio.metrics["covariance"],
                   portfolio.pf_config["ref_portfolios"][self.strat_config["ref_portfolio"]],
                   self.strat_config["tol"], self.strat_config["fee_rate"],
                   self.strat_config["bounds"], portfolio.weights, portfolio.capital,
                   portfolio.metrics["scale"], np.array(portfolio.pf_config["fixed_weights"]["index_And_weights"]),
                   False, False]
            return tracking_error(*arg)

        else:
            raise ValueError(f"Unknown strategy: {self.strat_config['strategy']}")
