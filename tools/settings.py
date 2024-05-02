import numpy as np
from datetime import datetime
from typing import Optional


class Position:
    def __init__(self, capital: float, weights: np.ndarray, date: datetime,
                 next_weights: Optional[np.ndarray] = None, horizon: Optional[int] = None):
        """
        Initializes a new instance of the Position class.

        Args:
            capital (float): Initial capital amount, must be greater than zero.
            weights (np.ndarray): Asset allocation weights, must sum to 1.
            date (datetime): Effective date of this position.
            next_weights (np.ndarray, optional): Future asset allocation weights.
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
        self._next_weights = next_weights
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
    def horizon(self) -> Optional[int]:
        """Returns the investment horizon in days."""
        return self._horizon

    def update(self, next_weights: np.ndarray, horizon: int):
        """
        Updates both the next weights and the investment horizon. Validates both before updating.

        Args:
            next_weights (np.ndarray): New next weights, must sum to 1.
            horizon (int): New investment horizon, must be a positive integer.

        Raises:
            ValueError: If the next weights do not sum to 1 or if the horizon is not a positive integer.
        """
        if not np.isclose(next_weights.sum(), 1):
            raise ValueError("Next weights must sum to 1.")
        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")

        self._next_weights = next_weights
        self._horizon = horizon

class Portfolio:
    def __init__(self, pf_config: dict, position: Optional[Position] = None):
        """
        Initializes a new instance of the Portfolio class, which manages positions and configurations.

        Args:
            pf_config (dict): Configuration of the portfolio, must include at least a 'symbols' key.
            position (Position, optional): Initial state of the portfolio. Defaults to None.
        """
        if position and len(pf_config["symbols"]) != position.weights.shape[0]:
            raise ValueError("weights does not match the number of portfolio assets.")

        self._pf_config = pf_config  # Stores the portfolio configuration as an attribute
        self._position = position  # Initial position of the portfolio, can be None
        self._risk_metric = None
    @property
    def position(self) -> Optional[Position]:
        """Returns the current position of the portfolio."""
        return self._position

    @property
    def pf_config(self) -> dict:
        """Returns the portfolio configuration."""
        return self._pf_config

    @property
    def risk_metric(self) -> dict:
        return self._risk_metric

    def update_position(self, position: Position):
        """
        Updates the portfolio's current position.
        """
        if position.weights.shape[0] != len(self._pf_config["symbols"]):
            raise ValueError("The provided position does not match the expected number of assets.")
        self._position = position

    def update_risk(self, risk: dict):
        """
        Updates the portfolio's risk metrics.
        """
        if 'mean' not in risk or 'covariance' not in risk or "horizon" not in risk:
            raise ValueError("Risk metrics must include 'mean' and 'covariance' and 'horizon'.")
        if risk['mean'].shape[0] != len(self._pf_config["symbols"]):
            raise ValueError("Risk 'mean' does not match the number of portfolio assets.")
        self._risk_metric = risk
