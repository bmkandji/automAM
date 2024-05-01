import numpy as np
from datetime import datetime
from typing import Optional

class Position:
    def __init__(self, capital: float, weights: np.ndarray, date: datetime, next_weights: Optional[np.ndarray] = None, horizon: Optional[int] = None):
        """
        Initializes a new instance of the Position class, representing a state of an investment portfolio at a given time.

        Args:
        capital (float): Initial capital amount.
        weights (np.ndarray): Initial weights for the assets in the portfolio, as a NumPy array.
        date (datetime): The date this position is applicable.
        next_weights (np.ndarray, optional): Planned next weights for the assets, as a NumPy array. Defaults to None.
        horizon (int, optional): Investment horizon in days. Defaults to None.
        """
        self.capital = capital  # The amount of capital in the portfolio
        self.weights = weights  # Current asset allocation weights as a NumPy array
        self.date = date  # The specific date for this position
        self.next_weights = next_weights  # Future planned weights for asset reallocation
        self.horizon = horizon  # The investment horizon associated with this position

    def update(self, next_weights: np.ndarray, horizon: int):
        """
        Updates the next planned weights and investment horizon of the position.

        Args:
        next_weights (np.ndarray): New weights to be set as the next weights.
        horizon (int): Updated investment horizon in days.
        """
        self.next_weights = next_weights  # Update the next_weights attribute to the new weights
        self.horizon = horizon  # Update the investment horizon

class Portfolio:
    def __init__(self, pf_config: dict, position: Optional[Position] = None):
        """
        Initializes a new instance of the Portfolio class, which manages positions and configurations.

        Args:
        pf_config (dict): Configuration of the portfolio, must include at least a 'symbols' key.
        position (Position, optional): Initial state of the portfolio. Defaults to None.
        """
        self.pf_config = pf_config  # Stores the portfolio configuration as an attribute
        self.position = position  # Initial position of the portfolio, can be None

    def update_position(self, position: Position):
        """
        Updates the portfolio's current position with the provided new position. Checks if the new position's asset weights match the expected number of symbols.

        Args:
        position (Position): New position to update.

        Raises:
        ValueError: If the number of elements in the position's weights does not match the number of symbols in the portfolio configuration.
        """
        if position.weights.shape[0] != len(self.pf_config["symbols"]):
            raise ValueError("The provided position does not match the expected number of assets.")
        self.position = position  # Updates the position with the new values
