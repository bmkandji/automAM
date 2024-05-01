import numpy as np
from datetime import datetime
from typing import Optional

class _Position:
    def __init__(self, capital: float, weights: np.ndarray, date: datetime, next_weights: Optional[np.ndarray] = None, horizon: int =None):
        """
        Initializes a new instance of the Position class.

        Args:
        capital (float): Initial capital amount.
        weights (np.ndarray): Initial weights for the assets in the portfolio, as a NumPy array.
        date (datetime): The date of this position.
        next_weights (np.ndarray, optional): Planned next weights for the assets, as a NumPy array. Defaults to None.
        """
        self.capital = capital  # The amount of capital in the portfolio
        self.weights = weights  # Asset allocation weights as a NumPy array
        self.date = date  # The date corresponding to this position
        self.next_weights = next_weights  # Future planned weights as a NumPy array
        self.horizon = horizon

    def update(self, next_weights: np.ndarray, horizon: int):
        """
        Updates the planned next weights of the position.

        Args:
        next_weights (np.ndarray): New weights to be set as the next weights.
        """
        self.next_weights = next_weights  # Update the next_weights attribute to the new weights
        self.horizon = horizon

class Portfolio:
    def __init__(self, pf_config: dict, _position: Optional[_Position] = None):
        """
        Initializes a new instance of the Portfolio class.

        Args:
        pf_config (dict): Configuration of the portfolio, must include at least a 'symbols' key.
        position (Position, optional): Initial state of the portfolio. Defaults to None.
        """
        self.pf_config = pf_config  # Stores the portfolio configuration as an attribute
        self._position = _position  # Initial position of the portfolio, can be None

    def update_position(self, _position: _Position):
        """
        Updates the portfolio's position with the provided data. Verifies that the position
        contains the same number of elements as the symbols in the portfolio configuration.

        Args:
        position (Position): New position to update.

        Raises:
        ValueError: If the number of elements in the position does not match the number of symbols.
        """
        # Checks if the number of elements in the new position matches the number of configured symbols
        if _position.weights.shape[0] != len(self.pf_config["symbols"]):
            raise ValueError("The provided position does not match the expected number of assets.")

        self._position = _position  # Updates the position with the new values
