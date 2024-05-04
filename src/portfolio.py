import numpy as np
from datetime import datetime
from typing import Dict, Any
from typing import Optional
from src.data_mn import Data
from utils.portfolio_tools import Position
from src.abstract import _Data, _Strategies
from utils.check import check_configs


class Portfolio:
    def __init__(self, pf_config: Dict[str, Any], position: Optional[Position] = None):
        """
        Initializes a new instance of the Portfolio class, which manages positions and configurations.

        Args:
            pf_config (dict): Configuration of the portfolio, must include at least a 'symbols' key.
            position (Position, optional): Initial state of the portfolio. Defaults to None.
        """
        if position and len(pf_config["symbols"]) != position.weights.shape[0]:
            raise ValueError("weights does not match the number of portfolio assets.")
        self._pf_config: Dict[str, Any] = pf_config
        self._position = position  # Initial position of the portfolio, can be None
        self._strategies: dict = {}
        self._metrics: Dict[str, Any] = {}

    @property
    def position(self) -> Optional[Position]:
        """Returns the current position of the portfolio."""
        return self._position

    @property
    def pf_config(self) -> Dict[str, Any]:
        return self._pf_config

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics

    @property
    def strategies(self) -> Dict[str, Any]:
        return self._strategies

    def update_position(self, position: Position):
        """
        Updates the portfolio's current position.
        """
        if position.weights.shape[0] != len(self._pf_config["symbols"]):
            raise ValueError("The provided position does not match the expected number of assets.")
        self._position = position

    def update_metrics(self, data: _Data) -> None:
        """
        Updates the portfolio metrics based on the data.
        """
        check_configs(portfolio=self, data=data)
        self._metrics = data.metrics

    def fit_strat(self, strategies: _Strategies):
        strategies.fit(self)

    def forward(self, data: Data):
        return None
