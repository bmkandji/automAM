import numpy as np
import pandas as pd
import utils.load_model as lo_m
from utils.portfolio_tools import mean_variance_portfolio
from tools.settings import _Position


class AmStrategies:
    def __init__(self, mean_var: dict, strat_config: str, _position: _Position):
        self.strat_config = lo_m.load_json_config(strat_config)
        self.mean_var = mean_var
        self._position = _position

    def fit(self):
        arg = [self.mean_var["mean"], self.mean_var["covariance"],
              self.strat_config["aversion"], self.strat_config["fee_rate"],self.strat_config["bounds"], self._position.weights, self._position.capital]
        self._position.update( mean_variance_portfolio(*arg), self.mean_var["n_ahead"])
