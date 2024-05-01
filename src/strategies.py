import numpy as np
import pandas as pd
import utils.load_model as lo_m
from utils.portfolio_tools import mean_variance_portfolio
from tools.settings import Position


class AmStrategies:
    def __init__(self, mean_var: dict, strat_config: str, position: Position):
        self.strat_config = lo_m.load_json_config(strat_config)
        self.mean_var = mean_var
        self.position = position

    def fit(self):
        arg = [self.mean_var["mean"], self.mean_var["covariance"],
              self.strat_config["aversion"], self.strat_config["fee_rate"],self.strat_config["bounds"], self.position.weights, self.position.capital]
        self.position.update( mean_variance_portfolio(*arg), self.mean_var["n_ahead"])
