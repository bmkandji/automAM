import numpy as np
from src.data_mn import Data
from src.strategies import Strategies
from src.models import Model
from src.portfolio import Portfolio
from utils.load import load_json_config
from datetime import datetime

# data
data_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\data_settings\data_settings.json')
stock_data = Data(data_config)
stock_data.fetch_data(datetime(2008, 1, 1), datetime(2024, 1, 10))
print(stock_data.data)  # Initial data

# model
horizon = datetime(2024, 1, 20)
model_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\model_settings\model_settings.json')
dcc_garch_model = Model(model_config)
dcc_garch_model.fit_fcast(stock_data, horizon)

# portfolio
pf_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\portfolio_settings\pf_settings.json')
n = len(pf_config["symbols"])
weights = np.ones(n) / n
date_deb = datetime(2024, 1, 9)


portfolio = Portfolio(pf_config, 1, weights, date_deb, horizon)
portfolio.update_metrics(stock_data)


# strategies
strat_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\strat_settings\strat_settings.json')
strategy = Strategies(strat_config)
portfolio.fit_strat(strategy)

# result

print(portfolio.metrics)
print(portfolio.strategies)
print(portfolio.next_weights)
