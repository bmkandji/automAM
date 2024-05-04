import numpy as np
from src.data_mn import Data
from src.strategies import Strategies
from src.models import Model
from src.portfolio import Portfolio
from utils.load import load_json_config
from datetime import datetime

# data
data_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\data_settings\data_settings.json')
data = Data(data_config)
data.fetch_data(datetime(2008, 1, 1), datetime(2024, 1, 10))
print(data.data)  # Initial data

# model
horizon = datetime(2024, 1, 20)
model_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\model_settings\model_settings.json')
model = Model(model_config)
model.fit_fcast(data, horizon)

# portfolio
pf_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\portfolio_settings\pf_settings.json')
n = len(pf_config["symbols"])
weights = np.ones(n) / n
date_deb = datetime(2024, 1, 9)


portfolio = Portfolio(pf_config, 1, weights, date_deb, horizon)
portfolio.update_metrics(data)


# strategies
strat_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\strat_settings\strat_settings.json')
strategy = Strategies(strat_config)
portfolio.fit_strat(strategy)

# result

print(portfolio.metrics)
print(portfolio.strategies)
print(portfolio.next_weights)



# suite

next_horizon = datetime(2024, 1, 31)
data.update_data(next_horizon)

portfolio.observed_returns(data)
print(portfolio.returns)

new_horizon = datetime(2024, 1, 31)
portfolio.forward(new_horizon)
print(portfolio.capital)
