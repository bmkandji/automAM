import numpy as np
from src.data_mn import Data
from src.strategies import Strategies
from src.models import Model
from src.portfolio import Portfolio
from utils.portfolio_tools import Position
from utils.load import load_json_config
from datetime import datetime


# Usage example
data_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\data_settings\data_settings.json')
stock_data = Data(data_config)
stock_data.fetch_data(datetime(2008, 1, 1), datetime(2024, 1, 10))
print(stock_data.data)  # Initial data

#portfolio
pf_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\portfolio_settings\pf_settings.json')

n = len(stock_data.data_config["symbols"])
weights = np.ones(n) / n
horizon = datetime(2024, 1, 20)
date_deb = datetime(2024, 1, 9)
position = Position(1, weights, date_deb, horizon)
portfolio = Portfolio(pf_config)
portfolio.update_position(position)


#model

model_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\model_settings\model_settings.json')
dcc_garch_model = Model(model_config)
dcc_garch_model.fit_fcast(stock_data, horizon)

portfolio.update_metrics(stock_data)

stock_data.update_data(datetime(2024, 3, 10))

print(stock_data.metrics)
print(dcc_garch_model.metrics)
#portfolio.update_metrics(stock_data)
print(portfolio.metrics)



# strategies

print(portfolio.pf_config)
strat_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\strat_settings\strat_settings.json')
strategy = Strategies(strat_config)
portfolio.fit_strat(strategy)
print(position._next_weights)
print(position._date)
