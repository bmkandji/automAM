import numpy as np
from src.data_monitoring import StockData
from src.strategies import AmStrategies
from src.models import MeanVar_Model
from tools.settings import Portfolio, Position
from utils.load_model import load_json_config
from datetime import datetime


# Usage example
data_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\data_settings\data_settings.json')
stock_data = StockData(data_config)
stock_data.fetch_data(datetime(2008, 1, 1), datetime(2024, 1, 10))
print(stock_data.data)  # Initial data

#portfolio
pf_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\tools\pf_settings.json')

n = len(stock_data.data_config["symbols"])
weights = np.ones(n) / n
horizon = datetime(2024, 1, 20)
date_deb = datetime(2024, 1, 9)
position = Position(1, weights, date_deb, horizon)
portfolio = Portfolio(pf_config)
portfolio.update_position(position)

#model

model_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\model_settings\model_settings.json')
dcc_garch_model = MeanVar_Model(model_config)
dcc_garch_model.fit_fcast(portfolio, stock_data)
stock_data.update_data(datetime(2024, 3, 10))

print(portfolio.risk_metric)



# strategies

print(portfolio.pf_config)
strat_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\strat_settings\strat_settings.json')
strategy = AmStrategies(strat_config)
strategy.fit(portfolio)
print(position._next_weights)
print(position._date)
