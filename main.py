import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_mn import Data
from src.strategies import Strategies
from src.models import Model
from src.portfolio import Portfolio
from utils.load import load_json_config
import datetime as dt
import copy
import pandas_market_calendars as mcal



# data
data_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\data_settings\data_settings.json')
data = Data(data_config)
data.fetch_data(dt.datetime(2008, 1, 1), dt.datetime(2018, 1, 3))
print(data.data)
# model
horizon = dt.datetime(2018, 1, 10)
model_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\model_settings\model_settings.json')
model = Model(model_config)
model.fit_fcast(data, horizon)

# portfolio
pf_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\portfolio_settings\pf_settings.json')
n = len(pf_config["symbols"]) + 1
weights = np.ones(n) / n
date_deb = dt.datetime(2018, 1, 2)

portfolio = Portfolio(pf_config, 1, weights, date_deb)
portfolio.update_metrics(data)

# strategies

strat_config = load_json_config(r'C:\Users\MatarKANDJI\automAM\src\strat_settings\strat_settings.json')
strategy = Strategies(strat_config)
portfolio.update_weights(strategy)



# Créer un calendrier pour le NASDAQ
nasdaq = mcal.get_calendar('NASDAQ')

# Définir les dates de début et de fin
start_date = dt.datetime(2018, 1, 4)
end_date = dt.datetime(2018, 4, 4)

# Récupérer les jours de trading
# Si vous voulez que le DatetimeIndex soit sans fuseau horaire dès le début
trading_days = nasdaq.valid_days(start_date=start_date, end_date=end_date).tz_localize(None)

# Convertir en liste de datetime pour une manipulation facile
date_list = [day.to_pydatetime() for day in trading_days]

# Afficher la liste des jours
for day in date_list:
    print(day)


portfolios = [copy.deepcopy(portfolio)]

for item in range(len(date_list)-11):
    # forward
    print(portfolio.date)
    data.update_data(date_list[item])
    model.fit_fcast(data, date_list[item+10])
    portfolio.forward(data)
    portfolios.append(copy.deepcopy(portfolio))

asset_path = pd.DataFrame()
ref_pf = list(portfolio.pf_config['ref_portfolios'].keys())

col_names = ["Date", "Our_pf"] + ref_pf

assets_value = []

for pf in portfolios:
    assets_value.append([pf.date, pf.capital] + [pf.pf_config['ref_portfolios'][key]["capital"] for key in ref_pf])

prices = pd.DataFrame(assets_value, columns=col_names)
prices['Date'] = pd.to_datetime(prices['Date'])
prices = prices.set_index('Date')


# plot
# Tracer toutes les colonnes du DataFrame en fonction de l'index
ax = prices.plot(figsize=(10, 5), title='Asset Prices Over Time')

# Nommer les axes
ax.set_xlabel('Date')
ax.set_ylabel('Price')

# Afficher la légende
ax.legend(title='Assets')

# Afficher le graphique
plt.show()
