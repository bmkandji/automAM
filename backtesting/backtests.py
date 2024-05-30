import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_mn import Data, Data
from src.strategies import Strategies
from src.models import Model
from src.local_portfolio import Portfolio
from utils.load import load_json_config
from datetime import timedelta, datetime
import copy
import exchange_calendars as ecals
import matplotlib.dates as mdates
from configs.root_config import set_project_root

# Configure the path to the project root
set_project_root()


# Load configurations
def load_configs():
    data_config = load_json_config(r'src/data_settings/data_settings.json')
    model_config = load_json_config(r'src/model_settings/model_settings.json')
    pf_config = load_json_config(r'src/portfolio_settings/pf_settings.json')
    strat_config = load_json_config('src/strat_settings/strat_settings.json')["tracking_error"]
    return data_config, model_config, pf_config, strat_config


# Initialize data, model, portfolio, and strategy
def initialize_components(data_config, model_config, pf_config, strat_config):
    data = Data(data_config)
    data.fetch_data(pd.Timestamp(year=2005, month=1, day=1).tz_localize('UTC').tz_localize(None),
                    pd.Timestamp(year=2018, month=1, day=3).tz_localize('UTC').tz_localize(None))
    model = Model(model_config)
    horizon = pd.Timestamp(year=2018, month=1, day=10).tz_localize('UTC').tz_localize(None)
    model.fit_fcast(data, horizon)

    n = len(pf_config["symbols"]) + 1
    weights = np.ones(n) / n
    portfolio = Portfolio(pf_config, 1, weights,
                          pd.Timestamp(year=2018, month=1, day=2).tz_localize('UTC').tz_localize(None))
    portfolio.update_metrics(data)

    strategy = Strategies(strat_config)
    portfolio.update_weights(strategy)
    return data, model, portfolio


# Retrieve NASDAQ trading days
def get_trading_days(start_date, end_date, market):
    start_date = datetime(start_date.year, start_date.month, start_date.day)
    end_date = datetime(end_date.year, end_date.month, end_date.day)
    nasdaq_cal = ecals.get_calendar(market)
    sessions = nasdaq_cal.sessions_in_range(start_date, end_date)
    return [pd.Timestamp(session).tz_localize('UTC').tz_localize(None) for session in sessions]


# Dynamic plotting setup
def dynamic_plotting(date_list, portfolio, data, model):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.ion()  # Interactive mode on

    portfolios = [copy.deepcopy(portfolio)]
    assets_value = [{'Date': portfolio.date, 'Our_pf': portfolio.capital,
                     **{key: portfolio.pf_config['ref_portfolios'][key]["capital"] for key in
                        portfolio.pf_config['ref_portfolios']}}]

    # Setup monthly date formatting on the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Ticks at the start of each month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year and month format
    plt.xticks(rotation=45)

    lines = {'Our_pf': ax.plot([], [], label='Our Portfolio')[0]}
    for key in portfolio.pf_config['ref_portfolios']:
        lines[key] = ax.plot([], [], label=key)[0]
    plt.legend(title="Assets")

    for item in range(len(date_list) - 11):
        current_date = date_list[item]
        data.update_data(current_date + timedelta(days=1))
        model.fit_fcast(data, date_list[item + 10])
        portfolio.forward(data)
        portfolios.append(copy.deepcopy(portfolio))

        new_data = {'Date': portfolio.date, 'Our_pf': portfolio.capital,
                    **{key: portfolio.pf_config['ref_portfolios'][key]["capital"] for key in
                       portfolio.pf_config['ref_portfolios']}}
        assets_value.append(new_data)

        prices = pd.DataFrame(assets_value)
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices.set_index('Date', inplace=True)

        for key in lines:
            lines[key].set_data(prices.index, prices[key])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot


# Main function to encapsulate the execution logic
def main():
    data_config, model_config, pf_config, strat_config = load_configs()
    data, model, portfolio = initialize_components(data_config, model_config, pf_config, strat_config)
    start_date = pd.Timestamp(year=2018, month=1, day=3).tz_localize('UTC').tz_localize(None)
    end_date = pd.Timestamp(year=2024, month=5, day=10).tz_localize('UTC').tz_localize(None)
    date_list = get_trading_days(start_date, end_date, portfolio.pf_config["market"])
    dynamic_plotting(date_list, portfolio, data, model)


if __name__ == "__main__":
    main()
