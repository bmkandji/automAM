import time
import threading
from typing import Dict, List, Union
import numpy as np
import pickle
from src.abstract import _Model, _Data, _Strategies
from src.local_portfolio import Portfolio
from src.remote_portfolio import RemotePortfolio
from src.models import Model
from src.data_mn import Data
from src.strategies import Strategies
from datetime import datetime, timedelta
from src.api import API
from src.common import (get_last_trading_day,
                        market_settings_date,
                        Check_and_update_Date,
                        trunc_decimal,
                        get_current_time,
                        normalize_order)
from utils import portfolio_tools as pf_t
from configs.root_config import set_project_root

# Configure the path to the project root
set_project_root()


class PortfolioManager:
    def __init__(self, pm_settings: Dict[str, Dict[str, Union[str, int, float]]]) -> None:
        """
        Initializes the portfolio manager with a brokerage API.

        :param pm_settings: A configuration dictionary for the portfolio manager.
        """
        # noinspection PyShadowingNames
        pm_config = pm_settings.copy()
        # Update configuration with symbols and market information
        pm_config["data"].update({"symbols": pm_config["symbols"],
                                  "horizon": pm_config["horizon"],
                                  "api_config": pm_config["api"]})

        pm_config["model"].update({"symbols": pm_config["symbols"]})

        pm_config["portfolio"].update({"symbols": pm_config["symbols"],
                                       "market": pm_config["market"]})

        pm_config["rportfolio"].update({"symbols": pm_config["symbols"],
                                        "market": pm_config["market"]})

        strat_config = pm_config["strategy"]
        api_config = pm_config["api"]

        today = get_current_time()
        end_date = today + timedelta(days=365)

        # Get rebalance dates (one hour after market open) for the next year
        self.rebal_date: List[datetime] = market_settings_date(pm_config["market"], today, end_date)

        # Fetch historical data
        self.data: _Data = Data(pm_config["data"])

        # Fit and forecast using the model
        self.model: _Model = Model(pm_config["model"])

        # Initialize strategy
        self.strategy: _Strategies = Strategies(strat_config)

        # Initialize remote portfolio and refresh it
        self.rportfolio: RemotePortfolio = RemotePortfolio(API(api_config), pm_config["rportfolio"])
        self.rportfolio.cancel_all_open_orders()
        position = self.rportfolio.weights()
        # Initialize local portfolio with the refreshed positions

        self.portfolio: Portfolio = Portfolio(
            pf_config=pm_config["portfolio"],
            capital=position["capital"],
            weights=np.array([position["weights"][asset]
                              for asset in
                              ["cash"] + self.rportfolio.pf_config["symbols"]]),
            date=get_last_trading_day(position["date"], pm_config["market"]))

        # Initialize pending orders list
        self.pending_orders: Dict[str, List[Dict[str, Union[str, float]]]] = {}
        self.pm_manager_path: str = pm_config["pm_manger"]
        self.to_init: bool = True

    def __save_state(self, date=None):
        """
        Private method to save the state of the current object to a file specified in self.pm_manager_path.
        """
        try:
            with open(self.pm_manager_path, 'wb') as f:
                pickle.dump(self, f)
            print("State saved successfully.")
        except (IOError, pickle.PickleError) as e:
            print(f"Failed to save state: {e}")

    def __load_state(self):
        """
        Private method to load the state of the current object from a file specified in self.pm_manager_path.
        This method updates the current object's attributes with the loaded state.
        """
        try:
            with open(self.pm_manager_path, 'rb') as f:
                loaded_object = pickle.load(f)
            self.__dict__.update(loaded_object.__dict__)
            print("State loaded successfully.")
        except (IOError, pickle.PickleError) as e:
            print(f"Failed to load state: {e}")

    def __init_update(self):
        today = get_current_time()
        self.data.fetch_data(
            today - timedelta(days=self.data.data_config["hist_dataLen"]), today)
        self.model.fit_fcast(self.data,
                             self.data.data_config["end_date"] +
                             timedelta(days=self.data.data_config["horizon"]))
        self.portfolio.update_metrics(self.data)
        self.portfolio.update_weights(self.strategy)

    def __update_portfolio(self, to_calib: bool = False):
        """
        Updates the portfolio weights with the complete logic to manage existing orders and place new orders.

        """

        if self.to_init:
            self.__init_update()
            self.to_init = False

        else:
            max_attempts = 5
            attempts = 0

            while self.rportfolio.open_orders and attempts < max_attempts:
                time.sleep(60)
                self.rportfolio.cancel_all_open_orders()
                attempts += 1

            if self.rportfolio.open_orders:
                raise ValueError("Failed to refresh portfolio: open orders still present after 5 attempts.")

            self.data.update_data()

            self.model.fit_fcast(self.data,
                                 self.data.data_config["end_date"] +
                                 timedelta(days=self.data.data_config["horizon"]))

            self.portfolio.forward(self.data,
                                   to_calib,
                                   self.strategy,
                                   self.rportfolio,
                                   update_ref_pf=True)

        capital_weights = self.rportfolio.weights()
        weights = np.array([capital_weights["weights"][asset]
                            for asset in
                            ["cash"] + self.portfolio.pf_config["symbols"]])

        next_capital = pf_t.capital_fw(self.portfolio.next_weights, weights,
                                       self.strategy.strat_config["fee_rate"], capital_weights["capital"])

        target_weights = {
            asset: weight
            for asset, weight in zip(["cash"] + self.portfolio.pf_config["symbols"], self.portfolio.next_weights)
        }

        target_values = {asset: next_capital * weight
                         for asset, weight in target_weights.items()
                         if asset != "cash"}

        current_positions = self.rportfolio.positions
        current_prices = (self.rportfolio.broker_api.
                          get_current_prices(self.rportfolio.pf_config["symbols"]))

        buy_orders = []
        sell_orders = []
        fee_norm = 1 / (1 - self.strategy.strat_config["fee_rate"])
        for asset, target_value in target_values.items():
            target_qty = target_values.get(asset, 0) / current_prices[asset]
            difference = target_qty - float(current_positions.get(asset, 0))
            trad_qty = trunc_decimal(abs(difference), 8)  # Troncature à 8 décimales
            if difference > 0:
                # Pour acheter, on utilise la valeur notionnelle
                buy_orders.append({"asset": asset,
                                   "action": "buy",
                                   "units": trad_qty,
                                   "value": trunc_decimal(fee_norm * abs(difference)
                                                          * current_prices[asset], 1),
                                   "type": "notional"})

            elif difference < 0 and trad_qty < float(current_positions.get(asset, 0)):
                # Pour vendre, on utilise les unités
                sell_orders.append({"asset": asset,
                                    "action": "sell",
                                    "units": trad_qty,
                                    "value": trunc_decimal(abs(difference)
                                                           * current_prices[asset], 1),
                                    "type": "qty"})

        # Normaliser et Trier les ordres par valeur décroissante
        buy_orders = normalize_order(buy_orders)
        sell_orders = normalize_order(sell_orders)
        buy_orders.sort(key=lambda x: x["value"], reverse=True)
        sell_orders.sort(key=lambda x: x["value"], reverse=True)

        self.pending_orders = {"sell": sell_orders, "buy": buy_orders}

    def __execute_orders(self):
        """
        Execute pending orders in order of their value and remove them from the list if successful.

        Input:
        - self.pending_orders: A dictionary containing lists of pending buy and sell orders.
          Each order is assumed to be a dictionary with relevant details such as 'value'.

        Output:
        - None. This method updates self.pending_orders by removing successfully executed orders.
        """

        # Execute sell orders
        # Iterate over a copy of the list to allow safe removal of items
        for order in self.pending_orders["sell"][:]:
            result = self.rportfolio.broker_api.place_orders([order])
            if result[0]["success"]:
                self.pending_orders["sell"].remove(order)
                print(f"Sell order executed and removed: {order}")

        # Adjust buy orders if there are no pending sell orders and there are buy orders
        if (not self.pending_orders["sell"]
                and not self.rportfolio.open_orders_Byside()["sell_open_orders"]
                and self.pending_orders["buy"]):
            time.sleep(15)
            total_buy_value = sum(order["value"] for order in self.pending_orders["buy"])
            current_cash = self.rportfolio.cash

            if total_buy_value > current_cash:

                # Scale down each buy order proportionally if total buy value exceeds available cash
                for order in self.pending_orders["buy"]:
                    order["value"] = trunc_decimal(order["value"] * (current_cash / total_buy_value), 1)

                # Remove buy orders with value less than or equal to 0.05 after adjustment
                self.pending_orders["buy"] = normalize_order(self.pending_orders["buy"])

        # Execute buy orders
        # Iterate over a copy of the list to allow safe removal of items
        for order in self.pending_orders["buy"][:]:
            current_cash = self.rportfolio.cash
            if order["value"] > current_cash:
                print(f"Not enough cash for buy order: {order}")
                continue  # Skip this order if not enough cash is available

            result = self.rportfolio.broker_api.place_orders([order])
            if result[0]["success"]:
                self.pending_orders["buy"].remove(order)
                print(f"Buy order executed and removed: {order}")

    def start(self):
        """
        Start the portfolio management process.

        The method operates in a loop, performing calibration and rebalancing
        based on specific dates. It checks whether calibration or rebalancing
        is needed, executes the necessary operations, and then sleeps until
        the next significant event.

        Input:
        - self.rebal_date (list of tuples): List containing tuples of rebalancing dates.
          Each tuple has two elements: the start and end of a rebalancing period.

        Output:
        - None. The method runs indefinitely, managing the portfolio based on dates.

        Workflow:
        - Continuously check and update dates using Check_and_update_Date.
        - If calibration is needed and not already done, update the portfolio.
        - If rebalancing is needed, execute orders and update the portfolio if necessary.
        - Sleep until the next significant event (calibration or rebalancing).
        """

        calib_done = False
        while self.rebal_date:
            to_calib, to_update, self.rebal_date = Check_and_update_Date(self.rebal_date)
            if to_calib and not calib_done:
                print("Calibration period")
                self.__update_portfolio(to_calib)
                calib_done = True
                self.__save_state()
                now = get_current_time()
                to_sleep = max((self.rebal_date[0][1] - now).total_seconds(), 0)
                print(f"Sleeping for {to_sleep} seconds after calibration period")
                time.sleep(to_sleep)
                continue

            if to_update:
                print("Rebalancing period")
                if self.portfolio.date != get_last_trading_day(self.rebal_date[0][1],
                                                               self.rportfolio.pf_config["market"]) or self.to_init:
                    print("Calibration has not been done, Calibration...")
                    self.__update_portfolio(True)

                self.__execute_orders()
                if not self.pending_orders["buy"] and not self.pending_orders["sell"]:
                    now = get_current_time()
                    to_sleep = max((self.rebal_date[1][0] - now).total_seconds(), 0)
                    print(f"Sleeping for {to_sleep} seconds after rebalancing")
                    time.sleep(to_sleep)
                calib_done = False
                continue

            else:
                print("Non-Calibrage-Or-rebalancing period")
                now = get_current_time()
                to_sleep = max((self.rebal_date[0][0] - now).total_seconds(), 0)
                print(f"Sleeping for {to_sleep} seconds during non-rebalancing period")
                time.sleep(to_sleep)

            # Add a sleep period to avoid a too-fast infinite loop
            time.sleep(30)
