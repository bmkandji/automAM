import time
import threading
from typing import Dict, List, Union
from src.abstract import _Model, _Data, _Strategies
from src.local_portfolio import Portfolio
from src.remote_portfolio import RemotePortfolio
from src.models import Model
from src.data_mn import Data
from src.strategies import Strategies
from datetime import datetime, timedelta
from api import API
from src.common import (get_last_trading_day,
                        market_settigs_date,
                        Check_and_update_Date,
                        trunc_decimal)
from utils import portfolio_tools as pf_t


class PortfolioManager:
    def __init__(self, pm_config: Dict[str, Dict[str, Union[str, int, float]]]) -> None:
        """
        Initializes the portfolio manager with a brokerage API.

        :param pm_config: A configuration dictionary for the portfolio manager.
        """
        # Update configuration with symbols and market information
        pm_config["data"].update({"symbols": pm_config["symbols"],
                                  "horizon": pm_config["horizon"]})

        pm_config["model"].update({"symbols": pm_config["symbols"]})

        pm_config["portfolio"].update({"symbols": pm_config["symbols"],
                                       "market": pm_config["market"]})

        pm_config["rportfolio"].update({"symbols": pm_config["symbols"],
                                        "market": pm_config["market"]})

        strat_config = pm_config["strategy"]
        api_config = pm_config["api"]

        today = datetime.now()
        end_date = today + timedelta(days=365)

        # Get rebalance dates (one hour after market open) for the next year
        self.rebal_date: List[datetime] = market_settigs_date(pm_config["market"], today, end_date)

        # Fetch historical data
        self.data: _Data = Data(pm_config["data"]).fetch_data(
            today - timedelta(days=pm_config["hist_dataLen"]), today)

        # Fit and forecast using the model
        self.model: _Model = Model(pm_config["model"])

        # Initialize strategy
        self.strategy: _Strategies = Strategies(strat_config)

        # Initialize remote portfolio and refresh it
        self.rportfolio: RemotePortfolio = RemotePortfolio(API(api_config), pm_config["rportfolio"])
        self.rportfolio.refresh_portfolio()
        position = self.rportfolio.weights()

        # Initialize local portfolio with the refreshed positions
        self.portfolio: Portfolio = (Portfolio(
            pm_config["portfolio"],
            position["capital"],
            position["weights"],
            get_last_trading_day(position["date"], pm_config["market"])).
                                     update_metrics(self.data))
        # Initialize pending orders list
        self.pending_orders: Dict[str, List[Dict[str, Union[str, float]]]] = {}

    def update_portfolio(self, to_calib: bool = False):
        """
        Updates the portfolio weights with the complete logic to manage existing orders and place new orders.

        """

        max_attempts = 5
        attempts = 0

        while self.rportfolio.open_orders and attempts < max_attempts:
            time.sleep(60)
            self.rportfolio.refresh_portfolio()
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
        weights = [capital_weights["weight"][asset]
                   for asset in
                   ["cash"] + self.portfolio.pf_config["symbols"]]

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
        for asset, target_value in target_values.items():
            target_qty = target_values.get(asset, 0) / current_prices[asset]
            difference = target_qty - float(current_positions.get(asset, 0))
            trad_qty = trunc_decimal(abs(difference), 8)  # Troncature à 8 décimales
            trad_notional = trunc_decimal(abs(difference)
                                          * current_prices[asset], 2)  # Troncature à 2 décimales
            if difference > 0:
                # Pour acheter, on utilise la valeur notionnelle
                buy_orders.append({"asset": asset,
                                   "action": "buy",
                                   "units": trad_qty,
                                   "value": trad_notional,
                                   "type": "notional"})
            elif difference < 0 and trad_qty < current_positions.get(asset, 0):
                # Pour vendre, on utilise les unités
                sell_orders.append({"asset": asset,
                                    "action": "sell",
                                    "units": trad_qty,
                                    "value": trad_notional,
                                    "type": "qty"})

        # Trier les ordres par valeur décroissante
        buy_orders.sort(key=lambda x: x["value"], reverse=True)
        sell_orders.sort(key=lambda x: x["value"], reverse=True)

        self.pending_orders = {"sell": sell_orders, "buy": buy_orders}

    def execute_orders(self):
        """
        Execute pending orders in order of their value and remove them from the list if successful.
        """

        # Exécuter les ordres de vente
        for order in self.pending_orders["sell"][:]:
            result = self.rportfolio.broker_api.place_orders([order])
            if result[0]["success"]:
                self.pending_orders["sell"].remove(order)

        # Exécuter les ordres d"achat
        for order in self.pending_orders["buy"][:]:
            current_cash = self.rportfolio.cash
            if order["value"] > current_cash:
                continue  # Skip this order if not enough cash is available

            result = self.rportfolio.broker_api.place_orders([order])
            if result[0]["success"]:
                self.pending_orders["buy"].remove(order)
                current_cash -= order["value"]  # Update available cash after a buy order

    def start(self):
        calib_done = False
        while True:
            to_calib, to_update, self.rebal_date = Check_and_update_Date(self.rebal_date)

            if to_calib and not calib_done:
                print("Période de calibrage")
                self.update_portfolio(to_calib)
                calib_done = True
            if to_update:
                print("Période de rééquilibrage")
                self.execute_orders()
                calib_done = False

            else:
                print("Non rééquilibrage")
            time.sleep(900)

    def run(self):
        thread = threading.Thread(target=self.start)
        thread.daemon = True
        thread.start()
