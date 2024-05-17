from src.abstract import _Model, _Data, _Strategies
from src.local_portfolio import Portfolio
from src.remote_portfolio import RemotePortfolio
from src.models import Model
from src.data_mn import Data
from src.strategies import Strategies
from typing import Dict, List, Union
from datetime import datetime
from api import API
from src.common import get_last_trading_day

class PortfolioManager:
    def __init__(self, pm_config):
        """
        Initializes the portfolio manager with a brokerage API.

        :param pm_config: A configuration dictionary for the portfolio manager.
        """
        pm_config["data"].update({"symbols": pm_config["symbols"]})
        pm_config["model"].update({"symbols": pm_config["symbols"]})
        pm_config["portfolio"].update({"symbols": pm_config["symbols"], "market": pm_config["market"]})
        pm_config["rportfolio"].update({"symbols": pm_config["symbols"], "market": pm_config["market"]})
        strat_config = pm_config["strategy"]
        api_config = pm_config["api"]

        self.data: _Data = Data(pm_config["data"])
        self.model: _Model = Model(pm_config["model"])
        self.strategy: _Strategies = Strategies(strat_config)
        self.rportfolio: RemotePortfolio = RemotePortfolio(API(api_config), pm_config["rportfolio"])

        self.rportfolio.refresh_portfolio()
        position = self.rportfolio.weights()
        self.portfolio: Portfolio = Portfolio(pm_config["portfolio"],
                                              position["capital"],
                                              position["weights"],
                                              get_last_trading_day(position["date"], pm_config["market"]))

        self.pending_orders: List[Dict[str, Union[str, float]]] = []
        self.rebal_date: List[datetime] = []

    def update_portfolio_weights(self, target_weights: Dict[str, float]):
        """
        Updates the portfolio weights with the complete logic to manage existing orders and place new orders.

        :param target_weights: A dictionary with asset symbols as keys and target weights as values.
        """
        current_positions = self.rportfolio.positions
        current_orders = self.rportfolio.open_orders
        current_prices = self.rportfolio.broker_api.get_current_prices(list(target_weights.keys()))
        current_cash = self.rportfolio.cash

        total_portfolio_value = sum(
            float(current_positions[asset]) * current_prices[asset] for asset in current_positions) + current_cash

        target_values = {asset: total_portfolio_value * weight for asset, weight in target_weights.items()}

        self.cancel_or_adjust_orders(current_orders, target_values, current_prices, total_portfolio_value)

        orders_to_place = []
        for asset, target_value in target_values.items():
            current_value = float(current_positions.get(asset, 0)) * current_prices[asset]
            difference = target_value - current_value
            weight_to_trade = abs(difference) / total_portfolio_value
            if difference > 0:
                orders_to_place.append(
                    {'asset': asset, 'action': 'buy', 'weight': weight_to_trade, 'value': abs(difference)})
            elif difference < 0:
                orders_to_place.append(
                    {'asset': asset, 'action': 'sell', 'weight': weight_to_trade, 'value': abs(difference)})

        orders_to_place.sort(key=lambda x: x['value'])

        self.pending_orders.extend(orders_to_place)

    def cancel_or_adjust_orders(self, current_orders: List[Dict[str, Union[str, float]]],
                                target_values: Dict[str, float], current_prices: Dict[str, float],
                                total_portfolio_value: float):
        """
        Cancels or adjusts existing orders based on target values.

        :param current_orders: A list of current open orders.
        :param target_values: A dictionary with asset symbols as keys and target values as values.
        :param current_prices: A dictionary with asset symbols as keys and their current prices as values.
        :param total_portfolio_value: The total value of the portfolio including cash.
        """
        for order in current_orders:
            asset = order['asset']
            order_type = order['type']
            order_weight = float(order['weight'])
            order_value = order_weight * total_portfolio_value
            target_value = target_values[asset]

            if (order_type == 'buy' and order_value > target_value) or (
                    order_type == 'sell' and order_value < target_value):
                self.rportfolio.broker_api.cancel_order(order['id'])

    def execute_pending_orders(self):
        """
        Execute pending orders in order of their value and remove them from the list if successful.
        """
        current_cash = self.rportfolio.cash

        for order in self.pending_orders[:]:
            order_value = order['value']
            if order['action'] == 'buy' and order_value > current_cash:
                continue  # Skip this order if not enough cash is available

            success = self.rportfolio.broker_api.place_order(order['asset'], order['action'], order['weight'])
            if success:
                self.pending_orders.remove(order)
                if order['action'] == 'buy':
                    current_cash -= order_value  # Update available cash after a buy order

    def add_pending_order(self, order: Dict[str, Union[str, float]]):
        """
        Adds a new order to the pending orders list.

        :param order: A dictionary representing the order to be added.
        """
        self.pending_orders.append(order)
        self.pending_orders.sort(key=lambda x: x['value'])  # Keep orders sorted by value
