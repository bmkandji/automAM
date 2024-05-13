from src.abstract import _BrokerAPI, _Model, _Data, _Strategies
from typing import Dict, List, Union


class PortfolioManager:
    def __init__(self, broker_api: _BrokerAPI, model: _Model, strategy: _Strategies):
        """
        Initializes the portfolio manager with a brokerage API.

        :param broker_api: An instance of the brokerage API to interact with the trading account.
        :param model: An instance of the model to use for portfolio management.
        :param strategy: An instance of the strategy to use for portfolio management.
        """
        self.broker_api = broker_api
        self.model = model
        self.strategy = strategy

    def update_portfolio_weights(self, target_weights: Dict[str, float]):
        """
        Updates the portfolio weights with the complete logic to manage existing orders and place new orders.

        :param target_weights: A dictionary with asset symbols as keys and target weights as values.
        """
        # Retrieve necessary information
        current_positions = self.broker_api.get_current_positions()
        current_orders = self.broker_api.get_open_orders()
        current_prices = self.broker_api.get_current_prices(list(target_weights.keys()))
        current_cash = self.broker_api.get_available_cash()

        # Calculate the total portfolio value including cash
        total_portfolio_value = sum(
            float(current_positions[asset]) * current_prices[asset] for asset in current_positions) + current_cash

        # Determine target values for each asset based on target weights
        target_values = {asset: total_portfolio_value * weight for asset, weight in target_weights.items()}

        # Cancel all open orders that are no longer needed or adjust existing orders
        self.cancel_or_adjust_orders(current_orders, target_values, current_prices, total_portfolio_value)

        # Calculate the quantities needed to reach target weights and place new orders
        orders_to_place = []
        for asset, target_value in target_values.items():
            current_value = float(current_positions.get(asset, 0)) * current_prices[asset]
            difference = target_value - current_value
            weight_to_trade = abs(difference) / total_portfolio_value
            if difference > 0:
                orders_to_place.append({'asset': asset, 'action': 'buy', 'weight': weight_to_trade})
            elif difference < 0:
                orders_to_place.append({'asset': asset, 'action': 'sell', 'weight': weight_to_trade})

        self.broker_api.place_orders(orders_to_place)

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
                self.broker_api.cancel_order(order['id'])
