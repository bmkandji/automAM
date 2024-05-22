from src.abstract import _BrokerAPI
from typing import Dict, Any, List
import alpaca_trade_api as tradeapi


class RemotePortfolio:
    def __init__(self, broker_api: _BrokerAPI, pf_config: dict):
        """
        Initializes the remote portfolio with a reference to the broker's API.

        :param broker_api: An instance of the broker's API to query and act on the trading account.
        """
        self._pf_config = pf_config
        self._broker_api = broker_api  # Stores the broker API instance

    @property
    def pf_config(self) -> Dict[str, Any]:
        """Returns the portfolio configuration dictionary."""
        return self._pf_config

    @property
    def broker_api(self) -> _BrokerAPI:
        """
        Returns the broker API instance.

        :return: The broker API instance.
        """
        return self._broker_api

    @property
    def positions(self) -> Dict[str, Any]:
        """
        Retrieves the current positions in the portfolio.

        :return: A dictionary of current positions.
        """
        return self._broker_api.get_current_positions(self._pf_config["symbols"])

    @property
    def open_orders(self) -> List[tradeapi.entity.Order]:
        """
        Retrieves all open orders.

        :return: The list of open orders.
        """
        return self._broker_api.get_open_orders(self._pf_config["symbols"])

    @property
    def cash(self) -> float:
        """
        Retrieves the amount of available cash in the portfolio.

        :return: The amount of available cash.
        """
        return self._broker_api.get_available_cash()

    def open_orders_Byside(self) -> dict[List[tradeapi.entity.Order]]:
        """
        Fetches all open orders from the Alpaca account and separates them into buy and sell orders.

        :return: A dictionary with 'buy' and 'sell' as keys and lists of open orders as values.
        """
        # Fetch all open orders
        open_orders = self.open_orders
        sell_open_orders = [order for order in open_orders if order.side == 'sell']
        buy_open_order = [order for order in open_orders if order.side == 'buy']

        return {"sell_open_orders": sell_open_orders, "buy_open_order": buy_open_order}

    def weights(self) -> Dict[str, Any]:
        """
        Calculates the weights of each asset, including cash, in the total capital after canceling all open orders.

        :return: A dictionary with symbols as keys and their respective weights in the total portfolio as values.
        """
        # Cancel all open orders first to stabilize the portfolio
        # Retrieve current positions and cash balance
        positions = self._broker_api.get_current_positions(self._pf_config["symbols"])
        cash = self.cash

        # If positions are empty and only cash is available
        if not positions:
            return {'cash': 1.0}  # 100% cash

        # Retrieve current prices for the assets
        assets = list(positions.keys())
        prices = self._broker_api.get_current_prices(assets)

        # Calculate total capital: sum of (price * quantity) for all assets + cash
        total_capital = cash + sum(prices[symbol] * float(quantity) for symbol, quantity in positions.items())

        # Calculate the weight of each asset including cash
        portfolio_weights = {symbol: (prices[symbol] * float(quantity)) / total_capital for
                             symbol, quantity in positions.items()}
        # Ajouter le poids du cash
        portfolio_weights["cash"] = cash / total_capital
        return {"date": prices["date"], "capital": total_capital, "weights": portfolio_weights}

    def cancel_all_open_orders(self) -> None:
        """
        Updates the portfolio information by retrieving the latest data from the broker's API.
        """
        self._broker_api.cancel_all_open_orders(self._pf_config["symbols"])

"""
########### TEST API ##############
from utils.load import load_json_config
from src.api import API
api_config = load_json_config(r'api_settings/api_settings.json')
rpf_config = load_json_config(r'portfolio_settings/rpf_settings.json')
alpaca_api = API(api_config)
rPortfolio = RemotePortfolio(alpaca_api, rpf_config)
rPortfolio.cancel_all_open_orders()
print(rPortfolio.weights())
"""
