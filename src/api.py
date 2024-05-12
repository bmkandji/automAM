import alpaca_trade_api as tradeapi
from abc import ABC
from src.abstract import _BrokerAPI


class AlpacaBrokerAPI(_BrokerAPI, ABC):
    def __init__(self, api_config: dict):
        """
        Initialize the Alpaca API client with user credentials.

        :param api_config: A dictionary containing API key, secret, base URL, and API version.
        """
        # Initialize the Alpaca trading API client
        self.api = tradeapi.REST(
            api_config["api_key"],
            api_config["api_secret"],
            api_config["base_url"],
            api_config["api_version"]
        )

    def get_current_positions(self):
        """
        Retrieves all current open positions from the Alpaca account.

        :return: A dictionary of current positions with symbols as keys and quantities as values.
        """
        # Fetch and return the current open positions in a dictionary format
        return {pos.symbol: pos.qty for pos in self.api.list_positions()}

    def get_open_orders(self):
        """
        Fetches all open orders from the Alpaca account.

        :return: A list of open orders.
        """
        # Return a list of all open orders
        return self.api.list_orders(status='open')

    def get_available_cash(self):
        """
        Gets the available cash balance in the Alpaca trading account.

        :return: Floating point number representing the available cash.
        """
        # Retrieve the account information and extract the cash available
        account = self.api.get_account()
        return float(account.cash)

    def get_current_prices(self, assets):
        """
        Retrieves the latest closing prices for a specified list of asset symbols using a single API call.

        :param assets: A list of asset symbols (e.g., ['AAPL', 'GOOGL']).
        :return: A dictionary mapping each symbol to its current closing price.
        """
        # Fetch the latest minute bar data for all specified assets
        barset = self.api.get_barset(assets, 'minute', 1)
        # Extract the last closing price safely checking if data is available
        prices = {asset: barset[asset][0].c for asset in assets if barset[asset]}
        return prices

    def place_order(self, asset, action, quantity):
        """
        Places a market order through the Alpaca API.

        :param asset: The symbol for the asset to trade (e.g., 'AAPL').
        :param action: 'buy' or 'sell'.
        :param quantity: The number of shares to buy or sell.
        """
        # Submit a market order to buy or sell the specified quantity of the asset
        self.api.submit_order(
            symbol=asset,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )

    def cancel_order(self, order_id):
        """
        Attempts to cancel an order with the given ID.

        :param order_id: The ID of the order to cancel.
        :return: A message indicating success or failure of the cancellation.
        """
        try:
            # Attempt to cancel the order and return a success message
            self.api.cancel_order(order_id)
            return f"Order {order_id} has been cancelled successfully."
        except Exception as e:
            # Return an error message if cancellation fails
            return f"Failed to cancel order {order_id}. Error: {str(e)}"
