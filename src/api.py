import alpaca_trade_api as tradeapi
from abc import ABC
from typing import List, Dict, Union
from src.abstract import _BrokerAPI
import time
from src.common import get_current_time


class API(_BrokerAPI, ABC):
    def __init__(self, api_config: Dict[str, str]):
        """
        Initialize the Alpaca API client with user credentials.

        :param api_config: A dictionary containing API key, secret, base URL, and API version.
        """
        # Store API keys separately
        self._api_key = api_config["api_key"]
        self._api_secret = api_config["api_secret"]

        # Initialize the internal _api attribute
        self._api = tradeapi.REST(
            self._api_key,
            self._api_secret,
            api_config["base_url"],
            api_config.get("api_version", "v2")  # Use 'v2' as default API version if not provided
        )

    @property
    def api(self) -> tradeapi.REST:
        """
        Property method to access the Alpaca trading API client.

        :return: The Alpaca trading API client.
        """
        return self._api

    def get_current_positions(self, assets: List[str] = None) -> Dict[str, str]:
        """
        Retrieves current open positions from the Alpaca account, optionally filtering by a list of assets.

        :param assets: Optional list of asset symbols to filter positions.
        :return: A dictionary of current positions with symbols as keys and quantities as values.
        """
        # Fetch all positions
        all_positions = self.api.list_positions()
        positions = {asset: 0 for asset in assets}
        # Filter positions if a list of assets is provided
        if assets:
            for pos in all_positions:
                if pos.symbol in assets:
                    positions[pos.symbol] = pos.qty
        else:
            positions = {pos.symbol: pos.qty for pos in all_positions}

        return positions

    def get_open_orders(self, assets: List[str] = None) -> List[tradeapi.entity.Order]:
        """
        Fetches all open orders from the Alpaca account, optionally filtering by a list of assets.

        :param assets: Optional list of asset symbols to filter orders.
        :return: A list of open orders.
        """
        # Fetch all open orders
        all_open_orders = self.api.list_orders(status="open")

        # Filter orders if a list of assets is provided
        if assets:
            open_orders = [order for order in all_open_orders if order.symbol in assets]
        else:
            open_orders = all_open_orders

        return open_orders

    def get_available_cash(self) -> float:
        """
        Gets the available cash balance in the Alpaca trading account.

        :return: Floating point number representing the available cash.
        """
        # Retrieve the account information and extract the cash available
        account = self.api.get_account()
        return float(account.cash)

    def get_current_prices(self, assets: List[str], retries: int = 3, delay: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the latest current prices and their trade dates for a specified list of asset symbols using a single API call.
        Retries if not all prices are obtained.

        :param assets: A list of asset symbols (e.g., ["AAPL", "GOOGL"]).
        :param retries: Number of retries if prices are missing.
        :param delay: Delay in seconds between retries.
        :return: A dictionary with a date key and a dictionary of asset prices.
        """
        prices = {asset: None for asset in assets}
        set_date = False

        for attempt in range(retries):
            for asset in assets:
                if prices[asset] is None:  # Only retry for assets that haven't got a price yet
                    try:
                        trade = self.api.get_latest_trade(asset)
                        if trade:
                            prices[asset] = trade.price
                            set_date = True  # Update date to the latest trade timestamp
                    except Exception:
                        pass  # Ignore exceptions and retry

            missing_prices = [asset for asset, price in prices.items() if price is None]
            if not missing_prices:  # All prices obtained
                break
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
        prices["date"] = get_current_time() if set_date else None
        return prices

    def get_total_portfolio_value(self, assets: List[str] = None) -> float:
        """
        Calculates the total value of the portfolio, including current positions and available cash,
        optionally filtered by a list of assets.

        :param assets: Optional list of asset symbols to filter positions.
        :return: Total portfolio value.
        """
        current_positions = self.get_current_positions(assets)
        current_prices = self.get_current_prices(list(current_positions.keys()))
        current_cash = self.get_available_cash()

        total_portfolio_value = sum(
            float(current_positions[asset]) * current_prices[asset] for asset in current_positions) + current_cash

        return total_portfolio_value

    def place_orders(self, orders: List[Dict[str, Union[str, float]]]) -> List[str]:
        """
        Places multiple market orders through the Alpaca API based on weights relative to the total portfolio value.

        :param orders: A list of orders, each represented as a dictionary with "asset", "action", and "weight".
        :return: A list of messages indicating the result of each order placement.

        Args:
            **kwargs:
        """
        results = []

        for order in orders:

            try:
                if order["type"] == "notional":
                    self.api.submit_order(
                        symbol=order["asset"],
                        notional=order["value"],
                        side=order["action"],
                        type="market",
                        time_in_force="day"
                    )
                    results.append({"success": True,
                                    "messages": f"Order to {order['action']} {order['value']}  shares of {order['asset']} in {order['type']} placed successfully."})
                elif order["type"] == "qty":
                    self.api.submit_order(
                        symbol=order["asset"],
                        qty=order["units"],
                        side=order["action"],
                        type="market",
                        time_in_force="day"
                    )
                    results.append({"success": True,
                                    "messages": f"Order to {order['action']} {order['units']}  shares of {order['asset']} in {order['type']} placed successfully."})
                else:
                    raise ValueError(f"Invalid order type: {order['type']}")

            except Exception as e:
                results.append({"success": False,
                                "messages": f"Failed to place order to {order['action']} {order['asset']} . Error: {str(e)}"})

        return results

    def cancel_order(self, order_id: str) -> str:
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

    def cancel_all_open_orders(self, assets: List[str] = None) -> List[str]:
        """
        Attempts to cancel all open orders, optionally filtered by a list of assets, retrying if some are not initially canceled.

        :param assets: Optional list of asset symbols to filter orders.
        :return: A list of messages indicating the result of each cancellation attempt.
        """
        attempts = 3  # Number of attempts to cancel each order
        delay = 2  # Delay in seconds before retrying

        # Retrieve all open orders
        all_open_orders = self.api.list_orders(status="open")

        # Filter orders if a list of assets is provided
        if assets:
            open_orders = [order for order in all_open_orders if order.symbol in assets]
        else:
            open_orders = all_open_orders

        cancellation_results = []

        while attempts > 0 and open_orders:
            current_orders = open_orders[:]
            open_orders = []

            for order in current_orders:
                try:
                    # Attempt to cancel the order
                    self.api.cancel_order(order.id)
                    # Append a success message for each cancellation
                    cancellation_results.append(f"Order {order.id} has been cancelled successfully.")
                except Exception as e:
                    # Append an error message if cancellation fails
                    cancellation_results.append(f"Failed to cancel order {order.id}. Error: {str(e)}")
                    open_orders.append(order)  # Add order to list for retry

            attempts -= 1  # Decrement the number of remaining attempts
            if open_orders and attempts > 0:
                time.sleep(delay)  # Wait for a few seconds before retrying

        # After all attempts, if there are still open orders that couldn't be canceled
        if open_orders:
            for order in open_orders:
                cancellation_results.append(f"Order {order.id} could not be cancelled after multiple attempts.")

        return cancellation_results

"""
########### TEST API ##############
from utils.load import load_json_config
api_config = load_json_config(r"api_settings/api_settings.json")
alpaca_api = API(api_config)
print(alpaca_api.get_available_cash())
print(alpaca_api.get_open_orders(["AAPL", "AMZN", "GOOGL"]))
print(alpaca_api.get_current_positions(["AAPL", "AMZN", "GOOGL"]))
print(alpaca_api.get_current_prices(["AAPL", "AMZN", "GOOGL"]))
print(alpaca_api.get_total_portfolio_value(["AAPL", "AMZN", "GOOGL"]))
#print(alpaca_api.place_orders([{"asset": "AMZN", "units": 0.1, "action": "sell", "type": "qty"}]))
#print(alpaca_api.place_orders([{"asset": "GOOGL", "units": 0.000001, "action": "buy", "type": "qty"}]))
#print(alpaca_api.cancel_all_open_orders())
"""
