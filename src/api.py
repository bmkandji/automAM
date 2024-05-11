from abc import ABC
import alpaca_trade_api as tradeapi
from src.abstract import _BrokerAPI


class AlpacaBrokerAPI(_BrokerAPI, ABC):
    def __init__(self, api_config: dict):
        self.api = tradeapi.REST(api_config["api_key"], api_config["api_secret"],
                                 api_config["base_url"], api_config["api_version"])

    def get_current_positions(self):
        return {pos.symbol: pos.qty for pos in self.api.list_positions()}

    def get_open_orders(self):
        return self.api.list_orders(status='open')

    def get_available_cash(self):
        account = self.api.get_account()
        return float(account.cash)

    def get_current_prices(self, assets):
        prices = {}
        for asset in assets:
            barset = self.api.get_barset(asset, 'minute', 1)
            prices[asset] = barset[asset][0].c  # Dernier prix de clôture
        return prices

    def place_order(self, asset, action, quantity):
        self.api.submit_order(
            symbol=asset,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )
