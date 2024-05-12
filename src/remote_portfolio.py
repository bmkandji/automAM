from src.abstract import _BrokerAPI


class RemotePortfolio:
    def __init__(self, broker_api: _BrokerAPI):
        """
        Initialise le portefeuille distant avec une référence à l'API du broker.
        :param broker_api: Une instance de l'API du broker pour interroger et agir sur le compte de trading.
        """
        self.broker_api = broker_api
        self.positions = self.fetch_positions()
        self.open_orders = self.fetch_open_orders()
        self.cash = self.fetch_available_cash()
        self.position = self.position()

    def fetch_positions(self):
        """
        Récupère les positions actuelles du portefeuille à partir de l'API du broker.
        """
        return self.broker_api.get_current_positions()

    def fetch_open_orders(self):
        """
        Récupère tous les ordres ouverts à partir de l'API du broker.
        """
        return self.broker_api.get_open_orders()

    def fetch_available_cash(self):
        """
        Récupère la quantité de cash disponible dans le portefeuille.
        """
        return self.broker_api.get_available_cash()

    def position(self):
        """
        Calculates the weights of each asset, including cash, in the total capital after cancelling all open orders.

        :return: A dictionary with symbols as keys and their respective weights in the total portfolio as values.
        """
        # Cancel all open orders first to stabilize the portfolio
        self.broker_api.cancel_all_open_orders()

        # Retrieve current positions and cash balance
        positions = self.broker_api.get_current_positions()
        cash = self.broker_api.get_available_cash()

        # If positions are empty and only cash is available
        if not positions:
            return {'cash': 1.0}  # 100% cash

        # Retrieve current prices for the assets
        assets = list(positions.keys())
        prices = self.broker_api.get_current_prices(assets)

        # Calculate total capital: sum of (price * quantity) for all assets + cash
        total_capital = cash + sum(prices[symbol] * float(quantity) for symbol, quantity in positions.items())

        # Calculate the weight of each asset including cash
        portfolio_weights = {'cash': cash / total_capital}
        for symbol, quantity in positions.items():
            asset_value = prices[symbol] * float(quantity)
            portfolio_weights[symbol] = asset_value / total_capital

        current_prices = self.broker_api.get_current_prices(self.positions.keys())
        capital = (sum(self.positions[asset] * price for asset, price in current_prices.items())
                           + self.cash)
        return {"capital": capital , "weights": portfolio_weights}

    def refresh_portfolio(self):
        """
        Met à jour les informations du portefeuille en récupérant les dernières données.
        """
        self.positions = self.fetch_positions()
        self.open_orders = self.fetch_open_orders()
        self.cash = self.fetch_available_cash()
        self.position = self.position()
