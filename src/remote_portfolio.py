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
        self.total_value = self.calculate_total_value()

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

    def calculate_total_value(self):
        """
        Calcule la valeur totale du portefeuille, incluant le cash et la valeur des positions.
        """
        current_prices = self.broker_api.get_current_prices(self.positions.keys())
        positions_value = sum(self.positions[asset] * price for asset, price in current_prices.items())
        return positions_value + self.cash

    def refresh_portfolio(self):
        """
        Met à jour les informations du portefeuille en récupérant les dernières données.
        """
        self.positions = self.fetch_positions()
        self.open_orders = self.fetch_open_orders()
        self.cash = self.fetch_available_cash()
        self.total_value = self.calculate_total_value()

    def place_order(self, asset, action, quantity):
        """
        Place un ordre sur le marché via l'API du broker.
        :param asset: L'actif pour lequel l'ordre doit être placé.
        :param action: 'BUY' ou 'SELL'.
        :param quantity: La quantité à acheter ou vendre.
        """
        self.broker_api.place_order(asset, action, quantity)
        self.refresh_portfolio()  # Refresh après placement d'ordre pour actualiser le portefeuille.

    def cancel_order(self, order_id):
        """
        Annule un ordre spécifique.
        :param order_id: L'identifiant de l'ordre à annuler.
        """
        self.broker_api.cancel_order(order_id)
        self.refresh_portfolio()  # Refresh après annulation pour actualiser le portefeuille.
