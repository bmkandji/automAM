from src.abstract import _BrokerAPI, _Model, _Data, _Strategies
from src.local_portfolio import Position, Portfolio


class PortfolioManager:
    def __init__(self, broker_api: _BrokerAPI, model: _Model, strategie: _Strategies):
        """
        Initialise le gestionnaire de portefeuille avec une API de courtage.
        """
        self.broker_api = broker_api

    def update_portfolio_weights(self, target_weights):
        """
        Met à jour les poids du portefeuille avec la logique complète pour gérer les ordres existants et placer de nouveaux ordres.
        """
        # Récupérer les informations nécessaires
        current_positions = self.broker_api.get_current_positions()
        current_orders = self.broker_api.get_open_orders()
        current_prices = self.broker_api.get_current_prices(list(target_weights.keys()))
        current_cash = self.broker_api.get_available_cash()

        # Calculer la valeur totale du portefeuille incluant le cash
        total_portfolio_value = sum(
            current_positions[asset] * current_prices[asset] for asset in current_positions) + current_cash

        # Déterminer les valeurs cibles pour chaque actif basées sur les poids cibles
        target_values = {asset: total_portfolio_value * weight for asset, weight in target_weights.items()}

        # Annuler tous les ordres ouverts qui ne sont plus nécessaires ou ajuster les ordres existants
        self.cancel_or_adjust_orders(current_orders, target_values, current_prices)

        # Calculer les quantités nécessaires pour atteindre les poids cibles et passer de nouveaux ordres
        for asset, target_value in target_values.items():
            if asset in current_positions:
                current_value = current_positions[asset] * current_prices[asset]
                difference = target_value - current_value
                quantity_to_trade = abs(difference) / current_prices[asset]
                if difference > 0:
                    self.broker_api.place_order(asset, 'BUY', quantity_to_trade)
                elif difference < 0:
                    self.broker_api.place_order(asset, 'SELL', quantity_to_trade)

    def cancel_or_adjust_orders(self, current_orders, target_values, current_prices):
        """
        Annule ou ajuste les ordres existants basés sur les valeurs cibles.
        """
        for order in current_orders:
            asset = order['asset']
            order_type = order['type']
            order_quantity = order['quantity']
            current_value = order_quantity * current_prices[asset]
            target_value = target_values[asset]

            if (order_type == 'BUY' and current_value > target_value) or (
                    order_type == 'SELL' and current_value < target_value):
                self.broker_api.cancel_order(order['id'])

    def calculate_optimal_weights(self):
        """
        Méthode fictive pour calculer les poids optimaux, à implémenter selon la stratégie.
        """
        # Logique pour calculer et retourner un dictionnaire de poids optimaux
        return {}

    def rebalance_portfolio(self):
        """
        Déclenche un rééquilibrage complet du portefeuille.
        """
        optimal_weights = self.calculate_optimal_weights()
        self.update_portfolio_weights(optimal_weights)
