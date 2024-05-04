from utils.portfolio_tools import mean_variance_portfolio  # Importe la fonction d'algorithme d'optimisation
from src.portfolio import Portfolio  # Importe la classe Position depuis les paramètres d'outils
from src.abstract import _Strategies


class Strategies(_Strategies):
    def __init__(self, strat_config: dict):
        """
        Initialise une instance de la classe Strategies avec les paramètres donnés.

        Paramètres :
        mean_var (dict) : Dictionnaire contenant les données sur les moyennes et variances.
        strat_config (str) : Chemin vers le fichier de configuration de la stratégie.
        position (Position) : Objet de position initial.

        Cette méthode initialise la classe en chargeant la configuration de la stratégie à partir d'un fichier,
        et en initialisant les données sur les moyennes et variances ainsi que l'objet de position.
        """
        super().__init__(strat_config)

    def fit(self, portfolio: Portfolio):
        """
        Exécute l'algorithme d'optimisation pour ajuster la position.

        Cette méthode utilise les données sur les moyennes et variances ainsi que la configuration de la stratégie
        pour exécuter un algorithme d'optimisation et ajuster la position en conséquence.
        """
        # Prépare les arguments nécessaires pour l'algorithme d'optimisation
        arg = [portfolio.metrics["mean"], portfolio.metrics["covariance"],
               self.strat_config["aversion"], self.strat_config["fee_rate"], self.strat_config["bounds"],
               portfolio.weights, portfolio.capital, portfolio.metrics["scale"]]
        # Exécute l'algorithme d'optimisation pour ajuster la position
        portfolio.update_opti_nweights(mean_variance_portfolio(*arg))
