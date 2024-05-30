from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
from datetime import datetime
from src.common import get_current_time
from utils.check import check_configs  # Importation des utilitaires nécessaires
import pandas as pd
import numpy as np


# Définition de la classe abstraite _Model
class _Model(ABC):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        # Initialisation avec une configuration du modèle, stockée dans un dictionnaire
        self._model_config: Dict[str, Any] = model_config
        self._metrics: Dict[str, Any] = {
            "fit_date": None,
            "scale": None,
            "to_update": True}  # Dictionnaire pour stocker les métriques associées au modèle

    @property
    def model_config(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture à la configuration du modèle
        return self._model_config

    @property
    def metrics(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture aux métriques du modèle
        return self._metrics

    def check_fit(self, data: _Data, horizon: datetime):
        """
        Méthode abstraite pour ajuster le modèle à des données et prédire jusqu'à un certain horizon.
        Doit être implémentée par des sous-classes spécifiques.
        """
        check_configs(data=data, model=self, check_date=False)

        if horizon <= data.data_config["end_date"]:
            raise ValueError("Please give a horizon posterior to the end date of the data")

        if self.metrics["fit_date"]:

            check_update = (data.data_config["end_date"] - self.metrics["fit_date"]).days
            if check_update <= 0 or self._metrics['scale'] != data.data_config["scale"]:
                raise ValueError("Please use recent data or same scale.")

            self._metrics["to_update"] = check_update >= self.model_config["model_config"]["recalib"]

        pass

    @abstractmethod
    def fit_fcast(self, data: _Data, horizon: datetime):
        pass


# Définition de la classe abstraite _Data
class _Data(ABC):
    def __init__(self, data_config: Dict[str, Any]) -> None:
        # Initialisation avec une configuration des données, stockée dans un dictionnaire
        self._data_config: Dict[str, Any] = data_config
        self._data: pd.DataFrame = pd.DataFrame()
        self._metrics: Dict[str, Any] = {}  # Dictionnaire pour stocker les métriques associées aux données

    @property
    def data_config(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture à la configuration des données
        return self._data_config

    @property
    def data(self) -> pd.DataFrame:
        # Propriété pour accéder en lecture à la configuration des données
        key, value = next(iter(self.data_config["cash"].items()))
        data = self._data.copy()
        data.insert(0, key, value)
        return data


    @property
    def metrics(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture aux métriques des données
        return self._metrics

    @abstractmethod
    def fetch_data(self, start_date: datetime, end_date: datetime) -> int:
        """
        Méthode abstraite pour récupérer des données entre deux dates.
        Doit être implémentée par des sous-classes pour fonctionner.
        """
        pass

    def update_data(self, new_end_date: datetime = None):
        """
        Méthode abstraite pour mettre à jour les données jusqu'à une nouvelle date de fin.
        La date de fin est optionnelle; si non spécifiée, peut-être mise à jour jusqu'à la date courante.
        """

        if new_end_date is None:
            new_end_date = get_current_time().replace(tzinfo=None)

        if self.data.empty:
            raise ValueError("please fetch before update")

        if self.data_config["end_date"] >= new_end_date:
            raise ValueError("The please provide recent date for update")

        start_date_update = self._data.index.max()
        new_rows_added = self.fetch_data(start_date_update, new_end_date)

        # Remove the same number of oldest rows as new rows added
        if len(self._data) > new_rows_added:
            self._data = self._data.iloc[new_rows_added:]  # Keeps the DataFrame size consistent
            self._data_config["start_date"] = self._data.index.min()

        self._metrics = {}

    def update_metrics(self, model: _Model) -> None:
        """
        Met à jour les métriques en fonction des résultats et configurations du modèle fourni.
        Utilise les configurations du modèle pour enrichir les métriques des données.
        """
        check_configs(data=self, model=model, check_scale=True)  # Validation des configurations
        self._metrics = {
            "model": model.model_config['model_config'],  # à ajuster en cas de plusieur types de model
            **model.metrics  # Intégration des métriques du modèle aux métriques des données
        }

    def window_returns(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """
        Filter the instance's DataFrame based on a date range,
         exclusive of the start date and inclusive of the end date,
        and return an array containing the sum of each column in the filtered DataFrame.

        Args:
        - start_date (datetime): The start date, exclusive.
        - end_date (datetime): The end date, inclusive.

        Returns:
        - np.ndarray: An array containing the sum of each column from the filtered DataFrame.

        Raises:
        - ValueError: If no data is available for the given date range.
        """
        # Ensure the index is in datetime format and filter the DataFrame
        self._data.index = pd.to_datetime(self._data.index)
        filtered_df = self.data.loc[(self._data.index > start_date) & (self._data.index <= end_date)]

        # Check if the filtered DataFrame is empty
        if filtered_df.empty:
            raise ValueError(f"No data available from {start_date} to {end_date}.")

        return filtered_df.sum()

    def replace_NA(self, window: int = 5) -> None:
        """
        Replaces NA values in the DataFrame with the rolling mean calculated over a specified window size.

        Parameters:
        window (int): The size of the rolling window to calculate the means, default is 5.
        """
        # Calculate the rolling mean with a specified window, minimum number of observations in the window required
        # to have a value is 1
        roll_means = self._data.rolling(window=window, min_periods=1, center=True).mean()

        # Replace NA values in the DataFrame with the calculated rolling means
        self._data.fillna(roll_means, inplace=True)


# Définition de la classe abstraite _Strategies
class _Strategies(ABC):
    def __init__(self, strat_config: Dict[str, Any]) -> None:
        # Initialisation avec une configuration de stratégies, stockée dans un dictionnaire
        self._strat_config: Dict[str, Any] = strat_config

    @property
    def strat_config(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture à la configuration des stratégies
        return self._strat_config

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Méthode abstraite pour ajuster une stratégie.
        Doit être implémentée par des sous-classes pour adapter la stratégie à un contexte spécifique.
        """
        pass


class _BrokerAPI(ABC):
    """
    Classe abstraite qui définit l'interface pour les interactions avec les brokers.
    """

    @abstractmethod
    def get_current_positions(self, assets: List[str] = None):
        """
        Retourne toutes les positions ouvertes du compte.
        """
        pass

    @abstractmethod
    def get_open_orders(self, assets: List[str] = None):
        """
        Retourne tous les ordres ouverts.
        """
        pass

    @abstractmethod
    def get_available_cash(self):
        """
        Retourne le montant de cash disponible.
        """
        pass

    @abstractmethod
    def get_current_prices(self, assets):
        """
        Récupère les prix actuels pour une liste d'actifs.
        """
        pass

    @abstractmethod
    def place_orders(self, orders: List[Dict[str, Union[str, float]]]) -> List[str]:
        """
        Place un ordre sur le marché.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """
        Annule un ordre spécifique.
        """
        pass

    @abstractmethod
    def cancel_all_open_orders(self, assets: List[str] = None):
        """
        Annule tous ls odres ouverts
        """
        pass
