from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
from utils.check import check_configs  # Importation des utilitaires nécessaires


# Définition de la classe abstraite _Model
class _Model(ABC):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        # Initialisation avec une configuration du modèle, stockée dans un dictionnaire
        self._model_config: Dict[str, Any] = model_config
        self._metrics: Dict[str, Any] = {
            "fit_date": None}  # Dictionnaire pour stocker les métriques associées au modèle

    @property
    def model_config(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture à la configuration du modèle
        return self._model_config

    @property
    def metrics(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture aux métriques du modèle
        return self._metrics

    @abstractmethod
    def fit_fcast(self, data: _Data, horizon: datetime):
        """
        Méthode abstraite pour ajuster le modèle à des données et prédire jusqu'à un certain horizon.
        Doit être implémentée par des sous-classes spécifiques.
        """
        check_configs(data=data, model=self, check_date=False)
        if self.metrics["fit_date"] and self.metrics["fit_date"] > data.data_config["end_date"]:
            raise ValueError("Please use recent data.")
        pass


# Définition de la classe abstraite _Data
class _Data(ABC):
    def __init__(self, data_config: Dict[str, Any]) -> None:
        # Initialisation avec une configuration des données, stockée dans un dictionnaire
        self._data_config: Dict[str, Any] = data_config
        self._metrics: Dict[str, Any] = {}  # Dictionnaire pour stocker les métriques associées aux données

    @property
    def data_config(self) -> Dict[str, Any]:
        # Propriété pour accéder en lecture à la configuration des données
        return self._data_config

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

    @abstractmethod
    def update_data(self, new_end_date: datetime = None):
        """
        Méthode abstraite pour mettre à jour les données jusqu'à une nouvelle date de fin.
        La date de fin est optionnelle; si non spécifiée, peut-être mise à jour jusqu'à la date courante.
        """
        pass

    def update_metrics(self, model: _Model) -> None:
        """
        Met à jour les métriques en fonction des résultats et configurations du modèle fourni.
        Utilise les configurations du modèle pour enrichir les métriques des données.
        """
        check_configs(data=self, model=model)  # Validation des configurations
        self._metrics = {
            "model": model.model_config['model_config'],  # à ajuster en cas de plusieur types de model
            **model.metrics  # Intégration des métriques du modèle aux métriques des données
        }

    @abstractmethod
    def window_returns(self, _date, _horizon):
        pass


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
