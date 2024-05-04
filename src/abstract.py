from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.check import check_configs


class _Model(ABC):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        self._model_config: Dict[str, Any] = model_config
        self._metrics: Dict[str, Any] = {}

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics


class _Data(ABC):
    def __init__(self, data_config: Dict[str, Any]) -> None:
        self._data_config: Dict[str, Any] = data_config
        self._metrics: Dict[str, Any] = {}

    @property
    def data_config(self) -> Dict[str, Any]:
        return self._data_config

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics

    def update_metrics(self, model: _Model) -> None:
        """
        Updates metrics based on the model.
        """
        check_configs(data=self, model=model)
        self._metrics = {
            "model": model.model_config['model_config'],
            **model.metrics
        }


class _Strategies(ABC):
    def __init__(self, strat_config: Dict[str, Any]) -> None:
        self._strat_config: Dict[str, Any] = strat_config

    @property
    def strat_config(self) -> Dict[str, Any]:
        return self._strat_config

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        An abstract method that fits a strategy to the portfolio.
        """
        pass
