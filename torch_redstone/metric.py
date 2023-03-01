from abc import abstractmethod
import torch


class Metric:
    @property
    def name(self):
        return getattr(self, '_name', self.__class__.__name__)

    @name.setter
    def name(self, n):
        self._name = n

    @abstractmethod
    def __call__(self, inputs, model_return) -> torch.Tensor:
        raise NotImplementedError
