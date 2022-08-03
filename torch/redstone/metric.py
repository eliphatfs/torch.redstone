from abc import abstractmethod
import torch

from .utils import ObjectProxy


class Metric:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, inputs: ObjectProxy, model_return: ObjectProxy) -> torch.Tensor:
        raise NotImplementedError
