from abc import abstractmethod
import torch

from .utils import ObjectProxy


class Metric:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, inputs, model_return) -> torch.Tensor:
        raise NotImplementedError
