from abc import abstractmethod
import torch

from .utils import ObjectProxy


class Loss:
    @abstractmethod
    def __call__(self, inputs: ObjectProxy, model_return: ObjectProxy, metrics: ObjectProxy) -> torch.Tensor:
        raise NotImplementedError


class DefaultLoss(Loss):
    def __call__(self, inputs: ObjectProxy, model_return: ObjectProxy, metrics: ObjectProxy) -> torch.Tensor:
        return metrics.loss
