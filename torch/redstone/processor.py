import torch.nn as nn
import torch.optim

from .utils import ObjectProxy
from .types import EpochResultInterface


class Processor:
    def pre_forward(self, inputs, model: nn.Module):
        pass

    def post_forward(self, inputs, model: nn.Module, model_return):
        pass

    def post_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, metrics):
        pass

    def pre_epoch(self, model: nn.Module, epoch: int):
        pass

    def post_epoch(self, model: nn.Module, epoch: int, epoch_result: EpochResultInterface):
        pass
