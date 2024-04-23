import torch
import torch.nn as nn
import torch.optim

from .types import EpochResultInterface


class Adapter:
    def transform(self, inputs):
        return inputs

    def feed(self, net: nn.Module, inputs):
        return net(inputs)


class Processor:
    gscaler: torch.cuda.amp.GradScaler
    _adapter: Adapter

    def feed(self, model: nn.Module, inputs):
        return self._adapter.feed(model, inputs)

    def pre_forward(self, inputs, model: nn.Module):
        pass

    def post_forward(self, inputs, model: nn.Module, model_return):
        pass

    def pre_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, metrics):
        pass

    def post_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, metrics):
        pass

    def pre_epoch(self, model: nn.Module, epoch: int):
        pass

    def post_epoch(self, model: nn.Module, epoch: int, epoch_result: EpochResultInterface):
        pass
