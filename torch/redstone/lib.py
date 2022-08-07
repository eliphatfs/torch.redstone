from typing import List, Union, Callable, Any
import torch
import torch.autograd.functional as ad
import torch.nn as nn
from torch import Tensor

from .processor import Processor
from .metric import Metric
from .utils import container_catamorphism, AttrPath, visit_attr


class Index:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __index__(self):
        return self.idx


class AdvTrainingPGD(Processor):
    def __init__(
        self, loss_metric: Metric,
        no_perturb_attrs: List[Union[AttrPath, str, Callable[[Any], Tensor]]]=[],
        eps=0.03, step_scale=0.5, n_steps=8
    ) -> None:
        """
        Processor for L_inf PGD adversarial (robust) training.
        """
        super().__init__()
        self.loss = loss_metric
        self.n_steps = n_steps
        self.eps = eps
        self.step = eps * step_scale
        self.skipped = no_perturb_attrs

    def pre_forward(self, inputs, model: nn.Module):
        if not model.training:
            return
        skip = []
        collect = []
        perturb = []
        for skip_attr in self.skipped:
            skip.append(visit_attr(inputs, skip_attr))

        def _cata_indexing(tnsr):
            if isinstance(tnsr, Tensor):
                if any(x is tnsr for x in skip):
                    return tnsr
                collect.append(tnsr)
                perturb.append(torch.zeros_like(tnsr))
                return Index(len(collect) - 1)
            return tnsr

        def _cata_fill(vals):
            return lambda x: collect[x] + vals[x] if isinstance(x, Index) else x

        def _cata_detach(tnsr):
            if isinstance(tnsr, Tensor):
                return tnsr.detach().clone()
            return tnsr

        def _wrap_fun(*vals):
            fill = container_catamorphism(indexed, _cata_fill(vals))
            loss_inputs = container_catamorphism(inputs, _cata_detach)
            return self.loss(loss_inputs, model(fill))

        indexed = container_catamorphism(inputs, _cata_indexing)

        for _ in range(self.n_steps):
            grad = ad.jacobian(_wrap_fun, tuple(perturb))
            with torch.no_grad():
                for i in range(len(collect)):
                    perturb[i] += torch.sign(grad[i]) * self.step
                    perturb[i] = torch.clamp(perturb[i], -self.eps, self.eps)
        # print(inputs, container_catamorphism(indexed, _cata_fill(perturb)))
        return container_catamorphism(indexed, _cata_fill(perturb))
