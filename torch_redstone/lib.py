from typing import List, Sequence, Union, Callable, Any, Optional
from typing_extensions import Literal
import torch
import torch.autograd.functional as ad
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .processor import Processor, Adapter
from .loss import Loss
from .metric import Metric
from .utils import container_catamorphism, AttrPath, visit_attr, ObjectProxy


AttrPathType = Union[AttrPath, str, Callable[[Any], Tensor], None]


class Index:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __index__(self):
        return self.idx


class AdvTrainingPGD(Processor):
    def __init__(
        self, loss_metric: Metric,
        no_perturb_attrs: List[AttrPathType]=[],
        eps=0.03, step_scale=0.5, n_steps=8, attack_at_test=False
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
        self.attack_at_test = attack_at_test

    def pre_forward(self, inputs, model: nn.Module):
        if not self.attack_at_test and not model.training:
            return
        skip = []
        collect = []
        perturb = []
        for skip_attr in self.skipped:
            skip.append(visit_attr(inputs, skip_attr))

        def _cata_indexing(tnsr):
            if isinstance(tnsr, Tensor) and torch.is_floating_point(tnsr):
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
            loss = self.loss(loss_inputs, self.feed(model, fill))
            return self.gscaler.scale(loss).reshape_as(loss)

        indexed = container_catamorphism(inputs, _cata_indexing)

        for _ in range(self.n_steps):
            grad = ad.jacobian(_wrap_fun, tuple(perturb))
            with torch.no_grad():
                for i in range(len(collect)):
                    perturb[i] += torch.sign(grad[i]) * self.step
                    perturb[i] = torch.clamp(perturb[i], -self.eps, self.eps)
        # print(inputs, container_catamorphism(indexed, _cata_fill(perturb)))
        return container_catamorphism(indexed, _cata_fill(perturb))


class Lambda(nn.Module):
    def __init__(self, lam) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, inputs):
        return self.lam(inputs)


class GetItem(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


def supercat(tensors: Sequence[Tensor], dim: int = 0):
    """
    Similar to `torch.cat`, but supports broadcasting. For example:

    [M, 32], [N, 1, 64] -- supercat 2 --> [N, M, 96]
    """
    ndim = max(x.ndim for x in tensors)
    tensors = [x.reshape(*[1] * (ndim - x.ndim), *x.shape) for x in tensors]
    shape = [max(x.size(i) for x in tensors) for i in range(ndim)]
    shape[dim] = -1
    tensors = [torch.broadcast_to(x, shape) for x in tensors]
    return torch.cat(tensors, dim)


xcat = supercat


def xreshape(
    tensor: torch.Tensor, shape: Sequence[int],
    s: Optional[int] = None, e: Optional[int] = None, dim: Optional[int] = None
):
    """
    Similar to torch.reshape, but supports reshaping a section (`s`-th dim to `e`-th dim, included) of the shape.

    If dim is set, s and e will be set to dim. An error will be raised if both dim and s or e is set.
    If either s or e is None, the reshape will start from the beginning or go through the end of the shape.

    [K, A * B, C] -- xreshape [A, B] dim 1 --> [K, A, B, C]

    [K, A * B, C * D] -- xreshape [A, -1, D] s -2 --> [K, A, B * C, D]
    """
    if dim is not None:
        if s is not None or e is not None:
            raise ValueError("`dim` and `s` or `e` cannot be set at the same time for `xreshape`")
        s = e = dim
    ndim = tensor.ndim
    if s is None:
        s = 0
    if e is None:
        e = -1
    if s < 0:
        s += ndim
    if e < 0:
        e += ndim
    if s > e:
        raise ValueError("Starting dim `s` cannot be greater than `e`")
    flat = tensor
    if s != e:
        flat = tensor.flatten(s, e)
    return flat.reshape(*tensor.shape[:s], *shape, *tensor.shape[e + 1:])


class MLP(nn.Module):
    def __init__(
        self,
        sizes: List[int],
        n_group_dims: Literal[0, 1, 2, 3] = 0,
        activation: Callable[[Tensor], Tensor] = F.relu,
        norm: Literal['batch', 'instance', None] = 'batch'
    ) -> None:
        """
        Multi-layer perceptron (Fully connected). The output layer is also normalized and activated.

        sizes: sizes of layers, including the input. [n_in, h_1, h_2, ..., n_out].
        n_group_dims: 0 if input is of shape [B, C], 1 if [B, C, N], 2 if [B, C, H, W], 3 if [B, C, D, H, W].
        activation: a `Tensor -> Tensor` function like `torch.relu`. Defaults to `relu`.
        norm: 'batch', 'instance', or `None`. Normalization layers type.
        """
        super().__init__()
        lin = [lambda a, b, _: nn.Linear(a, b), nn.Conv1d, nn.Conv2d, nn.Conv3d][n_group_dims]
        norm_dict = {
            'batch': [nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
            'instance': [nn.InstanceNorm1d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
        }
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        units = sizes[0]
        for units_latter in sizes[1:]:
            self.layers.append(lin(units, units_latter, 1))
            if norm is not None:
                self.norms.append(norm_dict[norm][n_group_dims](units_latter))
            else:
                self.norms.append(Lambda(lambda x: x))
            units = units_latter
        self.activation = activation

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = self.activation(norm(layer(x)))
        return x


class GradientNormOperator(Processor):

    def __init__(self, clip_norm: float, reject_norm: float) -> None:
        super().__init__()
        self.clip_norm = clip_norm
        self.reject_norm = reject_norm

    def pre_step(self, model: nn.Module, optimizer: Optimizer, metrics):
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.clip_norm
        )
        metrics.grad_norm = total_grad_norm
        if total_grad_norm > self.reject_norm:
            return True


class DirectPredictionAdapter(Adapter):
    def transform(self, inputs):
        x, y = inputs
        return ObjectProxy(x=x, y=y)

    def feed(self, net: nn.Module, inputs):
        return ObjectProxy(logits=net(inputs.x))


class TorchMetric(Loss, Metric):
    def __init__(
        self, torch_module: nn.Module, name = 'Loss',
        pred_path: AttrPathType = 'logits', label_path: AttrPathType = 'y'
    ) -> None:
        super().__init__()
        self.name = name
        self.torch_module = torch_module
        self.pred_path = pred_path
        self.label_path = label_path

    def __call__(self, inputs, model_return, metrics: ObjectProxy = None) -> torch.Tensor:
        return self.torch_module(
            visit_attr(model_return, self.pred_path),
            visit_attr(inputs, self.label_path)
        )


class BinaryAcc(nn.Module):

    def __init__(self, th_logits = 0.0, th_label = 0.5) -> None:
        super().__init__()
        self.th_logits = th_logits
        self.th_label = th_label

    def forward(self, inputs, targets):
        return ((inputs > self.th_logits) == (targets > self.th_label)).float().mean()

    def redstone(self, name: str = 'Acc', pred_path: AttrPathType = 'logits', label_path: AttrPathType = 'y'):
        return TorchMetric(self, name, pred_path, label_path)


class CategoricalAcc(nn.Module):

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs, targets):
        return (inputs.argmax(self.dim) == targets).float().mean()

    def redstone(self, name: str = 'Acc', pred_path: AttrPathType = 'logits', label_path: AttrPathType = 'y'):
        return TorchMetric(self, name, pred_path, label_path)


def _loss_redstone(self, name: str = 'Loss', pred_path: AttrPathType = 'logits', label_path: AttrPathType = 'y'):
    return TorchMetric(self, name, pred_path, label_path)

torch.nn.modules.loss._Loss.redstone = _loss_redstone
