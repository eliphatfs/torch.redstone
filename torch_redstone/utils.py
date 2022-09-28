import collections
import random
import types
from typing import Any, Callable, Sequence, Type, Union
import typing
import numpy
import torch
from torch.utils.data.dataloader import default_collate


class Meter:
    def __init__(self) -> None:
        self.k = collections.defaultdict(int)
        self.v = collections.defaultdict(int)

    def __getitem__(self, k):
        if self.k[k] == 0:
            return 0
        return self.v[k] / self.k[k]

    def u(self, k, v):
        self.k[k] += 1
        self.v[k] += v


class AttrPath:
    def __init__(self, get=lambda x: x, set=lambda x, v: None) -> None:
        self.get = get
        self.set = set

    def __getattr__(self, attr):
        return AttrPath(
            lambda x: getattr(self.get(x), attr),
            lambda x, v: setattr(self.get(x), attr, v)
        )


def visit_attr(Q, attr: Union[AttrPath, str, Callable[[Any], torch.Tensor]]):
    if isinstance(attr, str):
        return getattr(Q, attr)
    if isinstance(attr, AttrPath):
        return attr.get(Q)
    return attr(Q)


class ObjectProxy(types.SimpleNamespace):

    def __getattr__(self, name):
        proxy = ObjectProxy()
        setattr(self, name, proxy)
        return proxy

    @classmethod
    def zip(cls, **kwargs):
        """
        zip(k1=v1, k2=v2, ...) -> Iter[ObjectProxy], where `v1`, ... are iterable.

        Each `ObjectProxy` has attributes `k1`, `k2`, ... whose values are
        the corresponding items in `v1`, `v2`, ...
        """
        keys = []
        vals = []
        for k, v in kwargs.items():
            keys.append(k)
            vals.append(v)
        for vs in zip(*vals):
            yield cls(**dict(zip(keys, vs)))


TElem = typing.TypeVar('TElem')
TRes = typing.TypeVar('TRes')


class TC(typing.Generic[TElem]):
    pass


def container_catamorphism(
    data: TC[TElem], func: Callable[[TElem], TRes]
) -> TC[TRes]:
    """
    Transforms `TElem` in `list`, `dict`, `ObjectProxy`, `tuple`, `set` with `func`.
    Nested containers are also supported.
    """
    if isinstance(data, ObjectProxy):
        return ObjectProxy(**container_catamorphism(data.__dict__, func))
    if isinstance(data, dict):
        return {
            k: container_catamorphism(v, func) for k, v in data.items()
        }
    if isinstance(data, list):
        return [container_catamorphism(x, func) for x in data]
    if isinstance(data, tuple):
        return tuple(container_catamorphism(x, func) for x in data)
    if isinstance(data, set):
        return {container_catamorphism(x, func) for x in data}
    return func(data)


def torch_to(data: TC[torch.Tensor], reference: Union[str, torch.device, torch.Tensor]) -> TC[torch.Tensor]:
    """
    Recursively send `torch.Tensor` in `list`, `dict`, `ObjectProxy`, `tuple`, `set` to `reference`.
    """
    return container_catamorphism(data, lambda x: x.to(reference))


def torch_to_numpy(data: TC[torch.Tensor]) -> TC[numpy.ndarray]:
    """
    Recursively fetch `torch.Tensor` in `list`, `dict`, `ObjectProxy`, `tuple`, `set` as numpy array.
    """
    return container_catamorphism(data, lambda x: x.detach().cpu().numpy())


def container_pushdown(seq: Sequence[TC], target_cls: Type[TElem] = list) -> TC[TElem]:
    """
    Push sequence of containers (`list`, `dict`, `ObjectProxy`, `tuple`) inwards.
    Nested containers supported.

    Example: [{a: 0, b: 1}, {a: 1, b: 2}] -> {a: [0, 1], b: [1, 2]}
    """
    data = seq[0]
    c = target_cls
    if isinstance(data, ObjectProxy):
        return ObjectProxy(**container_pushdown([x.__dict__ for x in seq], c))
    if isinstance(data, dict):
        return {
            k: container_pushdown([x[k] for x in seq], c) for k in data.keys()
        }
    if isinstance(data, list):
        return [container_pushdown([x[i] for x in seq], c) for i in range(len(data))]
    if isinstance(data, tuple):
        return tuple(container_pushdown([x[i] for x in seq], c) for i in range(len(data)))
    return target_cls(seq)


class CollateList:
    def __init__(self, *args, **kwargs) -> None:
        self.wrap = list(*args, **kwargs)


def cat_proxies(proxies: Sequence[ObjectProxy], axis=0):
    """
    Merge (concatenate) sequence of `ObjectProxy` whose elements are numpy arrays.
    """
    return container_catamorphism(
        container_pushdown(proxies, CollateList),
        lambda x: numpy.concatenate(x.wrap, axis=axis)
    )


def collate_support_object_proxy(batch):
    return container_catamorphism(
        container_pushdown(batch, CollateList),
        lambda x: default_collate(x.wrap)
    )


def seed(seed: int):
    """
    Set random seeds to `seed` for `torch`, `numpy` and python `random`.
    """
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
