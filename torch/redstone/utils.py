import collections
import random
import types
from typing import Callable, Sequence, Union
import typing
import numpy
import torch


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


class ObjectProxy(types.SimpleNamespace):

    def __getattr__(self, name):
        proxy = ObjectProxy()
        setattr(self, name, proxy)
        return proxy


T = typing.TypeVar('T')


def tensor_catamorphism(data: T, func: Callable[[torch.Tensor], torch.Tensor]) -> T:
    """
    Transforms `torch.Tensor` or `list`, `dict`, `ObjectProxy`, `tuple`, `set` of `torch.Tensor` with `func`.
    Nested containers are also supported. Objects not recognized are returned as-is.
    """
    if isinstance(data, torch.Tensor):
        return func(data)
    if isinstance(data, ObjectProxy):
        return ObjectProxy(**tensor_catamorphism(data.__dict__, func))
    if isinstance(data, dict):
        return {
            k: tensor_catamorphism(v, func) for k, v in data.items()
        }
    if isinstance(data, list):
        return [tensor_catamorphism(x, func) for x in data]
    if isinstance(data, tuple):
        return tuple(tensor_catamorphism(x, func) for x in data)
    if isinstance(data, set):
        return {tensor_catamorphism(x, func) for x in data}
    return data


def torch_to(data: T, reference: Union[str, torch.device, torch.Tensor]) -> T:
    """
    Recursively send `torch.Tensor` in `list`, `dict`, `ObjectProxy`, `tuple`, `set` to `reference`.
    """
    return tensor_catamorphism(data, lambda x: x.to(reference))


def torch_to_numpy(data: T) -> T:
    """
    Recursively fetch `torch.Tensor` in `list`, `dict`, `ObjectProxy`, `tuple`, `set` as numpy array.
    """
    return tensor_catamorphism(data, lambda x: x.detach().cpu().numpy())


def cat_proxies(proxies: Sequence[ObjectProxy], axis=0):
    """
    Merge (concatenate) sequence of `ObjectProxy` whose elements are numpy arrays.
    """
    result = ObjectProxy()
    for proxy in proxies:
        for k in proxy.__dict__.keys():
            if not isinstance(getattr(result, k), list):
                setattr(result, k, list())
            getattr(result, k).append(getattr(proxy, k))
    for k in result.__dict__.keys():
        setattr(result, k, numpy.concatenate(getattr(result, k), axis=axis))
    return result


def seed(seed: int):
    """
    Set random seeds to `seed` for `torch`, `numpy` and python `random`.
    """
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
