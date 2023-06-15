import time
import torch
import torch.overrides
from itertools import chain
from functools import partial
from types import SimpleNamespace
from typing import Union, Optional
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


overridable = set(torch.overrides.get_overridable_functions()[torch])
memory_cap = {}

memory_headroom = 1024
all_evictable = defaultdict(set)


def unify_device(device: Union[None, int, torch.device] = None):
    if device is None:
        return unify_device(torch.cuda.current_device())
    if isinstance(device, torch.device):
        return device.index
    assert isinstance(device, int)
    return device


def get_memory_cap(device: Union[None, int, torch.device] = None) -> int:
    device = unify_device(device)
    if device not in memory_cap:
        set_memory_cap(device)
    return memory_cap[device]


def set_memory_cap(device: Union[None, int, torch.device] = None, megabytes: int = None):
    device = unify_device(device)
    if megabytes is None:
        memory_cap[device] = int(torch.cuda.mem_get_info(device)[1] * 0.95 / 1024 / 1024)
    else:
        memory_cap[device] = int(megabytes)


class DTRDispatcher(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(func, types, args)
        def unwrap_proxy(t):
            if isinstance(t, WrapperTensor):
                return t.dtr
            else:
                return t
        args = tree_map(unwrap_proxy, args)
        kwargs = tree_map(unwrap_proxy, kwargs)
        return WrapperTensor(torch.empty(0), dtr=DynamicRematerializedTensor(func, types, args, kwargs))


class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, dtr):
        return super().__new__(cls, elem)

    def __init__(self, elem, dtr):
        self.dtr = dtr


class DynamicRematerializedTensor:

    def __init__(self, op, types, args, kwargs: Optional[dict]) -> None:
        super().__init__()
        self.op = op
        self.types = types
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        self.neighbours = []
        self.stats = SimpleNamespace()
        self.pinned = 0
        self.v = None
        self.evicted = True
        self.pin()
        self.unpin()

        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, DynamicRematerializedTensor):
                arg.neighbours.append(self)
                self.neighbours.append(arg)

    def evict(self):
        assert self.pinned == 0
        self.evicted = True
        all_evictable[unify_device(self.v.device)].remove(self)
        self.v = None

    def pin(self):
        self.pinned += 1
        if self.evicted:
            self.rematerialize()
        self.record_use()

    def record_use(self):
        self.stats.last_access = time.process_time()

    def unpin(self):
        self.pinned -= 1
        assert self.pinned >= 0

    def rematerialize(self):
        device = torch.cuda.current_device()
        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, DynamicRematerializedTensor):
                arg.pin()
            if isinstance(arg, (torch.Tensor, DynamicRematerializedTensor)):
                device = arg.device
        device = unify_device(device)
        cvt_types = []
        cvt_args = []
        cvt_kwargs = {}
        for arg in self.args:
            if isinstance(arg, DynamicRematerializedTensor):
                cvt_args.append(arg.v)
                cvt_types.append(type(arg.v))
            else:
                cvt_args.append(arg)
                cvt_types.append(type(arg))
        for k, arg in self.kwargs.items():
            if isinstance(arg, DynamicRematerializedTensor):
                cvt_kwargs[k] = arg.v
            else:
                cvt_kwargs[k] = arg
        while torch.cuda.memory_allocated(device) / 1024 / 1024 + memory_headroom > get_memory_cap(device):
            to_evict: DynamicRematerializedTensor = min(
                all_evictable[device],
                key=partial(DynamicRematerializedTensor.h_dtr, current_time=time.process_time() + 1e-8)
            )
            to_evict.evict()
        t = time.perf_counter()
        self.v = self.op(*cvt_args, **cvt_kwargs)
        self.stats.cost = time.perf_counter() - t
        self.stats.mem = self.v.element_size() * self.v.nelement() / 1024 / 1024
        self.evicted = False
        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, DynamicRematerializedTensor):
                arg.unpin()
        all_evictable[unify_device(self.v.device)].add(self)

    def h_dtr(self, current_time):
        if self.pinned > 0:
            return float('inf')
        c = self.stats.cost
        for t in self.neighbours:
            c += t.stats.cost
        return c / self.stats.mem / (current_time - self.stats.last_access)

    def __repr__(self) -> str:
        if self.evicted:
            return f"DTREvictable(op={self.op}, evicted)"
        else:
            return f"DTREvictable(v={self.v})"


net = torch.nn.Sequential(*[
    torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
    ) for _ in range(16)
]).cuda()
# for ch in net.modules():
#     for k, p in ch.named_parameters(recurse=False):
#         setattr(ch, k, p.data)
set_memory_cap(0, 5120)
with DTRDispatcher.push():
    net(torch.randn(100000, 1024).cuda()).sum().backward()
print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_cached())
pass
