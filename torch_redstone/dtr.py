import time
import tqdm
import torch
from itertools import chain
from functools import partial
from types import SimpleNamespace
from typing import Union, Optional
from collections import defaultdict


memory_cap = {}

memory_headroom = 1024
all_evictable = defaultdict(set)
DEBUG = False


def dtr_debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


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


class DynamicRematerializedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, op, types, args, kwargs: Optional[dict]):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=elem.device,
        )

    @staticmethod
    def from_tensor(elem):
        return DynamicRematerializedTensor(elem, None, (), (), None)

    def __init__(self, elem, op, types, args, kwargs: Optional[dict]) -> None:
        super().__init__()
        self.op = op
        self.types = types
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        self.neighbours = []
        self.stats = SimpleNamespace(cost=0, mem=elem.element_size() * elem.nelement() / 1048576)
        self.pinned = 0
        self.v = elem
        self.evicted = False

        all_evictable[unify_device(self.v.device)].add(self)
        self.record_use()
        if op is None:
            self.pin()

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
        if self.pinned == 1 and not self.evicted:
            all_evictable[unify_device(self.v.device)].remove(self)
        self.record_use()

    def record_use(self):
        self.stats.last_access = time.process_time()

    def unpin(self):
        self.pinned -= 1
        if self.pinned == 0 and not self.evicted:
            all_evictable[unify_device(self.v.device)].add(self)
        assert self.pinned >= 0

    def rematerialize(self):
        self.v = self.do_compute(self.op, self.types, self.args, self.kwargs)
        self.evicted = False
        all_evictable[unify_device(self.v.device)].add(self)
        dtr_debug_print("[rematerialize]", id(self))

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

    @classmethod
    def do_compute(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        device = torch.cuda.current_device()
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, DynamicRematerializedTensor):
                arg.pin()
                device = arg.v.device
            elif isinstance(arg, torch.Tensor):
                device = arg.device
        device = unify_device(device)
        # cvt_types = []
        cvt_args = []
        cvt_kwargs = {}
        for arg in args:
            if isinstance(arg, DynamicRematerializedTensor):
                cvt_args.append(arg.v)
                # cvt_types.append(type(arg.v))
            else:
                cvt_args.append(arg)
                # cvt_types.append(type(arg))
        for k, arg in kwargs.items():
            if isinstance(arg, DynamicRematerializedTensor):
                cvt_kwargs[k] = arg.v
            else:
                cvt_kwargs[k] = arg
        while torch.cuda.memory_allocated(device) / 1048576 + memory_headroom > get_memory_cap(device):
            # dtr_debug_print("[targets]", device, [*map(id, all_evictable[device])])
            to_evict: DynamicRematerializedTensor = min(
                all_evictable[device],
                key=partial(DynamicRematerializedTensor.h_dtr, current_time=time.process_time() + 1e-8)
            )
            # dtr_debug_print("[evicting]", id(to_evict), torch.cuda.memory_allocated(device) / 1048576)
            to_evict.evict()
            dtr_debug_print("[evicted]", id(to_evict), torch.cuda.memory_allocated(device) / 1048576)
        v = func(*cvt_args, **cvt_kwargs)
        for arg in chain(args, kwargs.values()):
            if isinstance(arg, DynamicRematerializedTensor):
                arg.unpin()
        return v

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        t = time.perf_counter()
        v = cls.do_compute(func, types, args, kwargs)
        dtr = DynamicRematerializedTensor(
            v, func, types, args, kwargs
        )
        dtr.stats.cost = time.perf_counter() - t
        dtr.stats.mem = v.element_size() * v.nelement() / 1024 / 1024
        dtr_debug_print(func, types, '->', id(dtr))
        return dtr


# torch.manual_seed(0)
# net = torch.nn.Sequential(*[
#     torch.nn.Sequential(
#         torch.nn.Linear(1024, 1024),
#         torch.nn.ReLU(),
#     ) for _ in range(16)
# ]).cuda()
# opt = torch.optim.SGD(net.parameters(), 0.01)
# set_memory_cap(0, 5120)
# for i in tqdm.trange(32):
#     net(DynamicRematerializedTensor.from_tensor(torch.randn(100000, 1024).cuda())).sum().backward()
#     opt.step()
#     opt.zero_grad(True)
# # print(net[0][0].weight.grad)
# print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
