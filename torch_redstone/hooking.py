import contextlib
import torch.nn as nn
from typing import Optional
from torch.utils.hooks import RemovableHandle


class StopExecution(Exception):
    pass


class RedstoneDisposeHandle(object):
    def __init__(self) -> None:
        self.handle: Optional[RemovableHandle] = None

    def dispose(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class RedstoneCatchHook(RedstoneDisposeHandle):
    def __init__(self, count, raise_on_record) -> None:
        super().__init__()
        self.record = []
        self.count = count
        self.raise_on_record = raise_on_record

    def get_all(self):
        try:
            return self.record
        finally:
            self.record.clear()

    def get(self):
        return self.record.pop(0)

    def __del__(self):
        self.dispose()

    def add(self, x):
        if self.count and len(self.record) >= self.count:
            raise ValueError("`catch_input` got more catches than `check_catch_count`", self.count)
        self.record.append(x)
        if self.raise_on_record:
            raise StopExecution(x)


class RedstoneCatchInput(RedstoneCatchHook):

    def __call__(self, module, inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        self.add(inputs)


class RedstoneCatchOutput(RedstoneCatchHook):

    def __call__(self, module, inputs, output):
        self.add(output)


def catch_input_to(module: nn.Module, check_catch_count: Optional[int] = 1):
    rci = RedstoneCatchInput(check_catch_count, False)
    rci.handle = module.register_forward_pre_hook(rci)
    return rci


def catch_output_from(module: nn.Module, check_catch_count: Optional[int] = 1, raise_stop_execution: bool = False):
    rco = RedstoneCatchOutput(check_catch_count, raise_stop_execution)
    rco.handle = module.register_forward_hook(rco)
    return rco


def modify_input_to(module: nn.Module, apply_func):
    rc = RedstoneDisposeHandle()
    rc.handle = module.register_forward_pre_hook(lambda mod, inputs: apply_func(*inputs))
    return rc


def catching_scope():
    return contextlib.suppress(StopExecution)
