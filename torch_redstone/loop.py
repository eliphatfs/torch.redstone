from typing import List, Sequence, Optional, Union
import torch
import torch.optim
import torch.nn
from torch.utils.data import DataLoader, Dataset
import tqdm

from .loss import Loss, DefaultLoss
from .metric import Metric
from .task import Task
from .processor import Processor, Adapter
from .utils import Meter, ObjectProxy, torch_to, torch_to_numpy
from .utils import cat_proxies, collate_support_object_proxy, sanitize_name
from .types import EllipsisType, ResultInterface
from .log import Logger


def take_first(iterable, n):
    iterator = iter(iterable)
    for _ in range(n):
        yield next(iterator)


class DefaultLoop:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        loss: Optional[Loss] = None,
        metrics: Optional[Sequence[Union[Metric, EllipsisType]]] = None,
        processors: Sequence[Processor] = None,
        optimizer: Union[str, torch.optim.Optimizer] = 'adam',
        adapter: Adapter = Adapter(), *,
        batch_size=32, num_workers=0
    ) -> None:
        """
        Construct the default training loop.

        Notes on `metrics`:
            `None` for task defaults.
            A list for custom ones.
            Ellipsis in list means appending task defaults.
            Default displayed name for a metric is its class name.
            This can be overridden by setting `metric.name`.
            In returned `ObjectProxy` for metrics, attribute names are lower-case displayed names.

        Notes on `loss`:
            `None` for taking `metrics.loss` as loss.

        Notes on `optimizer`:
            It may be `torch.optim.Optimizer` instance or name of optimizer.
            Supported names are adaptive optimizers: `adam`, `adadelta` and `rmsprop`.

        Notes on data loaders:
            `batch_size` and `num_workers` are ignored if the task generates DataLoaders already.
            You may override the creation behavior in `create_data_loader`.
        """
        self.task = task
        self.model = model
        train, val = task.data()
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train = self.create_data_loader(train, True)
        self.val = self.create_data_loader(val, False)

        self.metrics: List[Metric] = []
        if metrics is None:
            metrics = [...]
        for met in metrics:
            if met is ...:
                for task_metric in task.metrics():
                    self.add_metric(task_metric)
            else:
                self.add_metric(met)

        if loss is None:
            loss = DefaultLoss()
        self.loss = loss

        if isinstance(optimizer, str):
            if optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters())
            elif optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters())
            elif optimizer == 'adadelta':
                optimizer = torch.optim.Adadelta(model.parameters())
            else:
                raise ValueError("Unsupported auto-optimizer", optimizer)
        self.optimizer = optimizer
        if processors is None:
            processors = [Logger()]
        self.processors = processors
        self.adapter = adapter

    def create_data_loader(self, data: Union[Dataset, list], is_train: bool):
        return DataLoader(
            data, self.batch_size, is_train, num_workers=self.num_workers,
            collate_fn=collate_support_object_proxy
        )

    def add_metric(self, metric: Metric):
        self.metrics.append(metric)

    def run(self, num_epochs, train=True, val=True, max_steps=None, quiet=False):
        for epoch in range(num_epochs):
            for prx in self.processors:
                prx._adapter = self.adapter
                prx.pre_epoch(self.model, epoch)
            train_rs = self.epoch(True, epoch, max_steps=max_steps, quiet=quiet) if train else None
            val_rs = self.epoch(False, epoch, max_steps=max_steps, quiet=quiet) if val else None
            epoch_rs = ObjectProxy(train=train_rs, val=val_rs)
            for prx in self.processors:
                prx.post_epoch(self.model, epoch, epoch_rs)
        return epoch_rs

    def epoch(
        self,
        training: bool = False,
        epoch: Optional[int] = None,
        loader: Optional[DataLoader] = None,
        max_steps: Optional[int] = None,
        return_input: bool = False,
        return_pred: bool = False,
        quiet: bool = False
    ) -> ResultInterface:
        self.model.train(training)
        if loader is None:
            loader = self.train if training else self.val
        meter = Meter()
        ref_pt = next(self.model.parameters())
        torch.set_grad_enabled(training)
        if max_steps is not None:
            if quiet:
                prog = take_first(loader, max_steps)
            else:
                prog = tqdm.tqdm(take_first(loader, max_steps), total=max_steps)
        else:
            if quiet:
                prog = loader
            else:
                prog = tqdm.tqdm(loader)
        result: ResultInterface = ObjectProxy(metrics=None, inputs=[], preds=[])
        for d in prog:
            if return_input:
                result.inputs.append(torch_to_numpy(d))
            d = torch_to(d, ref_pt.device)
            d = self.adapter.transform(d)
            for prx in self.processors:
                ret = prx.pre_forward(d, self.model)
                if ret is not None:
                    d = ret
            output = self.adapter.feed(self.model, d)
            for prx in self.processors:
                ret = prx.post_forward(d, self.model, output)
                if ret is not None:
                    output = ret
            metvals = ObjectProxy()
            for met in self.metrics:
                mval = met(d, output)
                setattr(metvals, sanitize_name(met.name.lower()), mval)
                meter.u(met.name, mval.item())
            if training:
                loss = self.loss(d, output, metvals)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for prx in self.processors:
                    prx.post_step(self.model, self.optimizer, metvals)
            if return_pred:
                result.preds.append(torch_to_numpy(output))
            desc = "VT"[training] + (" %02d" % epoch if epoch is not None else "")
            for k in sorted(meter.k):
                desc += " %s: %.4f" % (k, meter[k])
            if hasattr(prog, 'set_description'):
                prog.set_description(desc)
        result.metrics = ObjectProxy(**{sanitize_name(k.lower()): meter[k] for k in sorted(meter.k)})
        result.inputs = cat_proxies(result.inputs) if return_input else None
        result.preds = cat_proxies(result.preds) if return_pred else None
        return result
