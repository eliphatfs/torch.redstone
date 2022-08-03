from typing import List, Sequence, Optional, Union
import torch
import torch.optim
import torch.nn
from torch.utils.data import DataLoader
import tqdm

from .loss import Loss, DefaultLoss
from .metric import Metric
from .task import Task
from .processor import Processor
from .utils import Meter, torch_to, ObjectProxy, torch_to_numpy, cat_proxies
from .types import EllipsisType, ResultInterface


class DefaultLoop:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        loss: Optional[Loss] = None,
        metrics: Optional[Sequence[Union[Metric, EllipsisType]]] = None,
        processors: Sequence[Processor] = [],
        optimizer: Union[str, torch.optim.Optimizer] = 'adam'
    ) -> None:
        """
        Construct the default training loop.

        Notes on `metrics`:
            `None` for task defaults.
            A list for custom ones.
            Ellipsis in list means appending task defaults.
        """
        self.task = task
        self.model = model
        self.train, self.val = task.data()

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
        self.processors = processors

    def add_metric(self, metric: Metric):
        self.metrics.append(metric)

    def run(self, num_epochs, train=True, val=True, max_steps=None):
        for epoch in num_epochs:
            for prx in self.processors:
                prx.pre_epoch(self.model, epoch)
            if train:
                train_rs = self.epoch(True, epoch, max_steps=max_steps)
            if val:
                val_rs = self.epoch(False, epoch, max_steps=max_steps)
            for prx in self.processors:
                prx.post_epoch(self.model, epoch, ObjectProxy(train=train_rs, val=val_rs))

    def epoch(
        self,
        training: bool = False,
        epoch: Optional[int] = None,
        loader: Optional[DataLoader] = None,
        max_steps: Optional[int] = None,
        return_input: bool = False,
        return_pred: bool = False
    ) -> ResultInterface:
        self.model.train(training)
        if loader is None:
            loader = self.train if training else self.val
        meter = Meter()
        ref_pt = next(self.model.parameters())
        torch.set_grad_enabled(training)
        if max_steps is not None:
            prog = tqdm.tqdm(zip(loader, range(max_steps)), total=max_steps)
        else:
            prog = tqdm.tqdm(loader)
        result: ResultInterface = ObjectProxy(metrics=None, inputs=[], preds=[])
        for d in prog:
            if return_input:
                result.inputs.append(torch_to_numpy(d))
            d = torch_to(d, ref_pt.device)
            for prx in self.processors:
                prx.pre_forward(d, self.model)
            output = self.model(*d)
            for prx in self.processors:
                prx.post_forward(d, self.model, output)
            metvals = ObjectProxy()
            for met in self.metrics:
                mval = met(d, output)
                setattr(metvals, met.name.lower(), mval)
                meter.u(met.name, mval.item())
            if training:
                loss = self.loss(d, output, metvals)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if return_pred:
                result.preds.append(torch_to_numpy(output))
            desc = "VT"[training] + " %02d" % epoch
            for k in sorted(meter.k):
                desc += " %s: %.4f" % (k, meter[k])
            prog.set_description(desc)
        result.metrics = ObjectProxy({k.lower(): meter[k] for k in sorted(meter.k)})
        result.inputs = cat_proxies(result.inputs) if return_input else None
        result.preds = cat_proxies(result.preds) if return_pred else None
        return result
