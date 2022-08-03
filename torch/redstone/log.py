import os
import time
from typing import Type, Union
import torch

from .metric import Metric
from .utils import AttrPath
from .types import EpochResultInterface
from .processor import Processor


module_load_time = int(time.time())


class Logger(Processor):
    def __init__(self, exp_name: str = "training", directory: str = "./logs/"):
        super().__init__()
        self.exp_name = exp_name
        self.path = directory
        self.wrote_headers = False
        os.makedirs(self.path, exist_ok=True)

    def get_file_path(self):
        filename = "%s_%d.csv" % (self.exp_name, module_load_time)
        return os.path.join(self.path, filename)

    def write_log(self, *data):
        with open(self.get_file_path(), "a") as fo:
            for d in data:
                fo.write(str(d))
                fo.write(",")
            fo.write("\n")

    def post_epoch(self, model, epoch, epoch_result: EpochResultInterface):
        train = epoch_result.train.metrics.__dict__ if epoch_result.train else {}
        val = epoch_result.val.metrics.__dict__ if epoch_result.val else {}
        ti = list(train.items())
        vi = list(val.items())
        keys = ["epoch"] + ["train_" + k for k, v in ti] + ["val_" + k for k, v in vi]
        vals = [epoch] + [v for k, v in ti] + [v for k, v in vi]
        if not self.wrote_headers:
            self.write_log(*keys)
            self.wrote_headers = True
        self.write_log(*vals)


class BestSaver(Processor):
    def __init__(
        self, metric: Union[AttrPath, str, Type[Metric]] = "acc",
        model_name: str = "model", directory: str = "./logs/",
        lower_better: bool = False
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.path = directory
        if isinstance(metric, type):
            metric = metric.__name__.lower()
        self.metric_attr = metric
        self.best = float('inf') if lower_better else -float('inf')
        self.lower_better = lower_better
        os.makedirs(self.path, exist_ok=True)

    def get_file_path(self):
        filename = "%s_%d.dat" % (self.model_name, module_load_time)
        return os.path.join(self.path, filename)

    def post_epoch(self, model, epoch, epoch_result: EpochResultInterface):
        if isinstance(self.metric_attr, str):
            if epoch_result.val:
                met = getattr(epoch_result.val.metrics, self.metric_attr)
            elif epoch_result.train:
                met = getattr(epoch_result.val.metrics, self.metric_attr)
        else:
            met = self.metric_attr.get(epoch_result)
        if self.lower_better:
            if met < self.best:
                self.best = met
                torch.save(model.state_dict(), self.get_file_path())
        else:
            if met > self.best:
                self.best = met
                torch.save(model.state_dict(), self.get_file_path())


class BestLossSaver(BestSaver):
    def __init__(self, model_name: str = "model", directory: str = "./logs/") -> None:
        super().__init__("loss", model_name, directory, lower_better=True)
