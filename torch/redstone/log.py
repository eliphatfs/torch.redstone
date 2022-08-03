import os
import time

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
