import unittest
import torch
import torch.nn
import torch.nn.functional as F
import torch.redstone as rst
from torch.redstone import ObjectProxy
import numpy


class Acc(rst.Metric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        return ((model_return.logits > 0) == (inputs.y > 0)).float().mean()


class Loss(rst.Metric):
    def __call__(self, inputs: ObjectProxy, model_return: ObjectProxy) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(model_return.logits, inputs.y)


class SimpleClassificationTask(rst.Task):
    def gen(self):
        f32 = numpy.float32
        x = numpy.random.randn(3000, 4).astype(f32)
        y = (x.sum(-1) > 0).astype(f32)
        return list(ObjectProxy.zip(x=x, y=y))

    def data(self):
        return self.gen(), self.gen()

    def metrics(self):
        return [Acc(), Loss()]


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(4, 1, bias=False),
            torch.nn.Flatten(0)
        )

    def forward(self, proxy):
        return ObjectProxy(logits=self.cls(proxy.x))


class TestLinearClassifier(unittest.TestCase):
    def test_train_linear_cls(self):
        rst.seed(42)
        loop = rst.DefaultLoop(Model(), SimpleClassificationTask(), optimizer='adadelta')
        self.assertGreater(loop.run(1).val.metrics.acc, 0.8)


if __name__ == "__main__":
    unittest.main()
