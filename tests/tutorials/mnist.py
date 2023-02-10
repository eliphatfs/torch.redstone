import torch
import torchvision
import torch_redstone as rst
import torch.nn.functional as F


class Acc(rst.Metric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        return (model_return.argmax(-1) == inputs.y).float().mean()


class Loss(rst.Metric):
    def __call__(self, inputs: rst.ObjectProxy, model_return) -> torch.Tensor:
        return F.cross_entropy(model_return, inputs.y)


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        return rst.ObjectProxy(x=x, y=y)


class MNISTTask(rst.Task):
    def __init__(self) -> None:
        super().__init__()
        self.tr = torchvision.transforms.ToTensor()
        self.train = MNIST('logs', True, self.tr, download=True)
        self.test = MNIST('logs', False, self.tr, download=True)

    def data(self):
        return self.train, self.test

    def metrics(self):
        return [Acc(), Loss()]


def main(epochs=10, quiet=False):
    model = torch.nn.Sequential(
        rst.Lambda(lambda inputs: torch.cat([inputs.x] * 3, 1)),
        torchvision.models.resnet18(num_classes=10)
    )
    if torch.cuda.is_available():
        model = model.cuda()
    loop = rst.DefaultLoop(
        model, MNISTTask(), optimizer='adam',
        processors=[rst.Logger(), rst.BestSaver(verbose=0 if quiet else 1)],
        batch_size=256
    )
    metrics = loop.run(epochs, train=True, val=True, quiet=quiet)
    if not quiet:
        print("Final accuracy", metrics.val.metrics.acc)


if __name__ == '__main__':
    main()
