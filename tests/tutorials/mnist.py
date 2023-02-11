import torch
import torchvision
import torch_redstone as rst
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor


class MNISTTask(rst.Task):

    def data(self):
        transform = lambda x: to_tensor(x.convert('RGB'))
        train = MNIST('logs', True, transform, download=True)
        test = MNIST('logs', False, transform, download=True)
        return train, test

    def metrics(self):
        return [rst.CategoricalAcc().redstone(), torch.nn.CrossEntropyLoss().redstone()]


def main(epochs=10):
    model = torchvision.models.resnet18(num_classes=10)
    if torch.cuda.is_available():
        model = model.cuda()
    rst.DefaultLoop(
        model, MNISTTask(), optimizer='adam',
        adapter=rst.DirectPredictionAdapter(),
        batch_size=256
    ).run(epochs)


if __name__ == '__main__':
    main()
