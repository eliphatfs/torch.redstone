import os
import sys
import unittest
import torchvision
import torch_redstone as rst


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestTutorials(unittest.TestCase):
    def test_mnist(self):
        from tutorials.mnist import MNISTTask
        model = torchvision.models.resnet18(num_classes=10)
        rst.DefaultLoop(
            model, MNISTTask(), optimizer='adam',
            adapter=rst.DirectPredictionAdapter(),
            batch_size=32
        ).run(1, max_steps=3, quiet=True)
