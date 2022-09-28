import unittest
import torch
import torch.nn
import torch.redstone as rst


class TestLib(unittest.TestCase):
    def test_lambda_layer(self):
        net = torch.nn.Sequential(rst.Lambda(lambda x: x[0]))
        a = torch.randn(8, 3, 128)
        b = torch.randn(8, 3, 128)
        self.assertTrue(torch.allclose(net([a, b]), a))

    def test_supercat(self):
        self.assertSequenceEqual(
            rst.supercat((torch.rand(256, 3), torch.rand(1, 6)), -1).shape,
            [256, 9]
        )
        self.assertSequenceEqual(
            rst.supercat((torch.rand(128, 1, 32, 3), torch.rand(6)), -1).shape,
            [128, 1, 32, 9]
        )
        self.assertSequenceEqual(
            rst.supercat((torch.rand(64, 1), torch.rand(8))).shape,
            [65, 8]
        )

    def test_mlp_shapes(self):
        mlp = rst.MLP([32, 100, 256, 100, 128], 1)
        self.assertSequenceEqual(
            mlp(torch.rand(4, 32, 5)).shape,
            [4, 128, 5]
        )
        mlp = rst.MLP([32, 100, 256, 100, 128], 2)
        self.assertSequenceEqual(
            mlp(torch.rand(4, 32, 5, 7)).shape,
            [4, 128, 5, 7]
        )
        mlp = rst.MLP([32, 100, 256, 100, 128], 0)
        self.assertSequenceEqual(
            mlp(torch.rand(4, 32)).shape,
            [4, 128]
        )

    def test_mlp_nonorm(self):
        mlp = rst.MLP([32, 100, 256, 100, 128], 1, norm=None)
        self.assertSequenceEqual(
            mlp(torch.rand(4, 32, 5)).shape,
            [4, 128, 5]
        )


if __name__ == "__main__":
    unittest.main()
