import unittest
import numpy
import torch
import torch.redstone as rst


class TestTorchTo(unittest.TestCase):
    def test_to(self):
        container = [torch.zeros([5, 4]), torch.zeros([6, 7]), "hello"]
        inttainer = rst.torch_to(container, torch.tensor(1, dtype=torch.long))
        self.assertEqual(len(inttainer), 3)
        self.assertEqual(len(inttainer), len(container))
        self.assertSequenceEqual([x.dtype for x in inttainer[:2]], [torch.long] * 2)
        self.assertEqual(inttainer[2], "hello")
        
    @unittest.expectedFailure
    def test_to_strict(self):
        container = [torch.zeros([5, 4]), torch.zeros([6, 7]), "hello"]
        rst.torch_to(container, torch.tensor(1, dtype=torch.long), True)

    def test_numpy(self):
        a = numpy.random.randn(30, 5)
        self.assertTrue(numpy.allclose(a, rst.torch_to_numpy([torch.tensor(a), 0])[0]))

    @unittest.expectedFailure
    def test_numpy_strict(self):
        a = numpy.random.randn(30, 5)
        self.assertTrue(numpy.allclose(a, rst.torch_to_numpy([torch.tensor(a), 0], True)[0]))


if __name__ == "__main__":
    unittest.main()
