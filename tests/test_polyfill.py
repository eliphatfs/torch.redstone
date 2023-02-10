import unittest
import torch
from torch.redstone import Polyfill


class TestPolyfill(unittest.TestCase):
    def test_cdist(self):
        for i in range(1, 4):
            a = torch.randn(100, 32, 3)
            b = torch.randn(100, 64, 3)
            self.assertSequenceEqual(Polyfill.cdist(a, b).shape, [100, 32, 64])
            self.assertTrue(torch.allclose(Polyfill.cdist(a, b), torch.cdist(a, b), atol=2e-5))
            a = torch.randn(32, i * 4, 6)
            br = torch.randn(32, 1, 6)
            self.assertSequenceEqual(Polyfill.cdist(a, br).shape, torch.cdist(a, br).shape)
            self.assertTrue(torch.allclose(Polyfill.cdist(a, br), torch.cdist(a, br), atol=2e-5))

    def test_broadcast_to(self):
        a = torch.randn(100, 1, 16)
        self.assertSequenceEqual(Polyfill.broadcast_to(a, [100, 8, 16]).shape, [100, 8, 16])
        self.assertTrue(torch.allclose(Polyfill.broadcast_to(a, [100, 8, 16]), a * torch.ones(100, 8, 16)))
    
    def test_square(self):
        for i in range(1, 8):
            a = torch.randn(100, i, i)
            self.assertSequenceEqual(Polyfill.square(a).shape, a.shape)
            self.assertTrue(torch.allclose(Polyfill.square(a), torch.square(a)))


if __name__ == "__main__":
    unittest.main()
