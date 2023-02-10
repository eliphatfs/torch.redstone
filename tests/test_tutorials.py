import os
import sys
import unittest


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# class TestTutorials(unittest.TestCase):
#     def test_mnist(self):
#         import tutorials.mnist
#         tutorials.mnist.main(1, True)
# Too slow for CPU CI machines
