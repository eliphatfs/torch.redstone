import unittest
import numpy
import torch.redstone as rst


class TestContainerFunctions(unittest.TestCase):
    def test_catamorphism(self):
        cc = rst.container_catamorphism
        self.assertEqual(cc([1, [2, 3]], lambda x: x * 2), [2, [4, 6]])
        self.assertEqual(cc([1, rst.ObjectProxy(c=4)], lambda x: x + 3)[1].c, 7)

    def test_pushdown(self):
        self.assertEqual(
            rst.container_pushdown([{'a': 0, 'b': 1}, {'a': 1, 'b': 2}], tuple),
            {'a': (0, 1), 'b': (1, 2)}
        )

class TestObjectProxy(unittest.TestCase):
    def test_visit(self):
        x = rst.ObjectProxy(a=0, b=1)
        self.assertEqual(x.a, 0)
        self.assertEqual(x.b, 1)
        x.c.x.c = 5
        x.c.y.c = 3
        self.assertEqual(x.a, 0)
        self.assertEqual(x.b, 1)
        self.assertEqual(x.c.y.c, 3)
        self.assertEqual(x.c.x.c, 5)

    def test_cat_proxies(self):
        self.assertTrue(numpy.allclose(
            rst.cat_proxies([
                rst.ObjectProxy(a=numpy.array([[0, 1]])),
                rst.ObjectProxy(a=numpy.array([[3, 2]]))
            ]).a,
            numpy.array([[0, 1], [3, 2]])
        ))


if __name__ == "__main__":
    unittest.main()
