import collections


class Meter:
    def __init__(self) -> None:
        self.k = collections.defaultdict(int)
        self.v = collections.defaultdict(int)

    def __getitem__(self, k):
        if self.k[k] == 0:
            return 0
        return self.v[k] / self.k[k]

    def u(self, k, v):
        self.k[k] += 1
        self.v[k] += v


class ObjectProxy(object):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for name, val in kwargs.items():
            setattr(self, name, val)

    def __getattr__(self, name):
        proxy = ObjectProxy()
        setattr(self, name, proxy)
        return proxy
