from abc import abstractmethod
from typing import Sequence, Tuple, Union
from torch.utils.data import Dataset, DataLoader

from .metric import Metric


class Task:
    @abstractmethod
    def data(self) -> Tuple[Union[Dataset, DataLoader, list], Union[Dataset, DataLoader, list]]:
        raise NotImplementedError

    @abstractmethod
    def metrics(self) -> Sequence[Metric]:
        raise NotImplementedError
