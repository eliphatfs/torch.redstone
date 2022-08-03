from typing import Optional

from .utils import ObjectProxy


EllipsisType = type(...)


class ResultInterface:
    metrics: ObjectProxy
    inputs: Optional[ObjectProxy]
    preds: Optional[ObjectProxy]


class EpochResultInterface:
    train: ResultInterface
    val: ResultInterface
