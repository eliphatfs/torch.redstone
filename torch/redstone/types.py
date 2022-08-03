from typing import Any, Optional

from .utils import ObjectProxy


EllipsisType = type(...)


class ResultInterface:
    metrics: ObjectProxy
    inputs: Optional[Any]
    preds: Optional[Any]


class EpochResultInterface:
    train: Optional[ResultInterface]
    val: Optional[ResultInterface]
