from enum import Enum
from numbers import Number

import numpy as np
import numpy.typing as npt

_NUMBER_LIST_OPERATION_DESC = {
    "sum": "Sum",
    "product": "Multiply",
    "mean": "Find the mean of",
    "median": "Find the median of",
    "max": "Find the maximum of",
    "min": "Find the minimum of",
}

_NUMBER_LIST_OPERATION_FN = {
    "sum": np.sum,
    "product": np.prod,
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "max": np.max,
    "min": np.min,
}


class NumberListOperation(Enum):
    SUM = "sum"
    PRODUCT = "product"
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"

    def __str__(self) -> str:
        return _NUMBER_LIST_OPERATION_DESC[self.value]

    def __call__(
        self, numbers: npt.NDArray[np.number] | list[Number | np.number]
    ) -> float:
        return float(_NUMBER_LIST_OPERATION_FN[self.value](numbers))


class SortingOrder(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
