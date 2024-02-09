from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from enum import Enum
from typing import Any, Callable, List, Union

import numpy as np

from ldata.dataset import Dataset


class Benchmark(ABC, Dataset):
    """Abstract class for a benchmark."""

    @dataclass(kw_only=True)
    class Config(Dataset.Config):
        name: str = MISSING
        """The name of the benchmark used for reporting."""

    class AggregationMethod(Enum):
        """The method to aggregate the scores of the (input, output) pairs."""

        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        SUM = "sum"
        NONE = "none"

    def __init__(self, data_path: str, config: Config):
        super().__init__(data_path, config)

        self._name = config.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        """The name of the benchmark."""
        return self._name

    def evaluate(
        self,
        subject: Callable[[str], str],
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
    ) -> Union[float, List[float]]:
        """
        Evaluate the subjects on the benchmark.
        The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.
        The inputs and targets are taken from the complete test set.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes a single input string and returns a single output string.
        `aggregation_method`: the method to aggregate the scores of the (input, output) pairs.

        ### Returns
        ----------
        The aggregated score of the outputs.
        - If `aggregation_method` is `AggregationMethod.NONE`, the return value is a list of scores, one for each (input, output) pair.
        - Otherwise, the return value is a single score, which is the result of the aggregation method.
        """

        inputs = self.test_set.inputs
        targets = self.test_set.targets

        assert len(inputs) == len(
            targets
        ), "the number of inputs and targets must be the same."

        scores = [
            self._evaluate_impl(subject(inputs[i]), targets[i])
            for i in range(len(inputs))
        ]

        if aggregation_method == self.AggregationMethod.MEAN:
            return np.mean(scores)
        elif aggregation_method == self.AggregationMethod.MEDIAN:
            return np.median(scores)
        elif aggregation_method == self.AggregationMethod.MAX:
            return np.max(scores)
        elif aggregation_method == self.AggregationMethod.MIN:
            return np.min(scores)
        elif aggregation_method == self.AggregationMethod.SUM:
            return np.sum(scores)
        elif aggregation_method == self.AggregationMethod.NONE:
            return scores
        else:
            raise ValueError(
                f"aggregation method '{aggregation_method}' is not supported."
            )

    def _call_impl(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _evaluate_impl(self, output: str, target: str) -> float:
        """
        The benchmark's internal implementation of `evaluate` acting on a single (input, output) pair.
        It is recommended for the scores to be in the range of [0.0, 1.0] and to increase linearly with the quality of the results.

        ### Parameters
        ----------
        `output`: the output of the subject.
        `target`: the target output.

        ### Returns
        -------
        The score of the output.
        """

        pass
