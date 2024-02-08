from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Iterator, List, Union

import numpy as np
from multipledispatch import dispatch

from ldata import Dataset


class Benchmark(ABC, Dataset):
    """Abstract class for a benchmark."""

    class AggregationMethod(Enum):
        """The method to aggregate the scores of the (input, output) pairs."""

        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        SUM = "sum"
        NONE = "none"

    def __init__(self, data_path: str, config: Dataset.Config):
        super().__init__(data_path, config)

    @dispatch(
        Callable[[str], str],
        Union[str, List[str], Iterator[str]],
        Union[str, List[str], Iterator[str]],
        AggregationMethod,
    )
    def evaluate(
        self,
        subject: Callable[[str], str],
        input: Union[
            str,
            List[str],
            Iterator[str],
        ],
        target: Union[
            str,
            List[str],
            Iterator[str],
        ],
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
    ) -> Union[float, List[float]]:
        """
        Evaluate the subjects on the benchmark.
        The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes a single input string and returns a single output string.
        `aggregation_method`: the method to aggregate the scores of the (input, output) pairs.
        `input`: the input to evaluate, which must be a string, a list of strings, or an iterator of strings.
        `target`: the target output, which must be a string, a list of strings, or an iterator of strings.

        ### Returns
        ----------
        The aggregated score of the outputs.
        - If `aggregation_method` is `AggregationMethod.NONE`, the return value is a list of scores, one for each (input, output) pair.
        - Otherwise, the return value is a single score, which is the result of the aggregation method.

        ### Raises
        ----------
        `AssertionError`: if the input and target are not of the same length.
        `ValueError`: if the input type is not supported.
        `ValueError`: if the target type is not supported.
        `ValueError`: if the aggregation method is not supported.
        """

        if isinstance(input, str):
            input = [input]
        elif isinstance(input, Iterator):
            input = list(input)
        elif not isinstance(input, list):
            raise ValueError(
                f"Input type '{type(input)}' is not supported. It must be a string, a list of strings, or an iterator of strings."
            )

        if isinstance(target, str):
            target = [target]
        elif isinstance(target, Iterator):
            target = list(target)
        elif not isinstance(target, list):
            raise ValueError(
                f"Target type '{type(target)}' is not supported. It must be a string, a list of strings, or an iterator of strings."
            )

        assert len(input) == len(
            target
        ), "The number of inputs and targets must be the same."

        scores = [
            self._evaluate_impl(subject(input[i]), target[i]) for i in range(len(input))
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
                f"Aggregation method '{aggregation_method}' is not supported."
            )

    @dispatch(Callable[[str], str], AggregationMethod)
    def evaluate(  # noqa: F811
        self,
        subject: Callable[[str], str],
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
    ) -> Union[float, List[float]]:
        """
        Evaluate the subjects on the benchmark.
        The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.

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

        input = self.test_set.input
        target = self.test_set.target

        return self.evaluate(subject, input, target, aggregation_method)

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
