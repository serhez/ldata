from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

from ldata.dataset import Dataset


class Benchmark(ABC, Dataset):
    """Abstract class for a benchmark."""

    @dataclass(kw_only=True)
    class Config(Dataset.Config):
        """The configuration of a benchmark."""

        name: str
        """The name of the benchmark used for reporting."""

    class AggregationMethod(Enum):
        """The method to aggregate the scores of the (input, output) pairs."""

        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        SUM = "sum"

    class EvaluationMethod(Enum):
        """The level of exactness measured by the evaluation metric."""

        EXACT = "exact"
        WORD = "word"
        CHARACTER = "character"

    def __init__(self, config: Config):
        """
        Initialize a `Benchmark`.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

        self._name = config.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    @property
    def name(self) -> str:
        """The name of the benchmark."""
        return self._name

    @abstractmethod
    @overload
    def get_instructed(self, sample: str) -> str:
        """
        Add instructions relevant to solve the task to the sample.

        ### Parameters
        ----------
        `sample`: the sample to add instructions to.

        ### Returns
        ----------
        The sample with instructions.
        """

        ...

    @abstractmethod
    @overload
    def get_instructed(self, sample: Dataset.Split | None = None) -> Dataset.Split:
        """
        Add instructions relevant to solve the task to the sample.

        ### Parameters
        ----------
        [optional] `sample`: the sample to add instructions to.
        - If `None`, the test set is used.

        ### Returns
        ----------
        The sample with instructions; the instructions are added to the inputs only.
        """

        ...

    @abstractmethod
    def get_instructed(self, sample=None) -> str | Dataset.Split: ...

    @abstractmethod
    @overload
    def get_uninstructed(self, sample: str) -> str:
        """
        Remove instructions from the sample.

        ### Parameters
        ----------
        `sample`: the sample to remove instructions from.

        ### Returns
        ----------
        The sample without instructions.
        """

        ...

    @abstractmethod
    @overload
    def get_uninstructed(self, sample: Dataset.Split | None = None) -> Dataset.Split:
        """
        Remove instructions from the sample.

        ### Parameters
        ----------
        [optional] `sample`: the sample to remove instructions from.
        - If `None`, the test set is used.

        ### Returns
        ----------
        The sample without instructions; the instructions are removed from the inputs only.
        """

        ...

    @abstractmethod
    def get_uninstructed(self, sample=None) -> str | Dataset.Split: ...

    def evaluate_subject(
        self,
        subject: Callable[[list[Any]], tuple[list[Any], dict[str, Any]]],
        n_samples: int | None = None,
        evaluation_method: EvaluationMethod = EvaluationMethod.EXACT,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
        instructed: bool = True,
    ) -> tuple[float, npt.NDArray[np.float64], list[str], list[str], dict[str, Any]]:
        """
        Evaluate a subject on the benchmark.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes an array of input strings and returns a tuple containing an array of output strings and a dictionary of usage statistics.
        `n_samples`: the number of samples to evaluate the subject on.
        - If `None` (default), the whole test set is used.
        `evaluation_method`: the level of exactness measured by the evaluation metric.
        `aggregation_method`: the method to aggregate the scores of the (input, output) pairs.
        `instructed`: whether to use the instructed test set (as given by `get_instructed`) or the regular test set.

        ### Returns
        ----------
        A tuple containing the aggregated score, the list of scores, the list of raw outputs, the list of extracted solutions (via `extract_solution`) and the aggregate usage statistics of the underlying models.

        ### Raises
        ----------
        `ValueError`: if `aggregation_method` is not supported.
        `AssertionError`: if `n_samples` is not `None` and is less than 1, or greater than the number of samples in the test set.
        `AssertionError`: if the number of inputs and targets is not the same.
        `AssertionError`: if the number of output strings returned by the subject is not the same as the number of input strings.

        ### Notes
        ----------
        - The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.
        - The inputs and targets are taken from the whole test set.
        - `extract_solution` is used internally to extract the solution from the output and format it into the `target` format, hence you don't need to perform this step before calling this function.
        """

        if instructed:
            test_set = self.get_instructed()
        else:
            test_set = self.test_set

        assert (
            n_samples is None or len(test_set) >= n_samples >= 1
        ), "n_samples must be >= 1 and <= len(test_set)."

        if n_samples is not None:
            test_set = test_set.sample(n_samples)

        inputs = test_set.inputs
        targets = test_set.targets
        assert len(inputs) == len(
            targets
        ), "the number of inputs and targets must be the same."

        outputs, stats = subject(list(inputs))
        assert (
            len(outputs) == len(inputs)
        ), "the number of output strings returned by the subject must be the same as the number of input strings."

        scores, found_solutions = zip(
            *[
                self.evaluate_output(o, t, evaluation_method)
                for o, t in zip(outputs, targets)
            ]
        )
        scores = np.array(scores)

        if aggregation_method == self.AggregationMethod.MEAN:
            agg_score = float(np.mean(scores))
        elif aggregation_method == self.AggregationMethod.MEDIAN:
            agg_score = float(np.median(scores))
        elif aggregation_method == self.AggregationMethod.MAX:
            agg_score = np.max(scores)
        elif aggregation_method == self.AggregationMethod.MIN:
            agg_score = np.min(scores)
        elif aggregation_method == self.AggregationMethod.SUM:
            agg_score = np.sum(scores)
        else:
            raise ValueError(
                f"aggregation method '{aggregation_method}' is not supported."
            )

        return agg_score, scores, outputs, found_solutions, stats

    def _call_impl(self, *args, **kwargs):
        return self.evaluate_subject(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @classmethod
    def evaluate_output(
        cls, output: str, target: str, evaluation_method: EvaluationMethod
    ) -> tuple[float, str]:
        """
        Evaluate a single (input, output) pair.

        ### Parameters
        ----------
        `output`: the output of the subject.
        `target`: the target output.
        `evaluation_method`: the level of exactness measured by the evaluation metric.

        ### Returns
        -------
        A tuple containing the found solution (via `extract_solution`) and the score of the solution.

        ### Notes
        ----------
        - `extract_solution` is used internally to extract the solution from the output and format it into the `target` format, hence you don't need to perform this step before calling this function.
        """

        found_solution = cls.extract_solution([output], [target])[0]
        score = cls._evaluate_output_impl(found_solution, target, evaluation_method)

        return score, found_solution

    @classmethod
    @abstractmethod
    def _evaluate_output_impl(
        cls, output: str, target: str, evaluation_method: EvaluationMethod
    ) -> float:
        """
        The child class' internal implementation of `evaluate_output`.
        It can be assumed that `output` is an extracted solution via `extract_solution`, i.e., it follows the format of the `target`.
        It is recommended for the scores to be in the range of [0.0, 1.0] and to increase linearly with the quality of the results.

        ### Parameters
        ----------
        `output`: the output of the subject.
        `target`: the target output.
        `evaluation_method`: the level of exactness measured by the evaluation metric.

        ### Returns
        -------
        The score of the output.
        """

        ...

    @classmethod
    @abstractmethod
    def compute_target(cls, input: str, **kwargs: Any) -> str:
        """
        Compute the target output for the uninstructed input.
        If the input is instructed, you must use `get_uninstructed` before passing it to this function.

        ### Parameters
        ----------
        `input`: the uninstructed input string.

        ### Returns
        ----------
        The target output.
        """

        ...

    @classmethod
    @overload
    def extract_solution(cls, output: list[str], target: list[str]) -> list[str]:
        """
        Extracts the attempted solution from the outputs and formats it into the `target` format.
        If no approprate solution is found, an empty string is returned.

        ### Parameters
        ----------
        `outputs`: the output of the model, split by spaces.
        `targets`: the target output, split by spaces.

        ### Returns
        ----------
        The extracted and formatted solutions.

        ### Raises
        ----------
        `ValueError`: if `outputs` and `targets` are not of the same length.
        """

        ...

    @classmethod
    @overload
    def extract_solution(cls, output: str, target: str) -> str:
        """
        Extracts the attempted solution from the output and formats it into the `target` format.
        If no approprate solution is found, an empty string is returned.

        ### Parameters
        ----------
        `outputs`: the output of the model, split by spaces.
        `targets`: the target output, split by spaces.

        ### Returns
        ----------
        The extracted and formatted solution.
        """

        ...

    @classmethod
    def extract_solution(cls, output, target) -> str | list[str]:
        if isinstance(output, str):
            return cls._extract_solution_impl(output, target)

        assert len(output) == len(
            target
        ), "the number of outputs and targets must be the same."

        return [cls._extract_solution_impl(o, t) for o, t in zip(output, target)]

    @classmethod
    @abstractmethod
    def _extract_solution_impl(cls, output: str, target: str) -> str:
        """
        The benchmark's internal implementation of `extract_solution`.

        ### Parameters
        ----------
        `output`: the output of the model, split by spaces.
        `target`: the target output, split by spaces.

        ### Returns
        ----------
        The extracted and formatted solution.
        """

        ...
