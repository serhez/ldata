from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from ldata.dataset import Dataset


class Benchmark(ABC, Dataset):
    """Abstract class for a benchmark."""

    @dataclass(kw_only=True)
    class Config(Dataset.Config):
        """The configuration of a benchmark."""

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
    def get_instructed(
        self, sample: Optional[Union[str, Dataset.Split]] = None
    ) -> Union[str, Dataset.Split]:
        """
        Add instructions relevant to solve the task to the sample.

        ### Parameters
        ----------
        [optional] `sample`: the sample to add instructions to.
        - If `None`, the test set is used.

        ### Returns
        ----------
        The sample with instructions.
        - If `sample` is a `Dataset.Split`, the instructions are added to the inputs only.

        ### Raises
        ----------
        `ValueError`: if `sample` is neither a `Dataset.Split` nor a string.
        """

        if (
            sample is not None
            and not isinstance(sample, Dataset.Split)
            and not isinstance(sample, str)
        ):
            raise ValueError(
                f"`sample` must be either `None`, a `Dataset.Split` or a string, not {type(sample)}."
            )

    def evaluate(
        self,
        subject: Callable[[str, List[Tuple[str, str]]], str],
        evaluation_method: EvaluationMethod = EvaluationMethod.CHARACTER,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
        instructed: bool = True,
    ) -> Union[float, List[float]]:
        """
        Evaluate the subjects on the benchmark.
        The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.
        The inputs and targets are taken from the whole test set.
        Examples (i.e., shots) will be provided to the subjects to allow for in-context learning; the number of shots is determined by `Benchmark.Config.n_shots`.
        `extract_solution` is used internally to extract the solution from the output and format it into the `target` format, hence you don't need to perform this step before calling this function.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes a single input string and returns a single output string.
        `evaluation_method`: the level of exactness measured by the evaluation metric.
        `aggregation_method`: the method to aggregate the scores of the (input, output) pairs.
        `instructed`: whether to use the instructed test set (as given by `get_instructed`) or the regular test set; also applied to the shots.

        ### Returns
        ----------
        The aggregated score of the outputs.
        - If `aggregation_method` is `AggregationMethod.NONE`, the return value is a list of scores, one for each (input, output) pair.
        - Otherwise, the return value is a single score, which is the result of the aggregation method.
        """

        if instructed:
            test_set = self.get_instructed()
            shots = self.get_instructed(self.shots)
        else:
            test_set = self.test_set
            shots = self.shots

        inputs = test_set.inputs
        targets = test_set.targets

        assert len(inputs) == len(
            targets
        ), "the number of inputs and targets must be the same."

        scores = [
            self._evaluate_impl(
                self.extract_solution(subject(inputs[i], shots), targets[i]),
                targets[i],
                evaluation_method,
            )
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
    def _evaluate_impl(
        self, output: str, target: str, evaluation_method: EvaluationMethod
    ) -> float:
        """
        The benchmark's internal implementation of `evaluate` acting on a single (input, output) pair; do not call this method directly.
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

        raise NotImplementedError

    def extract_solution(
        self,
        outputs: Union[str, List[str]],
        targets: Union[str, List[str]],
        evaluation_method: EvaluationMethod = EvaluationMethod.CHARACTER,
    ) -> Union[str, List[str]]:
        """
        Extracts the attempted solution from the output and formats it into the `target` format.
        If no approprate solution is found, an empty string is returned.

        ### Parameters
        ----------
        `outputs`: the output of the model, split by spaces.
        `targets`: the target output, split by spaces.
        `evaluation_method`: the level of exactness measured by the evaluation metric.

        ### Returns
        ----------
        The extracted and formatted solution.
        - If `output` is a list of strings, the return value is a list of strings.

        ### Raises
        ----------
        `ValueError`: if `outputs` and `targets` are not of the same type (either both lists or strings).
        """

        if isinstance(outputs, list) and isinstance(targets, list):
            return [
                self._extract_solution_impl(o, t, evaluation_method)
                for o, t in zip(outputs, targets)
            ]
        elif isinstance(outputs, str) and isinstance(targets, str):
            return self._extract_solution_impl(outputs, targets, evaluation_method)
        else:
            raise ValueError(
                f"`outputs` and `targets` must be either both lists of strings or both strings, not {type(outputs)} and {type(targets)}."
            )

    @abstractmethod
    def _extract_solution_impl(
        self,
        output: str,
        target: str,
        evaluation_method: EvaluationMethod = EvaluationMethod.CHARACTER,
    ) -> str:
        """
        The benchmark's internal implementation of `extract_solution`.

        ### Parameters
        ----------
        `output`: the output of the model, split by spaces.
        `target`: the target output, split by spaces.
        `evaluation_method`: the level of exactness measured by the evaluation metric.

        ### Returns
        ----------
        The extracted and formatted solution.
        """

        raise NotImplementedError
