from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

from ldata.dataset import Dataset
from ldata.protocols import Logger


class Benchmark(ABC, Dataset):
    """Abstract class for a benchmark."""

    @dataclass(kw_only=True)
    class Config(Dataset.Config):
        """The configuration of a benchmark."""

        name: str
        """The name of the benchmark used for reporting."""

    @property
    @abstractmethod
    def config_cls(cls) -> type[Config]:
        """The configuration class for the Benchmark."""

        ...

    class Aggregation(Enum):
        """The method to aggregate the scores of the (input, output) pairs."""

        MEAN = "mean"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        SUM = "sum"

    class Evaluation(Enum):
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
        evaluation_method: Evaluation = Evaluation.EXACT,
        aggregation_method: Aggregation = Aggregation.MEAN,
        instructed: bool = True,
        shuffle: bool = False,
        unsafe: bool = False,
        logger: Logger | None = None,
    ) -> tuple[float, npt.NDArray[np.float64], list[str], list[str], dict[str, Any]]:
        """
        Evaluate a subject on the benchmark's test set.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes an array of input strings and returns a tuple containing an array of output strings and a dictionary of usage statistics.
        `n_samples`: the number of samples to evaluate the subject on.
        - If `None` (default), the whole test set is used.
        `evaluation_method`: the level of exactness measured by the evaluation metric.
        `aggregation_method`: the method to aggregate the scores of the (input, output) pairs.
        `instructed`: whether to use the instructed test set (as given by `get_instructed`) or the regular test set.
        `shuffle`: whether to shuffle the test set before selecting the samples.
        - If `shuffle == False`, repeated calls with the same `n_samples` will test the given subject on the same samples (the first `n_samples` of the current test set).
        - If `shuffle == True`, the test set will be shuffled before selecting the samples, hence the results will be different for each call.
        - You can also shuffle the test set before calling this function via the `shuffle` method.
        `unsafe`: if `True`, the exceptions raised by the subject will not be caught, which might lead to the function crashing in the middle of the evaluation and all the results being lost.
        - The advantage of setting `unsafe = True` is that the samples can be passed to the subject in batches, which might be faster depending on the subject's implementation.
        [optional] `logger`: the logger to use for logging the exceptions raised by the subject.

        ### Returns
        ----------
        A tuple containing the aggregated score, the list of scores, the list of raw outputs, the list of extracted solutions (via `extract_solution`) and the aggregate usage statistics of the underlying models.

        ### Raises
        ----------
        `ValueError`: if `aggregation_method` is not supported.
        `AssertionError`: if `n_samples` is not `None` and is less than 1, or greater than the number of samples in the test set.
        `AssertionError`: if the number of inputs and targets is not the same.
        - This will happen if the child class' implementation is incorrect.
        `AssertionError`: if the number of output strings returned by the subject is not the same as the number of input strings.
        - This can only happen if `unsafe == True`.

        ### Notes
        ----------
        - The evaluation metric and the the possible range of score values should be available in the benchmark's documentation.
        - `extract_solution` is used internally to extract the solution from the output and format it into the `target` format, hence you don't need to perform this step before calling this function.
        - All exceptions raised by the subject are caught, reported by the logger (if provided); the score is set to 0.0, the stats are set to `None` and the output is set to an empty string.
        """

        assert (
            n_samples is None or self.test_len >= n_samples >= 1
        ), "n_samples must be >= 1 and <= len(test_set)."

        if aggregation_method == self.Aggregation.MEAN:

            def agg_fn(x):  # type: ignore[reportRedeclaration]
                return float(np.mean(x))
        elif aggregation_method == self.Aggregation.MEDIAN:

            def agg_fn(x):  # type: ignore[reportRedeclaration]
                return float(np.median(x))
        elif aggregation_method == self.Aggregation.MAX:

            def agg_fn(x):
                return np.max(x)
        elif aggregation_method == self.Aggregation.MIN:

            def agg_fn(x):
                return np.min(x)
        elif aggregation_method == self.Aggregation.SUM:

            def agg_fn(x):
                return np.sum(x)
        else:
            raise ValueError(
                f"aggregation method '{aggregation_method}' is not supported."
            )

        if instructed:
            test_set = self.get_instructed()
        else:
            test_set = self.test_set

        if n_samples is not None:
            if shuffle:
                test_set = test_set.sample(n_samples)
            else:
                test_set = test_set[:n_samples]

        inputs = test_set.inputs
        targets = test_set.targets
        assert len(inputs) == len(
            targets
        ), "the number of inputs and targets must be the same."

        if unsafe:
            outputs, stats = subject(list(inputs))
        else:
            outputs, stats = [], {}
            for i in range(0, len(inputs), 1):
                if logger is not None:
                    logger.info(
                        f"[Benchmark.evaluate_subject] Evaluating sample {i + 1}/{len(inputs)}"
                    )
                try:
                    o, s = subject([inputs[i]])
                    outputs.append(o[0])
                    for k, v in s.items():
                        if k not in stats:
                            stats[k] = []
                        stats[k].append(v)
                except Exception as e:
                    if logger is not None:
                        logger.error(
                            {
                                "[Benchmark.evaluate_subject] Exception raised while evaluating the subject": str(
                                    e
                                ),
                                "Input": inputs[i],
                                "Corrective action": "The stats are set to None and the output is set to an empty string; the score will likely be 0.0.",
                            }
                        )
                    outputs.append("")
                    for k in stats.keys():
                        if k not in stats:
                            stats[k] = []
                        stats[k].append(None)
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
        agg_score = agg_fn(scores)

        return agg_score, scores, outputs, found_solutions, stats

    def _call_impl(self, *args, **kwargs):
        return self.evaluate_subject(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @classmethod
    def evaluate_output(
        cls, output: str, target: str, evaluation_method: Evaluation
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
        cls, output: str, target: str, evaluation_method: Evaluation
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
