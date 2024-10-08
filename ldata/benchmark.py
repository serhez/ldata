import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, overload

import numpy as np
import numpy.typing as npt

from ldata.dataset import Dataset
from ldata.protocols import Addable, Logger
from ldata.types import EvaluationMetric
from ldata.utils import NumberListOperation


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
        subject: Callable[[list[Any]], tuple[list[Any], Addable | None]],
        n_samples: int | None = None,
        n_skip_samples: int = 0,
        metric: EvaluationMetric = EvaluationMetric.EXACT,
        aggregation_fn: NumberListOperation = NumberListOperation.MEAN,
        instructed: bool = True,
        shuffle: bool = False,
        unsafe: bool = False,
        logger: Logger | None = None,
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        list[str],
        list[str],
        npt.NDArray[np.float64],
        float,
        Addable | None,
    ]:
        """
        Evaluate a subject on the benchmark's test set.

        ### Parameters
        ----------
        `subject`: the subject to evaluate, which must be a function that takes an array of input strings and returns a tuple containing an array of output strings and an optional addable object with extra information about the generation process.
        `n_samples`: the number of samples from the test set to evaluate the subject on.
        - If `None` (default), the whole test set is used.
        `n_skip_samples`: the number of samples to skip from the beginning of the test set.
        - The number of samples that will be used is `n_samples - n_skip_samples`. Thus, `n_skip_samples` must be strictly less than `n_samples`.
        - If `shuffle == True`, this parameter is ignored.
        `metric`: the metric to evaluate the output.
        `aggregation_fn`: the method to aggregate the scores of the (input, output) pairs.
        `instructed`: whether to use the instructed test set (as given by `get_instructed`) or the regular test set.
        `shuffle`: whether to shuffle the test set before selecting the samples.
        - If `shuffle == False`, repeated calls with the same `n_samples` will test the given subject on the same samples (the first `n_samples` of the current test set).
        - If `shuffle == True`, the test set will be shuffled before selecting the samples, hence the results will be different for each call. The parameter `n_skip_samples` is ignored in this case.
        - You can also shuffle the test set before calling this function via the `shuffle` method.
        `unsafe`: if `True`, the exceptions raised by the subject will not be caught, which might lead to the function crashing in the middle of the evaluation and all the results being lost.
        - The advantage of setting `unsafe = True` is that the samples can be passed to the subject in batches, which might be faster depending on the subject's implementation.
        [optional] `logger`: the logger to use for logging the exceptions raised by the subject.

        ### Returns
        ----------
        A tuple containing:
        - The list of inputs.
        - The list of targets.
        - The list of raw outputs.
        - The list of extracted solutions (via `extract_solution`).
        - The list of scores.
        - The aggregated score.
        - The information about the generation process given by the subjects.

        ### Raises
        ----------
        `ValueError`: if `aggregation_fn` is not supported.
        `AssertionError`: if `n_samples` is not `None` and is less than 1, or greater than the number of samples in the test set.
        `AssertionError`: if `n_skip_samples` is less than 0 or greater than the number of samples in the test set or than `n_samples`.
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
        assert (
            n_samples is None or n_skip_samples < n_samples
        ), "n_skip_samples must be less than n_samples."
        assert (
            0 <= n_skip_samples < self.test_len
        ), "n_skip_samples must be >= 0 and < len(test_set)."

        if instructed:
            test_set = self.get_instructed()
        else:
            test_set = self.test_set

        if n_samples is not None:
            if shuffle:
                test_set = test_set.sample(n_samples)
            else:
                test_set = test_set[n_skip_samples:n_samples]

        inputs = test_set.inputs
        targets = test_set.targets
        assert len(inputs) == len(
            targets
        ), "the number of inputs and targets must be the same."

        if unsafe:
            outputs, info = subject(list(inputs))
            scores, found_solutions = zip(
                *[
                    self.evaluate_output(o, t, metric, logger)
                    for o, t in zip(outputs, targets)
                ]
            )
        else:
            outputs, info = [], None
            scores, found_solutions = [], []
            for i in range(len(inputs)):
                if logger is not None:
                    logger.info(
                        f"[Benchmark.evaluate_subject] Evaluating sample {i+1+n_skip_samples}/{len(inputs)+n_skip_samples}"
                    )
                try:
                    ind_outputs, ind_info = subject([inputs[i]])
                    output = "" if ind_outputs[0] is None else ind_outputs[0]

                    score, found_solution = self.evaluate_output(
                        output, targets[i], metric, logger
                    )
                    if logger is not None:
                        logger.info(
                            {
                                f"[Benchmark.evaluate_subject] Evaluated sample {i+1+n_skip_samples}/{len(inputs)+n_skip_samples}": {
                                    "Input": inputs[i],
                                    "Output": output,
                                    "Target": targets[i],
                                    "Found solution": found_solution,
                                    "Score": score,
                                }
                            }
                        )
                except Exception:
                    if logger is not None:
                        logger.error(
                            {
                                "[Benchmark.evaluate_subject] Exception raised while evaluating the subject": traceback.format_exc(),
                                "Input": inputs[i],
                                "Corrective action": "The stats for this sample are ignored, the output and found solution are set to an empty string and the score will be 0.0.",
                            }
                        )
                    outputs.append("")
                    scores.append(0.0)
                    found_solutions.append("")
                    if info is not None:
                        info += None
                else:
                    scores.append(score)
                    outputs.append(output)
                    found_solutions.append(found_solution)
                    if info is None:
                        info = ind_info
                    else:
                        info += ind_info

        assert (
            len(outputs) == len(inputs)
        ), "the number of output strings returned by the subject must be the same as the number of input strings."

        scores = np.array(scores)
        if aggregation_fn not in NumberListOperation:
            if logger is not None:
                logger.error(
                    {
                        "[Benchmark.evaluate_subject] Unsupported aggregation function": str(
                            aggregation_fn
                        ),
                        "Corrective action": "`np.mean` will be used as the fallback aggregation operation.",
                    }
                )
            agg_score = float(np.mean(scores))
        else:
            agg_score = aggregation_fn(scores)

        return inputs, targets, outputs, found_solutions, scores, agg_score, info

    def evaluate_output(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric,
        logger: Logger | None = None,
    ) -> tuple[float, str]:
        """
        Evaluate a single (input, output) pair.

        ### Parameters
        ----------
        `output`: the output of the subject.
        `target`: the target output.
        `metric`: the evaluation metric.
        [optional] `logger`: the logger to use for logging the exceptions raised by the subject.

        ### Returns
        -------
        A tuple containing the score of the solution and the found solution (via `extract_solution`).

        ### Notes
        ----------
        - `extract_solution` is used internally to extract the solution from the output and format it into the `target` format, hence you don't need to perform this step before calling this function.
        """

        found_solution = self.extract_solution([output], [target])[0]
        score = self._evaluate_output_impl(found_solution, target, metric, logger)

        return score, found_solution

    @abstractmethod
    def _evaluate_output_impl(
        cls,
        output: str,
        target: str,
        metric: EvaluationMetric,
        logger: Logger | None = None,
    ) -> float:
        """
        The child class' internal implementation of `evaluate_output`.
        It can be assumed that `output` is an extracted solution via `extract_solution`, i.e., it follows the format of the `target`.
        It is recommended for the scores to be in the range of [0.0, 1.0] and to increase linearly with the quality of the results.

        ### Parameters
        ----------
        `output`: the output of the subject.
        `target`: the target output.
        `metric`: the evaluation metric.
        [optional] `logger`: the logger to use for logging the exceptions raised by the subject.

        ### Returns
        -------
        The score of the output.
        """

        ...

    @overload
    def extract_solution(self, output: list[str], target: list[str]) -> list[str]:
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

    @overload
    def extract_solution(self, output: str, target: str) -> str:
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

    def extract_solution(self, output, target) -> str | list[str]:
        if isinstance(output, str):
            return self._extract_solution_impl(output, target)

        assert len(output) == len(
            target
        ), "the number of outputs and targets must be the same."

        return [self._extract_solution_impl(o, t) for o, t in zip(output, target)]

    @abstractmethod
    def _extract_solution_impl(self, output: str, target: str) -> str:
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


class ComputableBenchmark(Benchmark, ABC):
    @classmethod
    @abstractmethod
    def compute_target(cls, input: str, **kwargs: Any) -> str:
        """
        Compute the target output for the uninstructed input.
        If the input is instructed, you must use `get_uninstructed` before passing it to this function.

        ### Parameters
        ----------
        `input`: the uninstructed input string.
        Other keyword arguments may be required by each specific benchmark.

        ### Returns
        ----------
        The target output.
        """

        ...
