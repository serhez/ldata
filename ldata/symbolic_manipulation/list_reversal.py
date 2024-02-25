import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class ListReversal(Benchmark):
    """
    Benchmark for the list reversal task.
    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the list reversal benchmark."""

        name: str = "ListReversal"
        """The name of the benchmark."""

    def __init__(self, config: Config):
        """
        Initialize the letter concatenation benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)
        self._alphanum_pattern = re.compile("[\W_]+")

    def get_instructed(
        self, sample: Optional[Union[str, Dataset.Split]] = None
    ) -> Union[str, Dataset.Split]:
        super().get_instructed(sample)

        if sample is None:
            sample = self.test_set

        instructions = "The task is to reverse the order of the items in a list. The list is [{}]. The output must be the reversed list."

        if isinstance(sample, str):
            return instructions.format(", ".join(sample.split(" ")))

        inputs = np.array(
            [instructions.format(", ".join(s.split(" "))) for s in sample.inputs]
        )
        return Dataset.Split(inputs, sample.targets)

    def _eval_word(self, output: str, target: str) -> float:
        return float(output == target)

    def _eval_char(self, output: str, target: str) -> float:
        return np.mean([float(output[i] == target[i]) for i in range(len(output))])

    def _evaluate_impl(
        self,
        output: str,
        target: str,
        evaluation_method: Benchmark.EvaluationMethod = Benchmark.EvaluationMethod.CHARACTER,
    ) -> float:
        if evaluation_method == Benchmark.EvaluationMethod.EXACT:
            return float(output == target)
        elif evaluation_method == Benchmark.EvaluationMethod.WORD:
            eval_fn = self._eval_word
        elif evaluation_method == Benchmark.EvaluationMethod.CHARACTER:
            eval_fn = self._eval_char
        else:
            raise NotImplementedError(
                f"Evaluation method {evaluation_method} is not implemented for this dataset."
            )

        output_list = output.split(" ")
        target_list = target.split(" ")

        tot_score = np.sum(
            [
                0.0
                if i >= len(target_list)
                else eval_fn(output_list[i], target_list[i])
                for i in range(len(output_list))
            ]
        )
        return tot_score / len(output_list)

    def _extract_solution_impl(
        self,
        output: str,
        target: str,
        evaluation_method: Benchmark.EvaluationMethod = Benchmark.EvaluationMethod.CHARACTER,
    ) -> str:
        target_list = target.split(" ")

        # Step 1: clean the output and split it into words
        words = [self._alphanum_pattern.sub("", w) for w in output.split(" ")]
        words = [w for w in words if w != ""]

        # Step 2: find the longest sequence of words that are in the target list
        current_match = []
        best_match = []
        best_score = 0
        for i in range(len(words)):
            end = i + len(target_list)
            if end >= len(words):
                current_match = words[i:]
            else:
                current_match = words[i:end]

            current_score = self._evaluate_impl(
                " ".join(current_match), target, evaluation_method
            )
            if current_score > best_score:
                best_match = current_match
                best_score = current_score

        return best_match.join(" ")


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_list_reversal", node=ListReversal.Config)
except ModuleNotFoundError:
    pass
