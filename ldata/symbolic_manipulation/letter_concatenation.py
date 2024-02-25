import re
from dataclasses import MISSING, dataclass
from typing import Optional, Union

import numpy as np

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class LetterConcatenation(Benchmark):
    """
    Benchmark for the letter concatenation task.
    The evaluation metric is the number of correct characters in the output word.
    The range of score values is [0.0, 1.0].
    """

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the letter concatenation benchmark."""

        name: str = "LetterConcatenation"
        """The name of the benchmark."""

        i: int = MISSING
        """The character's index of the words to concatenate."""

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

        instructions = "The task is to concatenate the character with index {} of each input word, where the input words are [{}]. The output must be a single word."

        if isinstance(sample, str):
            return instructions.format(self._config.i, ", ".join(sample.split(" ")))

        inputs = np.array(
            [
                instructions.format(self._config.i, ", ".join(s.split(" ")))
                for s in sample.inputs
            ]
        )
        return Dataset.Split(inputs, sample.targets)

    def _evaluate_impl(
        self, output: str, target: str, evaluation_method: Benchmark.EvaluationMethod
    ) -> float:
        if (
            evaluation_method == Benchmark.EvaluationMethod.EXACT
            or evaluation_method == Benchmark.EvaluationMethod.WORD
        ):
            return float(output == target)

        # EvaluationMethod.CHARACTER
        tot_score = np.sum(
            [
                0.0 if i >= len(target) else float(output[i] == target[i])
                for i in range(len(output))
            ]
        )
        return tot_score / len(output)

    def _extract_solution_impl(self, output: str, target: str) -> str:
        # Step 1: clean the output and split it into words
        words = [self._alphanum_pattern.sub("", w) for w in output.split(" ")]
        words = [w for w in words if w != ""]

        # Step 2: find the word that best match the target, either as a whole or as a concatenation of characters
        concat_letters = ""
        best_match = ""
        best_score = 0
        for w in words:
            if len(w) == 1:
                # Add the letter to the concatenation
                concat_letters += w
            else:
                # Score the word
                current_score = self._evaluate_impl(w, target)
                if current_score > best_score:
                    best_match = w
                    best_score = current_score

                # Reset the concatenated contiguous letters
                concat_letters = ""
                continue

            # Score the concatenated contiguous letters
            current_score = self._evaluate_impl(concat_letters, target)
            if current_score > best_score:
                best_match = concat_letters
                best_score = current_score

        return best_match


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_letter_concatenation", node=LetterConcatenation.Config)
except ModuleNotFoundError:
    pass
