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

    def get_instructed(
        self, sample: Optional[Union[str, Dataset.Split]] = None
    ) -> Union[str, Dataset.Split]:
        super().get_instructed(sample)

        if sample is None:
            sample = self.test_set

        instructions = "The task is to output the word resulting from the concatenation of the character with index {} of each word, where the words are [{}]."

        if isinstance(sample, str):
            return instructions.format(self._config.i, ", ".join(sample.split(" ")))

        inputs = np.array(
            [
                instructions.format(self._config.i, ", ".join(s.split(" ")))
                for s in sample.inputs
            ]
        )
        return Dataset.Split(inputs, sample.targets)

    def _evaluate_impl(self, output: str, target: str) -> float:
        tot_score = np.sum(
            [
                0.0 if i >= len(target) else float(output[i] == target[i])
                for i in range(len(output))
            ]
        )
        return tot_score / len(output)


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_letter_concatenation", node=LetterConcatenation.Config)
except ModuleNotFoundError:
    pass
