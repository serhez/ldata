from dataclasses import dataclass

import numpy as np

from ldata.benchmark import Benchmark


class LetterConcatenation(Benchmark):
    """
    Benchmark for the letter concatenation task.
    The evaluation metric is the number of correct characters in the output word.
    The range of score values is [0.0, 1.0].
    """

    @dataclass
    class Config(Benchmark.Config):
        """The configuration of the letter concatenation benchmark."""

        name = "LetterConcatenation"
        """The name of the benchmark."""

    def __init__(self, data_path: str, config: Config = Config()):
        """
        Initialize the letter concatenation benchmark.

        ### Parameters
        ----------
        `data_path`: the path to the data directory.
        `config`: the configuration of the benchmark.
        """

        super().__init__(data_path, config)

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
