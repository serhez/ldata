from dataclasses import dataclass

import numpy as np

from ldata.benchmark import Benchmark


class ListReversal(Benchmark):
    """
    Benchmark for the list reversal task.
    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    @dataclass
    class Config(Benchmark.Config):
        """The configuration of the list reversal benchmark."""

        name = "ListReversal"
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
        output_list = output.split(" ")
        target_list = target.split(" ")

        tot_score = np.sum(
            [
                0.0
                if i >= len(target_list)
                else float(output_list[i] == target_list[i])
                for i in range(len(output_list))
            ]
        )
        return tot_score / len(output_list)
