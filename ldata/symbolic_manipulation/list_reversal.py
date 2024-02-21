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

    def get_instructed(
        self, sample: Optional[Union[str, Dataset.Split]] = None
    ) -> Union[str, Dataset.Split]:
        super().get_instructed(sample)

        if sample is None:
            sample = self.test_set

        instructions = "The task is to reverse the order of the items in a list and output the resulting reversed list. The list is [{}]."

        if isinstance(sample, str):
            return instructions.format(", ".join(sample.split(" ")))

        inputs = np.array(
            [instructions.format(", ".join(s.split(" "))) for s in sample.inputs]
        )
        return Dataset.Split(inputs, sample.targets)

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


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_list_reversal", node=ListReversal.Config)
except ModuleNotFoundError:
    pass
