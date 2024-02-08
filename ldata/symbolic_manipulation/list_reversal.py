import numpy as np

from ldata import Benchmark


class ListReversal(Benchmark):
    """
    Benchmark for the list reversal task.
    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

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
