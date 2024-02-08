import numpy as np

from ldata.benchmark import Benchmark


class LetterConcatenation(Benchmark):
    """
    Benchmark for the letter concatenation task.
    The evaluation metric is the number of correct characters in the output word.
    The range of score values is [0.0, 1.0].
    """

    def _evaluate_impl(self, output: str, target: str) -> float:
        tot_score = np.sum(
            [
                0.0 if i >= len(target) else float(output[i] == target[i])
                for i in range(len(output))
            ]
        )
        return tot_score / len(output)
