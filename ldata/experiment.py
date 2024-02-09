from typing import Any, Callable, List, Optional

import numpy as np

from ldata.benchmark import Benchmark


class Experiment:
    """An experiment allows you to compare a list of methods/models against a list of benchmarks and to report the results."""

    def __init__(
        self,
        subjects: List[Callable[[str], str]],
        benchmarks: List[Benchmark],
        default_aggregation_method: Benchmark.AggregationMethod = Benchmark.AggregationMethod.MEAN,
    ):
        """
        Initialize the experiment.

        ### Parameters
        ----------
        `subjects`: the list of subjects to compare.
        `benchmarks`: the list of benchmarks to use.
        `default_aggregation_method`: the default method to aggregate the scores given by the benchmarks for each the (input, output) pairs.
        - `AggregationMethod.NONE` is not supported.

        ### Raises
        `ValueError`: if `AggregationMethod.NONE` is used as the default aggregation method.
        """

        if default_aggregation_method == Benchmark.AggregationMethod.NONE:
            raise ValueError(
                "The default aggregation method for an experiment cannot be `AggregationMethod.NONE`."
            )

        self._subjects = subjects
        self._benchmarks = benchmarks
        self._default_aggregation_method = default_aggregation_method

    def run(
        self, aggregation_method: Optional[Benchmark.AggregationMethod] = None
    ) -> np.ndarray[float]:
        """
        Runs the experiment.

        ### Parameters
        ----------
        `aggregation_method`: the method to aggregate the scores given by the benchmarks for each the (input, output) pairs.
        - If `None`, the default aggregation method is used.

        ### Returns
        ----------
        A matrix of scores, where the rows correspond to the subjects and the columns correspond to the benchmarks.

        ### Raises
        ----------
        `ValueError`: if `AggregationMethod.NONE` is used as the aggregation method.
        """

        if aggregation_method == Benchmark.AggregationMethod.NONE:
            raise ValueError(
                "The aggregation method for an experiment cannot be `AggregationMethod.NONE`."
            )

        if aggregation_method is None:
            aggregation_method = self._default_aggregation_method

        results = np.zeros((len(self._subjects), len(self._benchmarks)))
        for i, subject in enumerate(self._subjects):
            for j, benchmark in enumerate(self._benchmarks):
                results[i, j] = benchmark.evaluate(subject, aggregation_method)

        return results

    def _call_impl(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    # TODO: Add caption including aggregation method used.
    @staticmethod
    def latex_table(
        results: np.ndarray[float],
        subject_names: List[str],
        benchmark_names: List[str],
        decimals: int = 3,
    ) -> str:
        """
        Designs a latex table containing the results of an experiment.

        ### Parameters
        ----------
        `results`: the results of an experiment.
        `subject_names`: the names of the subjects referenced in the table.
        `benchmark_names`: the names of the benchmarks referenced in the table.
        `decimals`: the number of decimal places with which to display floating point numbers.

        ### Returns
        ----------
        The latex table.
        """

        header = " & ".join([benchmark_name for benchmark_name in benchmark_names])
        header = f"Subject & {header} \\\\"
        header = (
            f"\\begin{{tabular}}{{c|{len(benchmark_names) * 'c'}}}\n{header}\n\\hline"
        )

        rows = []
        for i, subject_name in enumerate(subject_names):
            row = f"{subject_name} & {' & '.join([f'{score:.{decimals}f}' for score in results[i]])} \\\\"
            rows.append(row)

        return "{}\n{}\n{}".format(header, "\\hline\n".join(rows), "\\end{tabular}")
