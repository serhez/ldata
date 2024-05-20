import re
from dataclasses import dataclass

import numpy as np
from utils import NumberListOperation, SortingOrder

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class SortedListResults(Benchmark):
    """
    Benchmark for the sorted list results task, where the results of `Config.operation` on a list of numbers are ordered in `Config.order`.

    Uninstructed input example: `[1 2 3 4 5][6 7 8 9 10][11 12 13 14 15]`.
    Instructed input example: `Sum the items in each list and sort the results in descending order: [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15].`.
    Target output example: `65 40 15`.

    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    _LIST_PATTERN = re.compile("\[.*?\]")
    _INSTRUCTIONS_TEMPLATE = "{operation} the items in each list and sort the results in {order} order: {lists}."
    _MIN_INT = 1
    _MAX_INT = 100

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the sorted list results benchmark."""

        name: str = "SortedListResults"
        """The name of the benchmark."""

        operation: NumberListOperation = NumberListOperation.SUM
        """The operation to perform on the list of numbers."""

        order: SortingOrder = SortingOrder.ASCENDING
        """The order in which to sort the results."""

    @property
    def config_cls(self) -> type[Config]:
        return SortedListResults.Config

    def __init__(self, config: Config):
        """
        Initialize the sorted list results benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)
        self._config: SortedListResults.Config  # pyright is too dumb for this

    @classmethod
    def compute_target(
        cls, input: str, operation: NumberListOperation, order: SortingOrder
    ) -> str:
        # Get each list of numbers from the input
        lists = []
        curr_list = []
        curr_num = ""
        for c in input:
            if c == "[":  # Start of a new list
                curr_list = []
            elif c == "]":  # End of the current list
                curr_list.append(int(curr_num))
                lists.append(curr_list)
                curr_num = ""
            elif c == " ":  # End of the current number
                curr_list.append(int(curr_num))
                curr_num = ""
            else:  # Part of the current number
                curr_num += c

        # Perform the operation on each list
        op_results = [operation(list) for list in lists]

        # Sort the results
        if order == SortingOrder.ASCENDING:
            op_results.sort()
        else:
            op_results.sort(reverse=True)

        return " ".join([str(result) for result in op_results])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_lists: int,
        n_list_items: int,
        operation: NumberListOperation,
        order: SortingOrder,
    ):
        """
        Build the sorted list results benchmark.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_lists`: the number of lists in each sample.
        `n_list_items`: the number of items in each list.
        `operation`: the operation to perform on the list of numbers.
        `order`: the order in which to sort the results.
        """

        # Generate the samples
        samples = [
            "".join(
                [
                    f"[{' '.join([str(np.random.randint(cls._MIN_INT, cls._MAX_INT)) for _ in range(n_list_items)])}]"
                    for _ in range(n_lists)
                ]
            )
            for _ in range(n_samples)
        ]

        # Generate the targets
        targets = [cls.compute_target(sample, operation, order) for sample in samples]

        # Write the samples and targets to a csv file
        with open(path, "w") as file:
            file.write("SAMPLE,TARGET\n")
            for sample, target in zip(samples, targets):
                file.write(f"{sample},{target}\n")

    def get_instructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            # Re-format the sample to include commas for better readability
            search = re.search(self._LIST_PATTERN, sample)
            if search is None:
                return sample
            lists = search.group(1).split("], [")
            lists = [list_.replace(" ", ", ") for list_ in lists]
            sample = sample.replace(search.group(1), ", ".join(lists))

            return self._INSTRUCTIONS_TEMPLATE.format(
                operation=str(self._config.operation),
                order=self._config.order.value,
                lists=sample,
            )

        # Re-format the inputs to include commas for better readability
        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            search = re.search(self._LIST_PATTERN, s)
            if search is None:
                inputs[i] = s
            else:
                lists = search.group(1).split("], [")
                lists = [list_.replace(" ", ", ") for list_ in lists]
                inputs[i] = s.replace(search.group(1), ", ".join(lists))

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(
                    operation=str(self._config.operation),
                    order=self._config.order.value,
                    lists=lists,
                )
                for lists in inputs
            ]
        )

        return Dataset.Split(inputs, sample.targets)

    def get_uninstructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            # Extract the lists from the sample
            parsed = re.findall(self._LIST_PATTERN, sample)

            # Re-format the lists to remove commas for better readability
            return "".join([list_.replace(", ", " ") for list_ in parsed])

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            # Extract the lists from the input
            parsed = re.findall(self._LIST_PATTERN, s)

            # Re-format the inputs to remove commas
            inputs[i] = "".join([list_.replace(", ", " ") for list_ in parsed])

        return Dataset.Split(inputs, sample.targets)

    @classmethod
    def _evaluate_output_impl(
        cls,
        output: str,
        target: str,
        evaluation_method: Benchmark.Evaluation = Benchmark.Evaluation.CHARACTER,
    ) -> float:
        if evaluation_method == Benchmark.Evaluation.EXACT:
            return float(output == target)
        elif evaluation_method == Benchmark.Evaluation.WORD:
            output_list = []
            for w in output.split(" "):
                try:
                    output_list.append(int(w))
                except ValueError:
                    output_list.append(None)
            target_list = [int(w) for w in target.split(" ")]
            return float(
                np.mean(
                    [
                        float(output_list[i] == target_list[i])
                        if output_list[i] is not None
                        else 0.0
                        for i in range(min(len(output_list), len(target_list)))
                    ]
                    + [0.0] * abs(len(output_list) - len(target_list))
                )
            )
        elif evaluation_method == Benchmark.Evaluation.CHARACTER:
            return float(
                np.mean(
                    [
                        float(output[i] == target[i])
                        for i in range(min(len(output), len(target)))
                    ]
                    + [0.0] * abs(len(output) - len(target))
                )
            )
        else:
            raise ValueError(
                f"Invalid evaluation method: {evaluation_method}. Must be one of {Benchmark.Evaluation}."
            )

    @classmethod
    def _extract_solution_impl(cls, output: str, target: str) -> str:
        # Replace all separators with spaces
        for sep in [",", ";", ":", "|"]:
            output = output.replace(sep, " ")

        # Remove double spaces
        output = " ".join(output.split())

        # Remove all characters but for digits and spaces
        output = "".join([c for c in output if c.isdigit() or c == " "]).strip()
        numbers = output.split(" ")

        # Find the longest sequence of numbers that are in the target list
        best_match = []
        best_score = 0
        for s in range(len(numbers)):
            for e in range(s + 1, len(numbers) + 1):
                current_match = numbers[s:e]
                current_score = cls._evaluate_output_impl(
                    " ".join(current_match),
                    target,
                    Benchmark.Evaluation.CHARACTER,
                )
                if current_score > best_score:
                    best_match = current_match
                    best_score = current_score

        return " ".join(best_match)
