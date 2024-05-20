import re
from dataclasses import dataclass

import numpy as np
from utils import SortingOrder

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class NumberListSorting(Benchmark):
    """
    Benchmark for the number list sorting task.

    Uninstructed input example: `1 3 4 2 5`.
    Instructed input example: `Sort the items in the list [1, 3, 4, 2, 5] in ascending order.`.
    Target output example: `1 2 3 4 5`.

    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    _LIST_PATTERN = re.compile("\[.*?\]")
    _INSTRUCTIONS_TEMPLATE = "Sort the items in the list [{list_}] in {order} order."
    _MIN_INT = 1
    _MAX_INT = 100

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the sorted list results benchmark."""

        name: str = "NumberListSorting"
        """The name of the benchmark."""

        order: SortingOrder = SortingOrder.ASCENDING
        """The order in which to sort the list."""

    @property
    def config_cls(self) -> type[Config]:
        return NumberListSorting.Config

    def __init__(self, config: Config):
        """
        Initialize the number list sorting benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)
        self._config: NumberListSorting.Config  # pyright is too dumb for this

    @classmethod
    def compute_target(cls, input: str, order: SortingOrder) -> str:
        # Get each list of numbers from the input
        numbers = [int(n) for n in input.split(" ")]

        # Sort the numbers
        if order == SortingOrder.ASCENDING:
            numbers.sort()
        else:
            numbers.sort(reverse=True)

        return " ".join([str(n) for n in numbers])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_list_items: int,
        order: SortingOrder,
    ):
        """
        Build the sorted list results benchmark.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_list_items`: the number of items in the list.
        `order`: the order in which to sort the numbers.
        """

        # Generate the samples
        samples = [
            " ".join(
                [
                    str(np.random.randint(cls._MIN_INT, cls._MAX_INT))
                    for _ in range(n_list_items)
                ]
            )
            for _ in range(n_samples)
        ]

        # Generate the targets
        targets = [cls.compute_target(sample, order) for sample in samples]

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
            return self._INSTRUCTIONS_TEMPLATE.format(
                order=self._config.order.value,
                list_=sample.replace(" ", ", "),
            )

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(
                    order=self._config.order.value,
                    list_=list_.replace(" ", ", "),
                )
                for list_ in sample.inputs
            ]
        )

        return Dataset.Split(inputs, sample.targets)

    def get_uninstructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            # Extract the list from the sample
            parsed = re.search(self._LIST_PATTERN, sample)
            if parsed is None:
                return sample

            return " ".join(parsed.group().split(", "))

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            parsed = re.search(self._LIST_PATTERN, s)
            if parsed is None:
                inputs[i] = s
            else:
                inputs[i] = " ".join(parsed.group().split(", "))

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
