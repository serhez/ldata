import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ldata.benchmark import Benchmark, ComputableBenchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.types import EvaluationMetric, Separator


class ConcatReversal(BuildableDataset, ComputableBenchmark):
    """
    Benchmark for the concat-reversal task, where the first and last characters of each word in a list are concatenated using a delimiter and then the order of the resulting items are reversed.
    The evaluation metric is the number of correct element positions in the output list, as well as the correctness of the elements, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    _ALPHANUM_PATTERN = re.compile("[\W_]+")  # type: ignore[reportInvalidStringEscapeSequence]
    _INSTRUCTIONS_TEMPLATE = "Concatenate using a {} the first and last characters of each word in the list [{}] and then reverse the order of the resulting list."

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the concat-reversal benchmark."""

        name: str = "ConcatReversal"
        """The name of the benchmark."""

        separator: Separator = Separator.SPACE
        """The separator between the characters in each item of the list."""

    @property
    def config_cls(self) -> type[Config]:
        return ConcatReversal.Config

    def __init__(self, config: Config):
        """
        Initialize the concat-reversal benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

        # pyright is too dumb to infer this...
        self._config: ConcatReversal.Config

    @classmethod
    def compute_target(cls, input: str, separator: Separator = Separator.SPACE) -> str:
        return "  ".join(
            [separator.value.join([w[0], w[-1]]).lower() for w in input.split(" ")][
                ::-1
            ]
        )

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_words: int,
        word_source: list[str] | Callable[[], list[str]],
    ):
        """
        Build the concat-reversal benchmark.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_words`: the number of words in each sample.
        `word_source`: a list of words or a function that returns a list of words.
        """

        # Create a list of words
        if callable(word_source):
            all_words = word_source()
        else:
            all_words = word_source

        # Create n_samples lists of n_words words chosen randomly from the list
        samples = [
            " ".join(random.choices(all_words, k=n_words))
            for _ in range(int(n_samples))
        ]

        # Generate the targets
        targets = [cls.compute_target(sample) for sample in samples]

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
                self._config.separator.descriptor, ", ".join(sample.split(" "))
            )

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(
                    self._config.separator.descriptor, ", ".join(s.split(" "))
                )
                for s in sample.inputs
            ]
        )
        return Dataset.Split(inputs, sample.targets)

    def get_uninstructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            search = re.search(r"\[(.*?)\]", sample)
            if search is None:
                return sample
            return " ".join([w.strip() for w in search.group(1).split(",")])

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            search = re.search(r"\[(.*?)\]", s)
            if search is None:
                inputs[i] = s
            else:
                inputs[i] = " ".join([w.strip() for w in search.group(1).split(",")])

        return Dataset.Split(inputs, sample.targets)

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric = EvaluationMetric.CHARACTER,
        _=None,
    ) -> float:
        target = self._ALPHANUM_PATTERN.sub("", target).lower()

        if metric == EvaluationMetric.EXACT:
            return float(output == target)

        # EvaluationMetric.CHARACTER & EvaluationMetric.WORD
        return float(
            np.mean(
                [output[i] == target[i] for i in range(min(len(output), len(target)))]
                + [0.0] * abs(len(output) - len(target))
            )
        )

    def _extract_solution_impl(self, output: str, target: str) -> str:
        # Step 1: clean the output
        output = self._ALPHANUM_PATTERN.sub("", output).lower()

        # Step 2: find the longest sequence of words that are in the target list
        best_match = []
        best_score = 0
        for s in range(len(output)):
            for e in range(s + 1, len(output) + 1):
                current_match = output[s:e]
                current_score = self._evaluate_output_impl(
                    "".join(current_match),
                    target,
                    EvaluationMetric.CHARACTER,
                )
                if current_score > best_score:
                    best_match = current_match
                    best_score = current_score

        return "".join(best_match)
