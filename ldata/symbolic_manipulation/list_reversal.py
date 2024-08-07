import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ldata.benchmark import Benchmark, ComputableBenchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.types import EvaluationMetric


class ListReversal(BuildableDataset, ComputableBenchmark):
    """
    Benchmark for the list reversal task.
    The evaluation metric is the number of correct element positions in the output list, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    _ALPHANUM_PATTERN = re.compile("[\W_]+")  # type: ignore[reportInvalidStringEscapeSequence]
    _INSTRUCTIONS_TEMPLATE = (
        "Reverse the order of the items in the following list: [{}]."
    )

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the list reversal benchmark."""

        name: str = "ListReversal"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return ListReversal.Config

    def __init__(self, config: Config):
        """
        Initialize the list reversal benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    @classmethod
    def compute_target(cls, input: str) -> str:
        return " ".join(input.split(" ")[::-1])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_words: int,
        word_source: list[str] | Callable[[], list[str]],
    ):
        """
        Build the list reversal benchmark.

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
            return self._INSTRUCTIONS_TEMPLATE.format(", ".join(sample.split(" ")))

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(", ".join(s.split(" ")))
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
            return " ".join(search.group(1).split(", "))

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            search = re.search(r"\[(.*?)\]", s)
            if search is None:
                inputs[i] = s
            else:
                inputs[i] = " ".join(search.group(1).split(", "))

        return Dataset.Split(inputs, sample.targets)

    @classmethod
    def _eval_word(cls, output_word: str, target_word: str) -> float:
        """
        Evaluate a word of the output for the list reversal task at the word level.

        ### Parameters
        ----------
        `output_word`: one of the words in the output list.
        `target_word`: the corresponding word in the target list.

        ### Returns
        ----------
        The evaluation score.
        """

        return float(output_word == target_word)

    @classmethod
    def _eval_char(cls, output_word: str, target_word: str) -> float:
        """
        Evaluate a word of the output for the list reversal task at the character level.

        ### Parameters
        ----------
        `output_word`: one of the words in the output list.
        `target_word`: the corresponding word in the target list.

        ### Returns
        ----------
        The evaluation score.
        """

        return float(
            np.mean(
                [
                    0.0
                    if i >= len(target_word)
                    else float(output_word[i] == target_word[i])
                    for i in range(min(len(output_word), len(target_word)))
                ]
                + [0.0] * abs(len(output_word) - len(target_word))
            )
        )

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric = EvaluationMetric.CHARACTER,
        _=None,
    ) -> float:
        if metric == EvaluationMetric.EXACT:
            return float(output == target)
        elif metric == EvaluationMetric.WORD:
            eval_fn = self._eval_word
        elif metric == EvaluationMetric.CHARACTER:
            eval_fn = self._eval_char
        else:
            raise NotImplementedError(
                f"Evaluation method {metric} is not implemented for this dataset."
            )

        output_list = output.split(" ")
        target_list = target.split(" ")

        return float(
            np.mean(
                [
                    eval_fn(output_list[i], target_list[i])
                    for i in range(min(len(output_list), len(target_list)))
                ]
                + [0.0] * abs(len(output_list) - len(target_list))
            )
        )

    def _extract_solution_impl(self, output: str, target: str) -> str:
        # Step 1: clean the output and split it into words
        words = [self._ALPHANUM_PATTERN.sub("", w) for w in output.split(" ")]
        words = [w for w in words if w != ""]

        # Step 2: find the longest sequence of words that are in the target list
        best_match = []
        best_score = 0
        for s in range(len(words)):
            for e in range(s + 1, len(words) + 1):
                current_match = words[s:e]
                current_score = self._evaluate_output_impl(
                    " ".join(current_match),
                    target,
                    EvaluationMetric.CHARACTER,
                )
                if current_score > best_score:
                    best_match = current_match
                    best_score = current_score

        return " ".join(best_match)
