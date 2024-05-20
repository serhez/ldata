import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class DoubleListReversal(Benchmark):
    """
    Benchmark for the double list reversal task, where the characters of each item in a list are reversed and then the order of the resulting items are reversed.
    The evaluation metric is the number of correct element positions in the output list, as well as the correctness of the reversed elements, normalized by the number of elements.
    The range of score values is [0.0, 1.0].
    """

    _ALPHANUM_PATTERN = re.compile("[\W_]+")
    _INSTRUCTIONS_TEMPLATE = "Reverse the characters of each item and then reverse the order of such items in the following list: [{}]."

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the double list reversal benchmark."""

        name: str = "DoubleListReversal"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return DoubleListReversal.Config

    def __init__(self, config: Config):
        """
        Initialize the double list reversal benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    @classmethod
    def compute_target(cls, input: str) -> str:
        return " ".join([w[::-1] for w in input.split(" ")][::-1])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_words: int,
        word_source: list[str] | Callable[[], list[str]],
    ):
        """
        Build the double list reversal benchmark.

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
    def _eval_word(cls, output: str, target: str) -> float:
        """
        Evaluate the output of the model for the double list reversal task at the word level.

        ### Parameters
        ----------
        `output`: the output of the model.
        `target`: the target output.

        ### Returns
        ----------
        The evaluation score.
        """

        word_scores = []
        output_words = output.split(" ")
        target_words = target.split(" ")
        for i in range(min(len(output), len(target))):
            word_scores.append(float(output_words[i] == target_words[i]))
        word_scores += [0.0] * abs(len(output_words) - len(target_words))
        return float(np.mean(word_scores))

    @classmethod
    def _eval_char(cls, output: str, target: str) -> float:
        """
        Evaluate the output of the model for the double list reversal task at the character level.

        ### Parameters
        ----------
        `output`: the output of the model.
        `target`: the target output.

        ### Returns
        ----------
        The evaluation score.
        """

        return float(
            np.mean(
                [
                    0.0 if i >= len(target) else float(output[i] == target[i])
                    for i in range(len(output))
                ]
            )
        )

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
            eval_fn = cls._eval_word
        elif evaluation_method == Benchmark.Evaluation.CHARACTER:
            eval_fn = cls._eval_char
        else:
            raise NotImplementedError(
                f"Evaluation method {evaluation_method} is not implemented for this dataset."
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

    @classmethod
    def _extract_solution_impl(cls, output: str, target: str) -> str:
        # Step 1: clean the output and split it into words
        words = [cls._ALPHANUM_PATTERN.sub("", w) for w in output.split(" ")]
        words = [w for w in words if w != ""]

        # Step 2: find the longest sequence of words that are in the target list
        best_match = []
        best_score = 0
        for s in range(len(words)):
            for e in range(s + 1, len(words) + 1):
                current_match = words[s:e]
                current_score = cls._evaluate_output_impl(
                    " ".join(current_match),
                    target,
                    Benchmark.Evaluation.CHARACTER,
                )
                if current_score > best_score:
                    best_match = current_match
                    best_score = current_score

        return " ".join(best_match)
