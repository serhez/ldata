import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class LetterConcatenation(Benchmark):
    """
    Benchmark for the letter concatenation task.
    The evaluation metric is the number of correct characters in the output word.
    The range of score values is [0.0, 1.0].
    """

    _ALPHANUM_PATTERN = re.compile("[\W_]+")
    _INSTRUCTIONS_TEMPLATE = "Concatenate the characters at index {} of each word in the list [{}], in the same order they appear in the list. Indices start at zero."

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the letter concatenation benchmark."""

        name: str = "LetterConcatenation"
        """The name of the benchmark."""

        letter_idx: int
        """The character's index of the words to concatenate."""

    @property
    def config_cls(self) -> type[Config]:
        return LetterConcatenation.Config

    def __init__(self, config: Config):
        """
        Initialize the letter concatenation benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    @classmethod
    def compute_target(cls, input: str, letter_idx: int) -> str:
        return " ".join([word[letter_idx] for word in input.split(" ")])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_words: int,
        letter_idx: int,
        word_source: list[str] | Callable[[], list[str]],
    ):
        """
        Build the letter concatenation dataset.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_words`: the number of words in each sample.
        `letter_idx`: the index of the character to concatenate.
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
        targets = [cls.compute_target(sample, letter_idx) for sample in samples]

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
                self._config.letter_idx, ", ".join(sample.split(" "))
            )

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(
                    self._config.letter_idx, ", ".join(s.split(" "))
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
    def _evaluate_output_impl(
        cls,
        output: str,
        target: str,
        evaluation_method: Benchmark.Evaluation = Benchmark.Evaluation.CHARACTER,
    ) -> float:
        output = output.replace(" ", "")
        target = target.replace(" ", "")

        if (
            evaluation_method == Benchmark.Evaluation.EXACT
            or evaluation_method == Benchmark.Evaluation.WORD
        ):
            return float(output == target)

        # Evaluation.CHARACTER
        return float(
            np.mean(
                [
                    float(output[i] == target[i])
                    for i in range(min(len(output), len(target)))
                ]
                + [0.0] * abs(len(output) - len(target))
            )
        )

    @classmethod
    def extract_letter_idx(cls, instructed_sample: str) -> int:
        """
        Extract the index of the character to concatenate from an instructed input sample.

        ### Parameters
        ----------
        `instructed_sample`: the instructed input sample.

        ### Returns
        ----------
        The index of the character to concatenate.

        ### Raises
        ----------
        `ValueError`: if the input sample does not follow the instructed format of this benchmark.
        """

        try:
            words = instructed_sample.split(" ")
            idx = words.index("index")
            return int(words[idx + 1])
        except Exception:
            raise ValueError(
                "The input sample does not follow the instructed format of this benchmark."
            )

    @classmethod
    def _extract_solution_impl(cls, output: str, target: str) -> str:
        # Step 1: clean the output
        output = cls._ALPHANUM_PATTERN.sub("", output)

        # Step 2: find the sequence that best matches the target,
        #         either as a single word or as a concatenation of single-character words
        best_match = ""
        best_score = 0
        for s in range(len(output)):
            for e in range(s + 1, len(output) + 1):
                current_match = output[s:e]

                # Score the concatenated contiguous letters
                current_score = cls._evaluate_output_impl(
                    current_match, target, Benchmark.Evaluation.CHARACTER
                )
                if current_score > best_score:
                    best_match = current_match
                    best_score = current_score

        return best_match
