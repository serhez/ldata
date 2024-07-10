import itertools
import random
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ldata.benchmark import Benchmark, ComputableBenchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.method_integrations import RecursivePromptingCompatible
from ldata.types import EvaluationMetric, Separator


class LetterConcatenation(
    BuildableDataset, ComputableBenchmark, RecursivePromptingCompatible
):
    """
    Benchmark for the letter concatenation task.
    The evaluation metric is the number of correct characters in the output word.
    The range of score values is [0.0, 1.0].
    """

    _ALPHANUM_PATTERN = re.compile("[\W_]+")  # type: ignore[reportInvalidStringEscapeSequence]
    _INSTRUCTIONS_TEMPLATE = "Concatenate using a {} the characters at index {} of each word in the list [{}]; indices start at zero."

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the letter concatenation benchmark."""

        name: str = "LetterConcatenation"
        """The name of the benchmark."""

        letter_idx: int
        """The character's index of the words to concatenate."""

        separator: Separator = Separator.SPACE
        """The separator between the words in the list."""

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

        # pyright is too dumb to infer this...
        self._config: LetterConcatenation.Config

    @classmethod
    def compute_target(
        cls, input: str, letter_idx: int, separator: Separator = Separator.SPACE
    ) -> str:
        return separator.value.join(
            [str(word[letter_idx]).lower() for word in input.split(separator.value)]
        )

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
                self._config.separator.descriptor,
                self._config.letter_idx,
                ", ".join(sample.split(" ")),
            )

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(
                    self._config.separator.descriptor,
                    self._config.letter_idx,
                    ", ".join(s.split(" ")),
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

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric = EvaluationMetric.CHARACTER,
        _=None,
    ) -> float:
        output = self._ALPHANUM_PATTERN.sub("", output).lower()
        target = self._ALPHANUM_PATTERN.sub("", target).lower()

        if metric == EvaluationMetric.EXACT or metric == EvaluationMetric.WORD:
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

    def _extract_solution_impl(self, output: str, target: str) -> str:
        # Step 1: clean the output
        output_list = [
            self._ALPHANUM_PATTERN.sub("", w).lower() for w in output.split(" ")
        ]

        # Step 2: find the sequence that best matches the target,
        # either as a single word or as a concatenation of single-character words
        best_match = ""
        best_score = 0
        for s in range(len(output_list)):
            for e in range(s + 1, len(output_list) + 1):
                current_match = self._config.separator.value.join(output_list[s:e])

                # Attempt a variant with the substring " and " removed from the output
                current_match_no_ands = current_match.replace(" and ", " ")

                for match in [current_match, current_match_no_ands]:
                    # Score the concatenated contiguous letters
                    current_score = self._evaluate_output_impl(
                        match, target, EvaluationMetric.CHARACTER
                    )
                    if current_score > best_score:
                        best_match = match
                        best_score = current_score

        return best_match.lower()

    ## Recursive Prompting Integration

    def get_subproblems(
        self,
        sample: str,
        n_subproblems: int,
    ) -> list[tuple[str, str]]:
        # Split the input into `n_subproblems` sub-problems
        words = sample.split(self._config.separator.value)
        subproblems = [
            self._config.separator.value.join(
                words[i : i + len(words) // n_subproblems]
            )
            for i in range(0, len(words), len(words) // n_subproblems)
        ]

        # Instruct the sub-problems
        subproblems = [self.get_instructed(subproblem) for subproblem in subproblems]

        # Generate the sub-solutions
        subsolutions = [
            self.compute_target(subproblem, 0)  # type: ignore[reportArgumentType]
            for subproblem in subproblems
        ]

        return list(zip(subproblems, subsolutions))  # type: ignore[reportReturnType]

    def evaluate_split(
        self,
        input: str,
        split: list[str],
        n_subproblems: int | None = None,
    ) -> float:
        # Check there are `n_subproblems` subproblems
        if n_subproblems is not None and len(split) != n_subproblems:
            return 0.0

        input_list = [w.lower() for w in input.split(self._config.separator.value)]

        # Find the lists substrings in each string of the split
        split_words = [re.search(r"\[(.*?)\]", s) for s in split]
        if not all(split_words):
            return 0.0

        # Create actual lists from the list-strings
        split_lists = [
            [w.strip().lower() for w in s.group(1).split(",")]  # type: ignore[reportOptionalMemberAccess]
            for s in split_words
        ]

        # Concatenate the lists
        split_concat_list = list(itertools.chain.from_iterable(split_lists))

        return float(split_concat_list == input_list)

    def evaluate_merge(
        self,
        split: list[str],
        merged: str,
    ) -> tuple[float, str]:
        # Get the sub-problems' words
        split_words = [re.search(r"\[(.*?)\]", s) for s in split]
        if not all(split_words):
            return -1.0, ""

        # Create actual lists from the list-strings
        split_lists = [
            [w.strip().lower() for w in s.group(1).split(",")]  # type: ignore[reportOptionalMemberAccess]
            for s in split_words
        ]

        # Concatenate the lists
        split_concat_list = list(itertools.chain.from_iterable(split_lists))

        # Create the target
        target = self.compute_target(
            self._config.separator.value.join(split_concat_list), 0
        )

        return self.evaluate_output(merged, target, EvaluationMetric.EXACT)
