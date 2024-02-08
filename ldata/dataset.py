from __future__ import annotations

from dataclasses import dataclass
from os import path
from typing import List, Optional

import numpy as np


# TODO: Allow the data file path to be in a remote server (e.g., a URL)
# TODO: Implement chaching the dataset into a file if coming from a remote server
# TODO: Implement paging to avoid loading the entire dataset into memory
class Dataset:
    """A dataset which can be split into training and test sets."""

    @dataclass(kw_only=True)
    class Config:
        test_percentage: float = 0.2
        """Percentage of the dataset to use for testing."""

        shuffle: bool = False
        """Whether to shuffle the dataset before splitting it into training and testing sets."""

    class Split:
        """A split of the dataset, which contains input and target data."""

        def __init__(self, input: np.ndarray[str], target: np.ndarray[Optional[str]]):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `input`: the input data.
            `target`: the target data.

            ### Raises
            ----------
            `AssertionError`: if the input and target lists have different lengths.
            """

            assert len(input) == len(
                target
            ), "Input and target lists must have the same length."

            self._input = input
            self._target = target

        @property
        def input(self) -> np.ndarray[str]:
            """The input data."""

            return self._input

        @property
        def target(self) -> np.ndarray[Optional[str]]:
            """The target data."""

            return self._target

        def __len__(self) -> int:
            return len(self._input)

        def __getitem__(self, idx: int) -> Dataset.Split:
            return Dataset.Split(self._input[idx], self._target[idx])

        def __iter__(self) -> Dataset.Split:
            return Dataset.Split(zip(self._input, self._target))

    def __init__(self, data_path: str, config: Config):
        """
        Initialize the dataset.

        ### Parameters
        ----------
        `data_path`: the path to the data file.
        `config`: the configuration for the dataset.

        ### Raises
        ----------
        `FileNotFoundError`: if the data file does not exist.
        `ValueError`: if the test percentage is not between 0.0 and 1.0.
        """

        if not path.isfile(data_path):
            raise FileNotFoundError(f"Data file '{data_path}' does not exist.")

        if config.test_percentage < 0 or config.test_percentage > 1:
            raise ValueError("Test percentage must be between 0.0 and 1.0.")

        self._data_path = data_path
        self._config = config

        if self._config.shuffle:
            self._train_idxs = np.random.choice(
                len(self.full_set),
                int(len(self.full_set) * (1 - self._config.test_percentage)),
                replace=False,
            )
            self._test_idxs = np.setdiff1d(
                np.arange(len(self.full_set)), self._train_idxs
            )
        else:
            self._train_idxs = np.arange(
                int(len(self.full_set) * (1 - self._config.test_percentage))
            )
            self._test_idxs = np.arange(
                int(len(self.full_set) * (1 - self._config.test_percentage)),
                len(self.full_set),
            )

    @property
    def full_set(self) -> Split:
        """The full dataset."""

        with open(self._data_path, "r") as file:
            lines = file.readlines()[1:]
            input = np.array([line.split(",")[0].strip() for line in lines])
            target = np.array(
                [
                    t if t else None
                    for t in [line.split(",")[1].strip() for line in lines]
                ]
            )

        return self.Split(input, target)

    @property
    def train_set(self) -> Split:
        """The training set."""

        return self.full_set[self._train_idxs]

    @property
    def test_set(self) -> Split:
        """The test set."""

        return self.full_set[self._test_idxs]

    @property
    def train_len(self) -> int:
        """Length of the training set."""

        return len(self._train_idxs)

    @property
    def test_len(self) -> int:
        """Length of the test set."""

        return len(self._test_idxs)

    def __len__(self) -> int:
        """Length of the dataset."""

        return self.train_len + self.test_len
