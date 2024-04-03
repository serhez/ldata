from __future__ import annotations

from dataclasses import MISSING, dataclass
from os import path
from typing import Any, Generic, Iterable, Type, TypeVar

import numpy as np

# from torch.utils.data import Dataset as TorchDataset

_InputDType = TypeVar("_InputDType")
_TargetDType = TypeVar("_TargetDType")


# TODO: Allow the data file path to be in a remote server (e.g., a URL)
# TODO: Implement chaching the dataset into a file if coming from a remote server
# TODO: Implement paging to avoid loading the entire dataset into memory
# FIX: The `Split` class should be a subclass of `TorchDataset` to be able to use PyTorch's `DataLoader` class
class Dataset(Generic[_InputDType, _TargetDType]):  # , TorchDataset):
    """A dataset which can be split into training and test sets."""

    @dataclass(kw_only=True)
    class Config:
        """The configuration of a dataset."""

        name: str = MISSING
        """The name of the dataset used for reporting."""

        data_path: str = MISSING
        """The path to the `CSV` file containing the data."""

        test_percentage: float = 0.2
        """Percentage of the dataset to use for testing."""

        n_shots: int = 0
        """
        The number of samples available at `Dataset.shots` for in-context learning purposes.
        The training set size will be reduced by `n` in order to create a shots set of size `n`.
        """

        shuffle: bool = False
        """Whether to shuffle the dataset before splitting it into training and testing sets."""

        chache: bool = False
        """Cache the full dataset into RAM for faster access at the cost of memory."""

    class Split:
        """A split of the dataset, which contains input and target data."""

        def __init__(
            self,
            inputs: np.ndarray[_InputDType],
            targets: np.ndarray[_TargetDType | None],
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `inputs`: the input data.
            `targets`: the target data.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            assert len(inputs) == len(
                targets
            ), "inputs and targets lists must have the same length."

            self._inputs = inputs
            self._targets = targets

        @property
        def inputs(self) -> np.ndarray[_InputDType]:
            """The input data."""

            return self._inputs

        @property
        def targets(self) -> np.ndarray[_TargetDType | None]:
            """The target data."""

            return self._targets

        def __len__(self) -> int:
            return len(self._inputs)

        def __getitem__(
            self, idx: int | np.ndarray[int]
        ) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
            if isinstance(idx, int):
                return (self._inputs[idx], self._targets[idx])
            return Dataset.Split(self._inputs[idx], self._targets[idx])

        def __iter__(self) -> Iterable[tuple[_InputDType, _TargetDType | None]]:
            return zip(self._inputs, self._targets)

        def sample(
            self, n: int = 1, replace: bool = False
        ) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
            """
            Get a random sample of the split.

            ### Parameters
            ----------
            `n`: the number of samples to get.

            ### Returns
            ----------
            A `Split` containing a random sample.
            """

            idxs = np.random.choice(len(self), n, replace=replace)
            return self[idxs]

    def __init__(
        self,
        config: Config,
        input_dtype: Type[_InputDType] = str,
        target_dtype: Type[_TargetDType] = str,
        transform: Any = None,
        target_transform: Any = None,
    ):
        """
        Initialize the dataset.

        ### Parameters
        ----------
        `config`: the configuration for the dataset.
        `input_dtype`: the data type of the input data.
        `target_dtype`: the data type of the target data.
        `transform`: the PyTorch transform to apply to the input data.
        `target_transform`: the PyTorch transform to apply to the target data.

        ### Raises
        ----------
        `FileNotFoundError`: if the data file does not exist.
        `ValueError`: if the test percentage is not between 0.0 and 1.0.
        """

        if not path.isfile(config.data_path):
            raise FileNotFoundError(f"Data file '{config.data_path}' does not exist.")

        if config.test_percentage < 0 or config.test_percentage > 1:
            raise ValueError("Test percentage must be between 0.0 and 1.0.")

        self._config = config
        self._input_dtype = input_dtype
        self._target_dtype = target_dtype
        self._transform = transform
        self._target_transform = target_transform

        if self._config.shuffle:
            self._train_idxs = np.random.choice(
                len(self.full_set),
                int(len(self.full_set) * (1 - self._config.test_percentage)),
                replace=False,
            )
            self._shots_idxs = np.random.choice(
                self._train_idxs, self._config.n_shots, replace=False
            )
            self._train_idxs = np.setdiff1d(self._train_idxs, self._shots_idxs)
            self._test_idxs = np.setdiff1d(
                np.arange(len(self.full_set)), self._train_idxs
            )
        else:
            self._train_idxs = np.arange(
                int(len(self.full_set) * (1 - self._config.test_percentage))
            )
            self._shots_idxs = self._train_idxs[: self._config.n_shots]
            self._train_idxs = self._train_idxs[self._config.n_shots :]
            self._test_idxs = np.arange(
                int(len(self.full_set) * (1 - self._config.test_percentage)),
                len(self.full_set),
            )

        if self._config.chache:
            self._full_set = self.full_set

    @property
    def transform(self) -> Any:
        """The transform to apply to the input data."""

        return self._transform

    @property
    def target_transform(self) -> Any:
        """The transform to apply to the target data."""

        return self._target_transform

    @property
    def full_set(self) -> Split:
        """The full dataset."""

        # The full dataset is cached
        if hasattr(self, "_full_set"):
            return self._full_set

        with open(self._config.data_path, "r") as file:
            lines = file.readlines()[1:]
            inputs = np.array(
                [self._input_dtype(line.split(",")[0].strip()) for line in lines]
            )
            targets = np.array(
                [
                    self._target_dtype(t) if t else None
                    for t in [line.split(",")[1].strip() for line in lines]
                ]
            )

        return self.Split(inputs, targets)

    @property
    def train_set(self) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
        """The training set."""

        return self.full_set[self._train_idxs]

    @property
    def test_set(self) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
        """The test set."""

        return self.full_set[self._test_idxs]

    @property
    def shots(self) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
        """The shots used for in-context learning."""

        return self.full_set[self._shots_idxs]

    @property
    def train_len(self) -> int:
        """Length of the training set."""

        return len(self._train_idxs)

    @property
    def test_len(self) -> int:
        """Length of the test set."""

        return len(self._test_idxs)

    @property
    def shots_len(self) -> int:
        """Length of the shots set."""

        return len(self._shots_idxs)

    def __len__(self) -> int:
        """Length of the dataset."""

        return self.train_len + self.test_len

    def __getitem__(
        self, idx: int | np.ndarray[int]
    ) -> tuple[_InputDType, _TargetDType | None] | Dataset.Split:
        if isinstance(idx, int):
            return (self.full_set.inputs[idx], self.full_set.targets[idx])
        return Dataset.Split(self.full_set.inputs[idx], self.full_set.targets[idx])

    def __iter__(self) -> Dataset.Split:
        return Dataset.Split(self.full_set.inputs, self.full_set.targets)
