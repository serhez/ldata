from __future__ import annotations

import os
import random
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass
from os import path
from typing import Any, Iterator, Sequence, Type, overload

import numpy as np
import numpy.typing as npt
import requests
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ldata.utils import DATASETS_API_URL, read_csv_columns


# TODO: Implement chaching the dataset into a file if coming from a remote server
# TODO: Implement paging to avoid loading the entire dataset into memory
class Dataset(TorchDataset):
    """A dataset which can be split into training and test sets."""

    @dataclass(kw_only=True)
    class Config:
        """The configuration of a dataset."""

        name: str
        """The name of the dataset used for reporting."""

        data_path: str
        """
        Either:
        - The system path of the CSV file containing the data.
        - The URL where the CSV file can be downloaded from (must include the protocol, e.g. `http://`).
        - The name of the Hugging Face dataset. In this case, the dataset will be downloaded from the Hugging Face Hub.
            - You need to specify your Hugging Face API token in the `HF_API_TOKEN` environment variable if the `data_path` is a Hugging Face dataset name.
        """

        inputs_column: str = "SAMPLE"
        """The name of the column containing the input data in the data source."""

        targets_column: str = "TARGET"
        """The name of the column containing the target data in the data source."""

        seed: int = 42
        """The random seed to use for all randomization purposes."""

        test_percentage: float = 0.2
        """Percentage of the dataset to use for testing."""

        n_shots: int = 0
        """
        The number of samples available at `Dataset.shots` for in-context learning purposes.
        The training set size will be reduced by `n` in order to create a shots set of size `n`.
        """

        shuffle: bool = True
        """Whether to shuffle the raw data before splitting it into training and testing sets."""

        chache: bool = False
        """Cache the full dataset into RAM for faster access at the cost of memory."""

    @property
    def config_cls(cls) -> type[Config]:
        """The configuration class for the dataset."""

        return cls.Config

    class Split:
        """A split of the dataset, which contains input and target data."""

        @overload
        def __init__(
            self,
            inputs: Sequence[Any] | npt.NDArray[Any],
            targets: Sequence[Any] | npt.NDArray[Any],
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

            ...

        @overload
        def __init__(
            self,
            data: Sequence[tuple[Any, Any]],
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `data`: the (input, target) data pairs.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            ...

        def __init__(self, *args, **_):
            if len(args) == 1:
                data = args[0]
                inputs, targets = zip(*data)
            else:
                inputs, targets = args

            if isinstance(inputs, list):
                inputs = np.array(inputs)
            if isinstance(targets, list):
                targets = np.array(targets)

            assert len(inputs) == len(
                targets
            ), "inputs and targets lists must have the same length."

            self._inputs = inputs
            self._targets = targets

        @property
        def inputs(self) -> npt.NDArray[Any]:
            """The input data."""

            return self._inputs

        @property
        def targets(self) -> npt.NDArray[Any]:
            """The target data."""

            return self._targets

        @property
        def raw(self) -> list[tuple[Any, Any]]:
            """The raw (input, target) data pairs."""

            return list(zip(self._inputs, self._targets))

        def __len__(self) -> int:
            return len(self._inputs)

        @overload
        def __getitem__(self, idx: int) -> tuple[Any, Any]: ...

        @overload
        def __getitem__(
            self, idx: list[int] | npt.NDArray[np.int_] | slice
        ) -> Dataset.Split: ...

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return (self._inputs[idx], self._targets[idx])
            return Dataset.Split(self._inputs[idx], self._targets[idx])

        def __iter__(self) -> Iterator[tuple[Any, Any]]:
            return iter(zip(self._inputs, self._targets))

        def sample(self, n: int = 1, replace: bool = False) -> Dataset.Split:
            """
            Get a random sample of the split.

            ### Parameters
            ----------
            `n`: the number of samples to get.
            `replace`: whether to sample with replacement.

            ### Returns
            ----------
            A `Split` containing a random sample.
            """

            idxs = np.random.choice(len(self), n, replace=replace)
            sample = self[idxs]

            if isinstance(sample, Dataset.Split):
                return sample

            return Dataset.Split(np.array([sample[0]]), np.array([sample[1]]))

    def __init__(
        self,
        config: Config,
        input_dtype: Type[Any] = str,
        target_dtype: Type[Any] = str,
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
        `ValueError`: if the test percentage is not between 0.0 and 1.0.
        """

        if config.test_percentage < 0 or config.test_percentage > 1:
            raise ValueError("Test percentage must be between 0.0 and 1.0.")

        self._config = config
        self._input_dtype = input_dtype
        self._target_dtype = target_dtype
        self._transform = transform
        self._target_transform = target_transform

        self.set_seed(config.seed)

        if self._config.shuffle:
            self.shuffle()
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

    def set_seed(self, value: int):
        """
        Set the random seed for the dataset.

        ### Parameters
        ----------
        `value`: the random seed to use.
        """

        self._config.seed = value
        np.random.seed(value)
        random.seed(value)

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

        inputs, targets = self._read_source(
            self._config.data_path,
            [self._config.inputs_column, self._config.targets_column],
        )

        return self.Split(inputs, targets)

    @property
    def train_set(self) -> Split:
        """The training set."""

        train_set = self.full_set[self._train_idxs]

        if isinstance(train_set, Dataset.Split):
            return train_set

        return Dataset.Split(np.array([train_set[0]]), np.array([train_set[1]]))

    @property
    def test_set(self) -> Split:
        """The test set."""

        test_set = self.full_set[self._test_idxs]

        if isinstance(test_set, Dataset.Split):
            return test_set

        return Dataset.Split(np.array([test_set[0]]), np.array([test_set[1]]))

    @property
    def shots(self) -> Split:
        """The shots used for in-context learning."""

        shots = self.full_set[self._shots_idxs]

        if isinstance(shots, Dataset.Split):
            return shots

        return Dataset.Split(np.array([shots[0]]), np.array([shots[1]]))

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

    @overload
    def __getitem__(self, idx: int) -> tuple[Any, Any]: ...

    @overload
    def __getitem__(
        self, idx: list[int] | npt.NDArray[np.int_] | slice
    ) -> Dataset.Split: ...

    def __getitem__(self, idx) -> tuple[Any, Any] | Dataset.Split:
        return self.full_set[idx]

    def __iter__(self) -> Dataset.Split:
        return self.full_set

    def _read_source(
        self, source: str, columns: list[str]
    ) -> tuple[npt.NDArray[Any], ...]:
        """
        Reads the data from the source.
        The source can be a file path, a URL (including the protocol), or a Hugging Face dataset name.

        ### Parameters
        ----------
        `source`: the source of the data.
        `columns`: the columns to read from the data source.

        ### Returns
        ----------
        A tuple of the data from the requested columns.

        ### Raises
        ----------
        `ValueError`: if the data path is invalid for any of the supported sources.
        `ValueError`: if the given columns are not found in the dataset.
        `AssertionError`: if the source is infered to be a Hugging Face dataset name and the Hugging Face API token is not set in the `HF_API_TOKEN` environment variable.
        """

        columns_error_msg = f"Columns `{columns}` not found in the dataset. Check `Config.inputs_column` and `Config.targets_column`."
        common_error_msg = "Check `Config.data_path` if you meant to load data from a file in your local system or hosted behind a URL."

        # File in the local system
        if path.exists(source):
            with open(self._config.data_path, "r") as file:
                data = read_csv_columns(
                    file, columns, dtype=self._input_dtype, error_msg=columns_error_msg
                )

        # File hosted behind a URL
        elif source.startswith("http"):
            with urllib.request.urlopen(source) as file:
                data = read_csv_columns(
                    file, columns, dtype=self._input_dtype, error_msg=columns_error_msg
                )

        # Hugging Face dataset
        else:
            # Check if the user's API token is set
            assert "HF_API_TOKEN" in os.environ, (
                "You must set the `HF_API_TOKEN` environment variable to load datasets from the Hugging Face Hub. "
                + common_error_msg
            )
            api_token = os.environ["HF_API_TOKEN"]

            # Check if the dataset name is valid
            url = f"{DATASETS_API_URL}/is-valid?dataset={source}"
            headers = {"Authorization": f"Bearer {api_token}"}

            def query():
                response = requests.get(url, headers=headers)
                return response.json()

            data = query()
            if data["viewer"] is False:
                raise ValueError(
                    f"The dataset `{source}` does not exist in the Hugging Face Hub or is not fully available to your account. "
                    + common_error_msg
                )

            # Load the dataset
            splits: DatasetDict = load_dataset(source)  # type: ignore[reportAssignmentType]
            dataset = concatenate_datasets([split for split in splits.values()])

            if any(column not in dataset.column_names for column in columns):
                raise ValueError(columns_error_msg)

            data = tuple(np.array(dataset[column]) for column in columns)

        return data

    def shuffle(self, seed: int | None = None):
        """
        Shuffle the dataset including the training set, testing set and shots.

        ### Parameters
        ----------
        `seed`: the random seed to use for shuffling.
        - If `None`, the random seed will not be set.
        """

        if seed is not None:
            np.random.seed(seed)

        self._train_idxs = np.random.choice(
            len(self.full_set),
            int(len(self.full_set) * (1 - self._config.test_percentage)),
            replace=False,
        )
        self._shots_idxs = np.random.choice(
            self._train_idxs, self._config.n_shots, replace=False
        )
        self._train_idxs = np.setdiff1d(self._train_idxs, self._shots_idxs)
        self._test_idxs = np.setdiff1d(np.arange(len(self.full_set)), self._train_idxs)


class BuildableDataset(Dataset):
    """A dataset object that can build its own data files."""

    @classmethod
    @abstractmethod
    def build(cls, path: str, n_samples: int, **kwargs: Any):
        """
        Build the dataset and save it to a file.

        ### Parameters
        ----------
        `path`: the path to save the dataset to.
        `n_samples`: the number of samples to generate.
        `**kwargs`: additional keyword arguments specific to the child class.
        """

        ...
