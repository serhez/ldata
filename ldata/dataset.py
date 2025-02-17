from __future__ import annotations

import os
import random
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass
from os import path
from typing import Any, Callable, Iterator, Sequence, Type, overload

import numpy as np
import numpy.typing as npt
import requests
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from ldata.utils import DATASETS_API_URL, read_csv_columns


# TODO: Implement chaching the dataset into a file if coming from a remote server
# TODO: Implement paging to avoid loading the entire dataset into memory
class Dataset:
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
        """A split of the dataset containing input and target data."""

        @overload
        def __init__(
            self,
            inputs: Sequence[Any] | npt.NDArray[Any],
            targets: Sequence[Any] | npt.NDArray[Any],
            transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            target_transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            cache_transforms: bool = True,
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `inputs`: the input data.
            `targets`: the target data.
            `transform`: a transformation to apply to the input data.
            `target_transform`: a transformation to apply to the target data.
            `cache_transforms`: whether to cache the transformed data.
            - Note that caching the transformed data results in the loss of the original data, thus using the `raw_*` properties will raise an error.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            ...

        @overload
        def __init__(
            self,
            data: Sequence[tuple[Any, Any]],
            transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            target_transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            cache_transforms: bool = True,
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `data`: the (input, target) data pairs.
            `transform`: a transformation to apply to the input data.
            `target_transform`: a transformation to apply to the target data.
            `cache_transforms`: whether to cache the transformed data.
            - Note that caching the transformed data results in the loss of the original data, thus using the `raw_*` properties will raise an error.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            ...

        def __init__(self, *args, **_):
            if len(args) == 4:
                data = args[0]
                inputs, targets = zip(*data)
                self._transform = args[1]
                self._target_transform = args[2]
                self._cache_transforms = args[3]
            else:
                inputs, targets = args
                self._transform = args[2]
                self._target_transform = args[3]
                self._cache_transforms = args[4]

            if isinstance(inputs, list):
                inputs = np.array(inputs)
            if isinstance(targets, list):
                targets = np.array(targets)

            assert len(inputs) == len(targets), (
                "inputs and targets lists must have the same length."
            )

            if self._cache_transforms:
                self._inputs = self._transform(inputs)
                self._targets = self._target_transform(targets)
            else:
                self._inputs = inputs
                self._targets = targets

        @property
        def inputs(self) -> npt.NDArray[Any]:
            """The transformed input data."""

            if self._cache_transforms:
                return self._inputs
            else:
                return self._transform(self._inputs)

        @property
        def raw_inputs(self) -> npt.NDArray[Any]:
            """
            The untransformed input data.

            ### Raises
            ----------
            `AttributeError`: if the transformed data has been cached (i.e., the raw data is not available).
            """

            if self._cache_transforms:
                raise AttributeError(
                    "The raw data is not available as the transformed data has been cached."
                )

            return self._inputs

        @property
        def targets(self) -> npt.NDArray[Any]:
            """The transformed target data."""

            if self._cache_transforms:
                return self._targets
            else:
                return self._target_transform(self._targets)

        @property
        def raw_targets(self) -> npt.NDArray[Any]:
            """
            The untransformed target data.

            ### Raises
            ----------
            `AttributeError`: if the transformed data has been cached (i.e., the raw data is not available).
            """

            if self._cache_transforms:
                raise AttributeError(
                    "The raw data is not available as the transformed data has been cached."
                )

            return self._targets

        @property
        def primitives(self) -> list[tuple[Any, Any]]:
            """The (transformed input, transformed target) data pairs."""

            if self._cache_transforms:
                inputs = self._inputs
                targets = self._targets
            else:
                inputs = self._transform(self._inputs)
                targets = self._target_transform(self._targets)

            return list(zip(self._transform(inputs), self._target_transform(targets)))

        @property
        def raw_primitives(self) -> list[tuple[Any, Any]]:
            """The (input, target) data pairs."""

            if self._cache_transforms:
                raise AttributeError(
                    "The raw data is not available as the transformed data has been cached."
                )

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
            inputs = self._inputs[idx]
            targets = self._targets[idx]
            if not self._cache_transforms:
                inputs = self._transform(inputs)
                targets = self._target_transform(targets)

            if isinstance(idx, int):
                return (inputs, targets)

            if self._cache_transforms:
                return Dataset.Split(self._inputs[idx], self._targets[idx])
            else:
                return Dataset.Split(
                    self._inputs[idx],
                    self._targets[idx],
                    self._transform,
                    self._target_transform,
                    False,
                )

        def __getitems__(self, idxs: list[int]) -> Dataset.Split:
            if self._cache_transforms:
                return Dataset.Split(self._inputs[idxs], self._targets[idxs])
            else:
                return Dataset.Split(
                    self._inputs[idxs],
                    self._targets[idxs],
                    self._transform,
                    self._target_transform,
                    False,
                )

        def __iter__(self) -> Iterator[tuple[Any, Any]]:
            inputs = self._inputs
            targets = self._targets
            if not self._cache_transforms:
                inputs = self._transform(inputs)
                targets = self._target_transform(targets)

            return iter(zip(inputs, targets))

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

        def to_pytorch(self) -> Dataset.PyTorchSplit:
            """
            Convert the split to a PyTorch-compatible split.
            This conversion should be memory-efficient as inputs and targets are copied by reference.
            """

            if self._cache_transforms:
                return Dataset.PyTorchSplit(self._inputs, self._targets)
            else:
                return Dataset.PyTorchSplit(
                    self._inputs,
                    self._targets,
                    self._transform,
                    self._target_transform,
                    False,
                )

    class PyTorchSplit(Split, TorchIterableDataset):
        """
        A PyTorch-compatible split of the dataset containing input and target data.
        The key difference w.r.t. `Split` is that the __getitem__ methods return the raw data, instead of a `Split` object.
        """

        @overload
        def __init__(
            self,
            inputs: Sequence[Any] | npt.NDArray[Any],
            targets: Sequence[Any] | npt.NDArray[Any],
            transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            target_transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            cache_transforms: bool = True,
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `inputs`: the input data.
            `targets`: the target data.
            `transform`: a transformation to apply to the input data.
            `target_transform`: a transformation to apply to the target data.
            `cache_transforms`: whether to cache the transformed data.
            - Note that caching the transformed data results in the loss of the original data, thus using the `raw_*` properties will raise an error.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            ...

        @overload
        def __init__(
            self,
            data: Sequence[tuple[Any, Any]],
            transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            target_transform: Callable[[npt.NDArray], npt.NDArray] = lambda x: x,
            cache_transforms: bool = True,
        ):
            """
            Initialize the dataset split.

            ### Parameters
            ----------
            `data`: the (input, target) data pairs.
            `transform`: a transformation to apply to the input data.
            `target_transform`: a transformation to apply to the target data.
            `cache_transforms`: whether to cache the transformed data.
            - Note that caching the transformed data results in the loss of the original data, thus using the `raw_*` properties will raise an error.

            ### Raises
            ----------
            `AssertionError`: if the inputs and targets lists have different lengths.
            """

            ...

        def __init__(self, *args, **_):
            super(Dataset.PyTorchSplit, self).__init__(*args)

        @overload
        def __getitem__(self, idx: int) -> tuple[Any, Any]: ...

        @overload
        def __getitem__(
            self, idx: list[int] | npt.NDArray[np.int_] | slice
        ) -> list[tuple[Any, Any]]: ...

        def __getitem__(self, idx):
            inputs = self._inputs[idx]
            targets = self._targets[idx]
            if not self._cache_transforms:
                inputs = self._transform(inputs)
                targets = self._target_transform(targets)

            if isinstance(idx, int):
                return (inputs, targets)

            return list(zip(inputs, targets))

        def __getitems__(self, idxs: list[int]) -> list[tuple[Any, Any]]:
            inputs = self._inputs[idxs]
            targets = self._targets[idxs]
            if not self._cache_transforms:
                inputs = self._transform(inputs)
                targets = self._target_transform(targets)

            return list(zip(inputs, targets))

    def __init__(
        self,
        config: Config,
        input_dtype: Type[Any] = str,
        target_dtype: Type[Any] = str,
        transform: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] = lambda x: x,
    ):
        """
        Initialize the dataset.

        ### Parameters
        ----------
        `config`: the configuration for the dataset.
        `input_dtype`: the data type of the input data.
        `target_dtype`: the data type of the target data.
        `transform`: the transformation to apply to the input data.

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
    def full_set(self) -> Split:
        """The full dataset."""

        # The full dataset is cached
        if hasattr(self, "_full_set"):
            return self._full_set

        inputs, targets = self._read_source(
            self._config.data_path,
            [self._config.inputs_column, self._config.targets_column],
        )

        return self.Split(inputs, targets, self._transform)

    @property
    def train_set(self) -> Split:
        """The training set."""

        train_set = self.full_set[self._train_idxs]

        if isinstance(train_set, Dataset.Split):
            return train_set

        return Dataset.Split(
            np.array([train_set[0]]), np.array([train_set[1]]), self._transform
        )

    @property
    def test_set(self) -> Split:
        """The test set."""

        test_set = self.full_set[self._test_idxs]

        if isinstance(test_set, Dataset.Split):
            return test_set

        return Dataset.Split(
            np.array([test_set[0]]), np.array([test_set[1]]), self._transform
        )

    @property
    def shots(self) -> Split:
        """The shots used for in-context learning."""

        shots = self.full_set[self._shots_idxs]

        if isinstance(shots, Dataset.Split):
            return shots

        return Dataset.Split(
            np.array([shots[0]]), np.array([shots[1]]), self._transform
        )

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
