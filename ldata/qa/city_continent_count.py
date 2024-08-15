import random
import re
from dataclasses import dataclass

import numpy as np

from ldata.benchmark import Benchmark, ComputableBenchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.types import EvaluationMetric


class CityContinentCount(BuildableDataset, ComputableBenchmark):
    """
    Benchmark for the city counting per continent task, where a list of cities is given and the task is to count the number of cities per continent.
    """

    _CONTINENTS = [
        "africa",
        "asia",
        "europe",
        "north america",
        "oceania",
        "south america",
    ]
    _CONTINENTS_ALT_NAMES = {
        "africa": ["africa"],
        "asia": ["asia"],
        "europe": ["europe"],
        "north america": [
            "north america",
            "north-america",
            "northamerica",
            "n. america",
            "n america",
        ],
        "oceania": ["oceania"],
        "south america": [
            "south america",
            "south-america",
            "southamerica",
            "s. america",
            "s america",
        ],
    }

    _ALPHANUM_PATTERN = re.compile("[\W_]+")  # type: ignore[reportInvalidStringEscapeSequence]
    _INSTRUCTIONS_TEMPLATE = (
        "Count the number of cities per continent in the following list: [{}]. "
        "The list of continents is [Africa, Asia, Europe, North America, Oceania, South America]. "
        'Answer in the following format: "Africa: <count>, Asia: <count>, Europe: <count>, North America: <count>, Oceania: <count>, South America: <count>". '
        "You can omit continents if their count is 0."
    )

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the city counting per continent benchmark."""

        name: str = "CityContinentCount"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return CityContinentCount.Config

    def __init__(self, config: Config):
        """
        Initialize the city counting per continent benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

        # pyright is too dumb to infer this...
        self._config: CityContinentCount.Config

    @classmethod
    def _build_map(cls, cities_data_path: str) -> dict[str, str]:
        """
        Build a dictionary of city-continent pairs from a CSV file.

        ### Parameters
        ----------
        `cities_data_path`: path to a CSV file containing two columns: `City` and `Continent`, in that order.

        ### Returns
        ----------
        A dictionary of city-continent pairs.
        """

        city_continent_map = {}
        with open(cities_data_path, "r") as file:
            next(file)
            for line in file:
                city, continent = line.strip().split(",")
                city_continent_map[city] = continent.lower()

        return city_continent_map

    @classmethod
    def compute_target(cls, input: str, city_continent_map: dict[str, str]) -> str:
        counts = {continent: 0 for continent in cls._CONTINENTS}
        for city in input.split(";"):
            if city in city_continent_map:
                counts[city_continent_map[city]] += 1
            else:
                raise ValueError(
                    f"City '{city}' not found in the internal city-continent map. Perhaps the given `cities_data_path` is incorrect."
                )

        return " ".join([str(counts[continent]) for continent in cls._CONTINENTS])

    def get_instructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            return self._INSTRUCTIONS_TEMPLATE.format(", ".join(sample.split(";")))

        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(", ".join(s.split(";")))
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
            return ";".join([w.strip() for w in search.group(1).split(",")])

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            search = re.search(r"\[(.*?)\]", s)
            if search is None:
                inputs[i] = s
            else:
                inputs[i] = ";".join([w.strip() for w in search.group(1).split(",")])

        return Dataset.Split(inputs, sample.targets)

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric = EvaluationMetric.CHARACTER,
        _=None,
    ) -> float:
        """
        Evaluate the output against the target.

        ### Parameters
        ----------
        `output`: the output string.
        `target`: the target string.
        `metric`: the evaluation metric to use.

        ### Returns
        ----------
        The evaluation score.

        ### Notes
        ----------
        The evaluation score is computed as follows:
        - For `EvaluationMetric.EXACT`, the score is 1 if all counts are correct, 0 otherwise.
        - For `EvaluationMetric.WORD`, the score is the number of correct counts divided by the number of continents.
        - For `EvaluationMetric.CHARACTER`, the score is the average for all continents of the negated absolute difference between the output count and the target count, divided by the target count.
        """

        if metric == EvaluationMetric.EXACT:
            return 1 if output == target else 0

        if metric == EvaluationMetric.WORD:
            return sum(
                [1 for o, t in zip(output.split(" "), target.split(" ")) if o == t]
            ) / len(self._CONTINENTS)

        # EvaluationMetric.CHARACTER
        output_counts = [int(count) for count in output.split(" ")]
        target_counts = [int(count) for count in target.split(" ")]
        return sum(
            [
                -abs(output_count - target_count)
                / (target_count if target_count != 0 else 1)
                for output_count, target_count in zip(output_counts, target_counts)
            ]
        ) / len(self._CONTINENTS)

    def _extract_solution_impl(self, output: str, target: str) -> str:
        output = output.lower()
        counts = {continent: 0 for continent in self._CONTINENTS}

        for continent in self._CONTINENTS:
            for alt_name in self._CONTINENTS_ALT_NAMES[continent]:
                index = output.find(alt_name)
                if index != -1:
                    match = re.search(r"\d+", output[index:])
                    if match is not None:
                        counts[continent] = int(match.group())

        return " ".join([str(counts[continent]) for continent in self._CONTINENTS])

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_cities: int,
        cities_data_path: str,
    ):
        """
        Build the city counting per continent benchmark.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_cities`: the number of cities in each sample.
        `cities_data_path`: path to a CSV file containing two columns: `City` and `Continent`, in that order.
        """

        # Build a dictionary of city-continent pairs from the `cities_data_path`
        city_continent_map = cls._build_map(cities_data_path)

        # Create n_samples lists of n_cities cities chosen randomly from the list
        samples = [
            ";".join(random.sample(list(city_continent_map.keys()), n_cities))
            for _ in range(int(n_samples))
        ]

        # Generate the targets
        targets = [cls.compute_target(sample, city_continent_map) for sample in samples]

        # Write the samples and targets to a csv file
        with open(path, "w") as file:
            file.write("SAMPLE,TARGET\n")
            for sample, target in zip(samples, targets):
                file.write(f"{sample},{target}\n")
