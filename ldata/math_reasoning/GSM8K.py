import os
from dataclasses import dataclass

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset

from ldata.benchmark import Benchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.protocols import Logger
from ldata.types import EvaluationMetric
from ldata.utils import write_csv


class GSM8K(BuildableDataset, Benchmark):
    """
    # The GSM8K benchmark.

    ## Description
    The GSM8K benchmark is a benchmark for evaluating the generalization of mathematical reasoning models.
    The problems consist of a question and an answer, where the answer is a floating point number. A rationale is also provided.

    ## Evaluation
    The evaluation scores are as follows:
    - `EvaluationMetric.EXACT`: if the numbers are equal to the same number of decimal places as the target, then 1.0 is returned; otherwise, 0.0 is returned.
    - Other metrics: the score will be the negated absolute difference beteween the numbers, divided by the magnitude of the target for normalization purposes.
    """

    _PENALTY = -1000
    """
    The evaluation score for non-exact metrics if no floating number is detected in the output.
    This value is approximately the 90th percentile of the targets.
    """

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the GSM8K benchmark."""

        name: str = "GSM8K"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return GSM8K.Config

    def __init__(self, config: Config):
        """
        Initialize the GSM8K benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    def get_instructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            return self.test_set

        return sample

    def get_uninstructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        return sample

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        metric: EvaluationMetric,
        logger: Logger | None = None,
    ) -> float:
        """
        Evaluate the output of the GSM8K benchmark.

        ### Parameters
        ----------
        `output`: the output to evaluate.
        `target`: the target output.
        `metric`: the evaluation method to use.
        [optional] `logger`: the logger to use for logging.

        ### Returns
        ----------
        The evaluation score.

        ### Notes
        ----------
        The metrics are interpreted as follows:
        - `EXACT`: the numbers must be equal to the same number of decimal places as the target.
        - Others: the negated absolute difference beteween the numbers will be the score, divided by the magnitude of the target for normalization.
            - If the output is not a number, the score will be -1,000.
        """

        # Cast the target to a float
        try:
            target_float = float(target)
        except ValueError:
            raise ValueError(f"Could not convert target to float: {target}")

        # Cast the output to a float
        try:
            output_float = float(output)
            output_float = round(output_float, len(str(target_float).split(".")[1]))
        except ValueError:
            if metric == EvaluationMetric.EXACT:
                return 0.0
            return self._PENALTY

        if metric == EvaluationMetric.EXACT:
            return 1.0 if output_float == target_float else 0.0

        return -(abs(output_float - target_float) / abs(target_float))

    def _extract_solution_impl(self, output: str, _) -> str:
        # Extract all floating point numbers
        floats_str = "".join(
            [c for c in output if c.isdigit() or c == "." or c == " " or c == "-"]
        )

        # Attempt to use the floats from last to first
        floats = [
            s.strip() if s.strip()[-1] != "." else s[:-1].strip()
            for s in floats_str.split()
        ][::-1]

        # Attempt to convert the result to a float
        for result_str in floats:
            try:
                return str(float(result_str))
            except ValueError:
                pass

        return ""

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Build the GSM8K dataset.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `shuffle`: whether to shuffle the dataset before sampling `n_samples`.
        `seed`: the random seed to use for shuffling the dataset.

        ### Raises
        ----------
        `AssertionError`: if the `HF_API_TOKEN` environment variable is not set.
        """

        assert (
            "HF_API_TOKEN" in os.environ
        ), "You must set the `HF_API_TOKEN` environment variable to build the APPS benchmark. "

        # Load the dataset
        splits: DatasetDict = load_dataset(
            "openai/gsm8k", "main", trust_remote_code=True
        )  # type: ignore[reportAssignmentType]
        dataset = concatenate_datasets([split for split in splits.values()])

        # Shuffle the dataset
        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        # Select the first `n_samples` samples
        dataset = dataset.select(range(n_samples))

        # Split solutions into sample rationales and final answers (targets)
        solutions = [e for e in dataset["answer"]]
        inputs = []
        targets = []
        rationales = []
        for i, solution in enumerate(solutions):
            if i % 1000 == 0:
                msg = f"[INFO] Processing sample {i}/{len(solutions)}"
                if i == 0:
                    msg += " (this is only printed every 1,000 samples)"
                print(msg)

            parts = [s.strip() for s in solution.split("####")]

            if len(parts) == 1:
                # Remove thousands separator (commas)
                parts[0] = parts[0].replace(",", "")

                try:
                    targets.append(float(parts[0]))
                    rationales.append("")
                    question = dataset["question"][i]
                    if question[0] == '"':
                        question = question[1:]
                    if question[-1] == '"':
                        question = question[:-1]
                    inputs.append(question)
                except ValueError:
                    print(
                        f"[WARNING] Ignoring sample. Could not convert target to float: {parts[0]}"
                    )
            else:
                # Remove thousands separator (commas)
                parts[1] = parts[1].replace(",", "")

                try:
                    targets.append(float(parts[1]))
                    rationale = parts[0]
                    if rationale[0] == '"':
                        rationale = rationale[1:]
                    if rationale[-1] == '"':
                        rationale = rationale[:-1]
                    rationales.append(rationale)
                    question = dataset["question"][i]
                    if question[0] == '"':
                        question = question[1:]
                    if question[-1] == '"':
                        question = question[:-1]
                    inputs.append(question)
                except ValueError:
                    print(
                        f"[WARNING] Ignoring sample. Could not convert target to float: {parts[1]}"
                    )

        inputs = np.array(inputs)
        targets = np.array(targets)
        rationales = np.array(rationales)

        # Write the dataset to a csv file
        with open(path, "w", newline="", encoding="utf-8") as file:
            data = np.column_stack((inputs, targets, rationales))
            write_csv(file, ["SAMPLE", "TARGET", "EXAMPLE_RATIONALE"], data)
