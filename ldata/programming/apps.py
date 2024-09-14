import faulthandler
import json
import os
import signal
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Any
from unittest.mock import mock_open, patch

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset

from ldata.benchmark import Benchmark
from ldata.dataset import BuildableDataset, Dataset
from ldata.protocols import Logger
from ldata.utils import (
    Capturing,
    reliability_guard,
    stripped_string_compare,
    write_csv,
)


class APPS(BuildableDataset, Benchmark):
    """
    Wrapper for the APPS benchmark (D. Hendrycks et al., 2021). All credits go to the authors and contributors of the original implementation and publication.
    The evaluation metric is provided by [the original APPS implementation](https://github.com/hendrycks/apps).
    The dataset files contain the regular "SAMPLE" and "TARGET" columns, and additionally an "EXAMPLE_SOLUTION" column that contains an example code solution for each sample.
    The range of score values is [0.0, 1.0].

    ### Notes
    ----------
    - You probably want to use the `APPS.build` method to cache the dataset to a file in your local filesystem and then provide the file path via `Config.data_path`. Otherwise, the dataset will be downloaded every time you instantiate this class.
    """

    _TIMEOUT = 4  # seconds
    _INSTRUCTIONS_TEMPLATE = "A programming task is given below. To solve it, you must only output the code and no other text.\n{}"

    class Difficulty(Enum):
        """The difficulty levels of the APPS benchmark."""

        INTRODUCTORY = "introductory"
        INTERVIEW = "interview"
        COMPETITION = "competition"

    class _CodeType(Enum):
        """The type of code to evaluate."""

        CALL_BASED = "call_based"
        STANDARD_INPUT = "standard_input"

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the APPS benchmark."""

        name: str = "APPS"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return APPS.Config

    def __init__(self, config: Config):
        """
        Initialize the APPS benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    def get_instructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            return self._INSTRUCTIONS_TEMPLATE.format(sample)

        inputs = np.array(
            [self._INSTRUCTIONS_TEMPLATE.format(s) for s in sample.inputs]
        )
        return Dataset.Split(inputs, sample.targets)

    def get_uninstructed(
        self, sample: str | Dataset.Split | None = None
    ) -> str | Dataset.Split:
        if sample is None:
            sample = self.test_set

        if isinstance(sample, str):
            return sample[sample.index("\n") + 1 :].strip()

        inputs = np.empty(len(sample.inputs), dtype=np.str_)
        for i, s in enumerate(sample.inputs):
            inputs[i] = s[s.index("\n") + 1 :].strip()

        return Dataset.Split(inputs, sample.targets)

    @classmethod
    def _custom_compare(cls, output: Any, ground_truth: Any) -> bool:
        if isinstance(output, list):
            output_1 = "\n".join(output)
            if stripped_string_compare(output_1, ground_truth):
                return True

        if isinstance(output, list):
            output_2 = [o.lstrip().rstrip() for o in output]
            output_2 = "\n".join(output_2)
            if stripped_string_compare(output_2, ground_truth):
                return True

        return False

    @classmethod
    def _call_method(cls, method, inputs):
        if isinstance(inputs, list):
            inputs = "\n".join(inputs)

        inputs_line_iterator = iter(inputs.split("\n"))

        # sys.setrecursionlimit(10000)

        # @patch('builtins.input', side_effect=inputs.split("\n"))
        @patch("builtins.open", mock_open(read_data=inputs))
        @patch("sys.stdin", StringIO(inputs))
        @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
        @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
        @patch("sys.stdin.read", lambda *args: inputs)
        # @patch('sys.stdout.write', print)
        def _inner_call_method(_method):
            try:
                return _method()
            except SystemExit:
                pass
            finally:
                pass

        return _inner_call_method(method)

    def _evaluate_output_impl(
        self,
        output: str,
        target: str,
        _,
        logger: Logger | None = None,
    ) -> float:
        """
        Evaluate the output of the APPS benchmark.
        This method executes the output on the tests provided in the target.
        This code is an adaptation from the original APPS implementation, hence all credits go to the original authors.

        ### Parameters
        ----------
        `output`: the output to evaluate.
        `target`: the target output.
        `metric`: the evaluation method to use.
        - Not used, but required by the signature of the method.
        [optional] `logger`: the logger to use for logging.

        ### Returns
        ----------
        The evaluation score.
        """

        reliability_guard()

        results = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"

        in_outs = json.loads(repr(target)[2:-2])
        if in_outs.get("fn_name") is None:
            which_type = APPS._CodeType.STANDARD_INPUT
            method_name = "code"
        else:
            which_type = APPS._CodeType.CALL_BASED
            method_name = in_outs["fn_name"]

        if which_type == APPS._CodeType.CALL_BASED:
            sol += output
            signal.alarm(self._TIMEOUT)
            try:
                namespace = {}
                exec(sol, namespace)

                if "class Solution" in output:
                    Solution = namespace["Solution"]
                    tmp = Solution()
                    method = getattr(tmp, method_name)
                else:
                    method = namespace[method_name]

                signal.alarm(0)

            except Exception as e:
                signal.alarm(0)
                if logger:
                    logger.error(f"[APPS.evaluate_output] Compilation error = {e}")
                return 0.0
            signal.alarm(0)

        elif which_type == APPS._CodeType.STANDARD_INPUT:
            # sol
            tmp_test = output.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            signal.alarm(self._TIMEOUT)
            try:
                namespace = {}
                exec(sol, namespace)
                method = namespace[method_name]
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if logger:
                    logger.error(
                        f"[APPS.evaluate_output] Type 1 compilation error = {e}"
                    )
                return 0.0
            signal.alarm(0)

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index].items()}
                    ]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index][0].items()}
                    ]
            except Exception:
                pass

            if which_type == APPS._CodeType.CALL_BASED:
                signal.alarm(self._TIMEOUT)
                faulthandler.enable()
                try:
                    result = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(result, tuple):
                        result = list(result)

                    tmp_result = result == in_outs["outputs"][index]
                    if (
                        isinstance(in_outs["outputs"][index], list)
                        and in_outs["outputs"][index]
                    ):
                        tmp_result = tmp_result or (
                            result == in_outs["outputs"][index][0]
                        )

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(result[0], tuple):
                            tmp_result = tmp_result or (
                                [list(x) for x in result]
                                == in_outs["outputs"][index][0]
                            )
                    except Exception:
                        pass
                    results.append(float(tmp_result))

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Standard input runtime error or time limit exceeded error = {e}"
                        )
                    results.append(0.0)
                    continue
                faulthandler.disable()
                signal.alarm(0)

            elif which_type == APPS._CodeType.STANDARD_INPUT:
                faulthandler.enable()
                signal.alarm(self._TIMEOUT)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)  # type: ignore[reportCallIssue]
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                with Capturing() as _:
                    try:
                        self._call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        if logger:
                            logger.error(
                                f"[APPS.evaluate_output] Call-based runtime error or time limit exceeded error = {repr(e)}{e}"
                            )
                        results.append(0.0)
                    signal.alarm(0)

                if not passed:
                    continue

                if self._custom_compare(result, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(float(tmp_result))
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(result, tuple):
                    result = list(result)

                tmp_result = False
                try:
                    tmp_result = result == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (result == in_outs["outputs"][index])
                        if isinstance(result[0], str):  # type: ignore[reportIndexIssue]
                            tmp_result = tmp_result or (
                                [e.strip() for e in result] == in_outs["outputs"][index]  # type: ignore[reportAttributeAccessIssue]
                            )
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check1 exception = {e}"
                        )
                    pass

                if tmp_result is True:
                    results.append(float(tmp_result))
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(
                        filter(len, in_outs["outputs"][index])
                    )
                    in_outs["outputs"][index] = list(
                        map(lambda x: x.strip(), in_outs["outputs"][index])
                    )

                try:
                    tmp_result = result == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (result == in_outs["outputs"][index])
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check2 exception = {e}"
                        )
                    pass

                if tmp_result is True:
                    results.append(float(tmp_result))
                    continue

                # try by converting the result into a split up list too
                if isinstance(result, list):
                    result = list(filter(len, result))

                if tmp_result is True:
                    results.append(float(tmp_result))
                    continue

                try:
                    tmp_result = result == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (result == in_outs["outputs"][index])
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check3 exception = {e}"
                        )
                    pass

                try:
                    result_float = [float(e) for e in result]  # type: ignore[reportArgumentType]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or (
                        (len(result_float) == len(gt_float))
                        and np.allclose(result_float, gt_float)
                    )
                except Exception:
                    pass
                try:
                    if isinstance(result[0], list):  # type: ignore[reportIndexIssue]
                        result_float = [float(e) for e in result[0]]  # type: ignore[reportIndexIssue]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or (
                            (len(result_float) == len(gt_float))
                            and np.allclose(result_float, gt_float)
                        )
                except Exception:
                    pass

                if tmp_result is True:
                    results.append(float(tmp_result))
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = result == in_outs["outputs"][index]
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check4 exception = {e}"
                        )
                    continue

                if tmp_result is True:
                    results.append(float(tmp_result))
                    continue

                # try by converting the result into a split up list too
                if isinstance(result, list):
                    for tmp_index, i in enumerate(result):
                        result[tmp_index] = i.split()
                    result = list(filter(len, result))
                    for tmp_index, i in enumerate(result):
                        result[tmp_index] = set(i)
                else:
                    result = result.split()  # type: ignore[reportAttributeAccessIssue]
                    result = list(filter(len, result))
                    result = set(result)

                try:
                    tmp_result = set(frozenset(s) for s in result) == set(  # type: ignore[reportArgumentType]
                        frozenset(s) for s in in_outs["outputs"][index]
                    )
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check5 exception = {e}"
                        )

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (
                        set(frozenset(round(float(t), 3) for t in s) for s in result)  # type: ignore[reportGeneralTypeIssues]
                        == set(
                            frozenset(round(float(t), 3) for t in s)
                            for s in in_outs["outputs"][index]
                        )
                    )
                except Exception as e:
                    if logger:
                        logger.error(
                            f"[APPS.evaluate_output] Failed check6 exception = {e}"
                        )

                results.append(float(tmp_result))

        return float(np.mean(results))

    def _extract_solution_impl(self, output: str, _) -> str:
        solution = ""
        inside_code_block = False
        detected_code_block = False

        for line in output.split("\n"):
            if line.startswith("```"):
                inside_code_block = not inside_code_block
                detected_code_block = True
            elif inside_code_block:
                solution += line + "\n"

        if not detected_code_block:
            return output

        return solution

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        difficulties: list[Difficulty] = [
            Difficulty.INTRODUCTORY,
            Difficulty.INTERVIEW,
            Difficulty.COMPETITION,
        ],
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Build the APPS dataset.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `difficulties`: the difficulties of the samples to generate.
        `shuffle`: whether to shuffle the dataset before sampling `n_samples`.
        `seed`: the random seed to use for shuffling the dataset.

        ### Raises
        ----------
        `AssertionError`: if the `HF_API_TOKEN` environment variable is not set.
        """

        # Check if the user's API token is set
        assert (
            "HF_API_TOKEN" in os.environ
        ), "You must set the `HF_API_TOKEN` environment variable to build the APPS benchmark. "

        # Load the dataset
        splits: DatasetDict = load_dataset("codeparrot/apps", trust_remote_code=True)  # type: ignore[reportAssignmentType]
        dataset = concatenate_datasets([split for split in splits.values()])

        # Shuffle the dataset
        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        # Filter the dataset by difficulty
        dataset = dataset.filter(
            lambda x, *_: x["difficulty"] in [d.value for d in difficulties],
            with_indices=True,
        )

        # Select the first `n_samples` samples
        dataset = dataset.select(range(n_samples))

        # Get dataset columns
        inputs = np.array([e for e in dataset["question"]])
        targets = np.array([e for e in dataset["input_output"]])
        solutions = np.array([e for e in dataset["solutions"]])

        # Add the starting code to the inputs
        starter_code = np.array([e for e in dataset["starter_code"]])
        inputs = np.array(
            [
                inp if sc == "" else f"{inp}\n{sc}"
                for inp, sc in zip(inputs, starter_code)
            ]
        )

        # Write the dataset to a csv file
        with open(path, "w", newline="", encoding="utf-8") as file:
            data = np.column_stack((inputs, targets, solutions))
            write_csv(file, ["SAMPLE", "TARGET", "EXAMPLE_SOLUTION"], data)
