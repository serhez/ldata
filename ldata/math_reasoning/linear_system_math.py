import random
import re
from dataclasses import dataclass

import numpy as np

from ldata.benchmark import Benchmark
from ldata.dataset import Dataset


class LinearSystemMath(Benchmark):
    """
    Benchmark consisting on solving a single variable given a linear system of equations, expressed in mathematical notation.
    The evaluation metric is the correctness of the numerical solution, up to two decimal places.
    The score values can be either 0.0 or 1.0, regardless of the `EvaluationMethod`.
    All samples follow the format: "Solve for {variable}, given that {system}.".
    - The variable is a single letter.
    - The system is a list of equations, separated by semicolons. Each equation is in the form $\\sum_{i=1}^{n} a_i x_i = b$, where $a_i$ and $b$ are integers and $n$ is the number of variables. For example: "2x + 3y = 5; 4x = 6".
    The target is the numerical solution of the variable, up to two decimal places.
    All systems have the same number of equations as the number of variables and are guaranteed to have a unique solution for the variable to solve.
    All coefficients and constants are integers in the range [-10, 10].
    """

    _INSTRUCTIONS_TEMPLATE = "Solve for {}, given the linear system of equations [{}]."
    _LETTERS = "abcdefghijklmnopqrstuvwxyz"
    _NUMBERS = "0123456789"
    _N_RETRIES = int(10e2)

    @dataclass(kw_only=True)
    class Config(Benchmark.Config):
        """The configuration of the linear system of equations benchmark."""

        name: str = "LinearSystemMath"
        """The name of the benchmark."""

    @property
    def config_cls(self) -> type[Config]:
        return LinearSystemMath.Config

    def __init__(self, config: Config):
        """
        Initialize the linear system of equations benchmark.

        ### Parameters
        ----------
        `config`: the configuration of the benchmark.
        """

        super().__init__(config)

    @classmethod
    def compute_target(cls, input: str) -> str:
        equations = input.split(";")
        var_to_solve = equations[0]
        equations = equations[1:]

        coefficients = np.zeros((len(equations), len(equations)))
        constants = np.zeros(len(equations))

        # Create an order of the variables to solve the system
        vars = list(
            set([c for equation in equations for c in equation if c in cls._LETTERS])
        )
        var_to_solve_idx = vars.index(var_to_solve)

        for e, equation in enumerate(equations):
            equation = equation.strip()
            if equation == "":
                continue

            terms, constant = equation.split("=")
            constants[e] = int(constant.strip())

            terms = terms.replace(" ", "")
            terms = terms.replace("+", " +")
            terms = terms.replace("-", " -")
            terms = terms.split(" ")
            terms = [t for t in terms if t != ""]

            for term in terms:
                sign = 1
                if term[0] == "-":
                    sign = -1
                    term = term[1:]
                elif term[0] == "+":
                    term = term[1:]

                unsigned_value = ""
                for i in range(len(term)):
                    if term[i] in cls._NUMBERS or term[i] == ".":
                        unsigned_value = term[: i + 1]
                    else:
                        break

                var = term[len(str(unsigned_value)) :]
                unsigned_value = 1 if unsigned_value == "" else float(unsigned_value)

                idx = vars.index(var)
                coefficients[e][idx] = sign * unsigned_value

        # Solve the system
        solution = np.linalg.solve(coefficients, constants)

        return f"{solution[var_to_solve_idx]:.2f}"

    @classmethod
    def build(
        cls,
        path: str,
        n_samples: int,
        n_vars: int,
    ):
        """
        Build the linear system of equations dataset.
        The dataset is saved in a csv file with the following format:
        ```
        SAMPLE,TARGET
        {variable_to_solve};{system},{solution}
        ...
        ```, where:
        - `variable_to_solve` is a single letter.
        - `system` is a list of equations, separated by semicolons. Each equation is in the form $\\sum_{i=1}^{n} a_i x_i = b$, where $a_i$ and $b$ are integers and $n$ is the number of variables.
        - `solution` is the numerical solution of the variable to solve, up to two decimal places.

        ### Parameters
        ----------
        `path`: the path to save the dataset.
        `n_samples`: the number of samples to generate.
        `n_vars`: the number of variables in the system.

        ### Notes
        ----------
        - All systems are guaranteed to have a unique solution for the variable to solve.
        - All coefficients and constants are integers in the range [-10, 10].
        - The progress can be quite slow: generating each target requires solving a system of equations.
        """

        samples = []
        targets = []

        for _ in range(n_samples):
            vars = random.sample(cls._LETTERS, n_vars)
            var = random.choice(vars)

            for _ in range(cls._N_RETRIES):
                equations = []
                coefficients = np.random.randint(-10, 10, (n_vars, n_vars))

                # Ensure that the system has a unique solution by checking if the determinant of the coefficient matrix is non-zero
                if np.isclose(np.linalg.det(coefficients), 0.0):
                    continue

                constants = np.random.randint(-10, 10, n_vars)

                # Generate each equation
                for e in range(n_vars):
                    idxs = list(range(n_vars))
                    random.shuffle(idxs)
                    equation = ""

                    # Add the coefficients and variables
                    first = True
                    for i in range(n_vars):
                        c = coefficients[e][idxs[i]]
                        v = vars[idxs[i]]

                        if c == 0:
                            continue
                        if c < 0:
                            if first:
                                equation += f"-{-c if c != -1 else ''}{v}"
                                first = False
                            else:
                                equation += f" - {-c if c != -1 else ''}{v}"
                        else:
                            if first:
                                equation += f"{c if c != 1 else ''}{v}"
                                first = False
                            else:
                                equation += f" + {c if c != 1 else ''}{v}"

                    # Add the constant
                    equation += f" = {constants[e]}"

                    equations.append(equation)

                break

            if len(equations) < n_vars:
                raise ValueError(
                    f"Try again: could not generate a random system with a unique solution after {cls._N_RETRIES} re-tries."
                )

            samples.append(f"{var};{';'.join(equations)}")
            targets.append(cls.compute_target(samples[-1]))

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
            split_sample = sample.split(";")
            return self._INSTRUCTIONS_TEMPLATE.format(
                split_sample[0], ", ".join(split_sample[1:])
            )

        split_inputs = [s.split(";") for s in sample.inputs]
        inputs = np.array(
            [
                self._INSTRUCTIONS_TEMPLATE.format(s[0], ", ".join(s[1:]))
                for s in split_inputs
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
        *_,
    ) -> float:
        """
        Evaluate the output of the model given the target.
        All samples are evaluated based on the correctness of the numerical solution, up to two decimal places.
        The score values can be either 0.0 or 1.0, regardless of the `EvaluationMethod`.

        ### Parameters
        ----------
        `output`: the output of the model.
        `target`: the target of the sample.

        ### Returns
        ----------
        The score of the model output.

        ### Raises
        ----------
        `ValueError`: if the target is not a valid floating point number.
        """

        float_target = float(target.replace(" ", ""))

        try:
            float_output = float(output)
        except ValueError:
            return 0.0

        return round(float_output, 2) == float_target

    @classmethod
    def _extract_solution_impl(cls, output: str, target: str) -> str:
        # Search for all floating point numbers in the output
        search = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if len(search) == 0:
            return ""

        # Evaluate all found matches
        for match in search:
            if cls._evaluate_output_impl(match, target) == 1.0:
                return str(match)

        # Return the first match if no exact match was found
        return search[0]
