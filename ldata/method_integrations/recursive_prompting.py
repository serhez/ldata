from typing import Protocol

_SUBPROBLEM_PREFIXES = (
    [
        "sub-problem:",
        "subproblem:",
        "sub problem:",
        "problem:",
    ]
    + ["-", "*", "•", "+"]
    + [str(i) for i in range(1, 50)]
)


class RecursivePromptingCompatible(Protocol):
    """
    A benchmark compatible with the evaluation of the Recursive Prompting method.
    Implementing this protocol allows for the isolated evaluation of decomposing and merging problems using Recursive Prompting.
    """

    def get_subproblems(
        self,
        sample: str,
        n_subproblems: int,
    ) -> list[tuple[str, str]]:
        """
        Get a correct decomposition of an uninstructed sample: `n_subproblems` plausible sub-problem and sub-solution pairs.
        - Note that the returned decomposition may not likely be the only possible correct one.

        ### Parameters
        ----------
        `sample`: the uninstructed sample.
        `n_subproblems`: the number of sub-problems the input should be split into.

        ### Returns
        -------
        A list of tuples containing the pairs of instructed sub-problems and sub-solutions.
        """

        ...

    def evaluate_split(
        self,
        input: str,
        split: list[str],
        n_subproblems: int | None = None,
    ) -> float:
        """
        Evaluate a single (input, split) pair.
        It returns 1.0 if the output is a valid split of the input, and 0.0 otherwise.
        The input must be uninstructed.

        ### Parameters
        ----------
        `input`: the uninstructed input.
        `split`: the splitting of the input.
        `n_subproblems`: the number of sub-problems the input should be split into.
        - If `None`, then the score will not be affected by the num. of sub-problems in the split.

        ### Returns
        -------
        The score of the output.
        """

        ...

    def evaluate_merge(
        self,
        split: list[str],
        merged: str,
        **kwargs,
    ) -> tuple[float, str]:
        """
        Evaluate a single (split, merged) pair.

        ### Parameters
        ----------
        `split`: the splitting of the problem.
        `merged`: the merged solution.
        Other keyword arguments may be used by each specific benchmark.

        ### Returns
        -------
        A tuple containing the score of the merged answer and the found solution within the answer.
        - 1.0 if the output is the correct merged solution.
        - 0.0 if the split is valid and the merged solution is incorrect.
        - -1.0 if the split is invalid, in which case the found solution is an empty string.
        """

        ...
