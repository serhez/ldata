from enum import Enum


class EvaluationMetric(Enum):
    """The metric measured by the evaluation."""

    EXACT = "exact"
    WORD = "word"
    CHARACTER = "character"
