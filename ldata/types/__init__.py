from enum import Enum

from .evaluation import EvaluationMetric

__all__ = ["EvaluationMetric", "Separator"]


class Separator(Enum):
    SPACE = " "
    COMMA = ","
    SEMICOLON = ";"
    COLON = ":"
    DASH = "-"
    UNDERSCORE = "_"
    PIPE = "|"

    def __str__(self) -> str:
        return self.value

    @property
    def descriptor(self) -> str:
        """The natural language descriptor of the separator."""

        return {
            Separator.SPACE: "space",
            Separator.COMMA: "comma",
            Separator.SEMICOLON: "semicolon",
            Separator.COLON: "colon",
            Separator.DASH: "dash",
            Separator.UNDERSCORE: "underscore",
            Separator.PIPE: "pipe",
        }[self]
