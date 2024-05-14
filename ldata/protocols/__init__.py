from __future__ import annotations

from typing import Protocol

from .logger import Logger

__all__ = ["Addable", "Logger"]


class Addable(Protocol):
    """Protocol for objects that can be added together."""

    def __add__(self, other: Addable | None) -> Addable: ...

    def __radd__(self, other: Addable | None) -> Addable: ...

    def __iadd__(self, other: Addable | None) -> Addable: ...
