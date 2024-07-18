"""A collection of interfaces that `Dataset` and `Benchmark` classes can implement to better integrate with specific LLM-based methods."""

from .recursive_prompting import RecursivePromptingCompatible

__all__ = ["RecursivePromptingCompatible"]
