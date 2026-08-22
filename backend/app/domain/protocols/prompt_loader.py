from typing import Protocol


class PromptNotFoundError(Exception):
    """Levantado quando um arquivo de prompt não existe."""


class PromptLoader(Protocol):
    def load(self, category: str, name: str) -> str: ...
