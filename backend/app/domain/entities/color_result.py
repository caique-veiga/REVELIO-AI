from dataclasses import dataclass
from enum import StrEnum


class ColorName(StrEnum):
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"


@dataclass(frozen=True)
class ColorResult:
    name: ColorName
    rgb: tuple[int, int, int]
    confidence: float
