from dataclasses import dataclass
from enum import StrEnum


class HorizontalPosition(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VerticalPosition(StrEnum):
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class Region(StrEnum):
    FRONT_LEFT = "front-left"
    FRONT_CENTER = "front-center"
    FRONT_RIGHT = "front-right"
    UPPER_LEFT = "upper-left"
    UPPER_CENTER = "upper-center"
    UPPER_RIGHT = "upper-right"
    LOWER_LEFT = "lower-left"
    LOWER_CENTER = "lower-center"
    LOWER_RIGHT = "lower-right"


@dataclass(frozen=True)
class Position:
    horizontal: HorizontalPosition
    vertical: VerticalPosition
    region: Region
