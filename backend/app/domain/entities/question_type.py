from enum import StrEnum


class QuestionType(StrEnum):
    GENERAL = "general"
    SPATIAL = "spatial"
    COLOR = "color"
    OBJECT = "object"
    UNCERTAINTY = "uncertainty"
