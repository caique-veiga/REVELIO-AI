from typing import Protocol

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorResult


class ColorAnalyzer(Protocol):
    def analyze(self, image: bytes, bbox: BoundingBox) -> ColorResult: ...
