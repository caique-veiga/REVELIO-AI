from typing import Protocol

from app.domain.entities.detection import Detection


class ObjectDetector(Protocol):
    def detect(self, image: bytes) -> list[Detection]: ...
