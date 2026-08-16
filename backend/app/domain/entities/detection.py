import uuid
from dataclasses import dataclass, field

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.position import Position


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    object_id: uuid.UUID = field(default_factory=uuid.uuid4)
    position: Position | None = None
