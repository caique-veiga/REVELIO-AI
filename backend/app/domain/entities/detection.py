import uuid
from dataclasses import dataclass, field

from app.domain.entities.bounding_box import BoundingBox


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    object_id: uuid.UUID = field(default_factory=uuid.uuid4)
