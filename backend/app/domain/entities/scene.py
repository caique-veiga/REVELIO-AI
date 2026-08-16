import uuid
from dataclasses import dataclass, field

from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.stored_image import StoredImage


@dataclass(frozen=True)
class Scene:
    image: StoredImage
    model: ModelMetadata
    objects: list[Detection]
    scene_id: uuid.UUID = field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID | None = None
