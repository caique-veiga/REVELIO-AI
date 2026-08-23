import uuid
from dataclasses import dataclass, field

from app.domain.entities.stored_image import StoredImage


@dataclass(frozen=True)
class Scene:
    image: StoredImage
    scene_id: uuid.UUID = field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID | None = None
