import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.infrastructure.database.models import DetectedObject


class ObjectRepository(Protocol):
    def add_many(self, detected_objects: list["DetectedObject"]) -> list["DetectedObject"]: ...

    def list_by_scene_id(self, scene_id: uuid.UUID) -> list["DetectedObject"]: ...
