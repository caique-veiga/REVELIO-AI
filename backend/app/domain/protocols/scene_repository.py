import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.infrastructure.database.models import Scene


class SceneRepository(Protocol):
    def add(self, scene: "Scene") -> "Scene": ...

    def get_by_id(self, scene_id: uuid.UUID) -> "Scene | None": ...
