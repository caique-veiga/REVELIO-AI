import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.infrastructure.database.models import SceneModel


class SceneRepository(Protocol):
    def add(self, scene: "SceneModel") -> "SceneModel": ...

    def get_by_id(self, scene_id: uuid.UUID) -> "SceneModel | None": ...
