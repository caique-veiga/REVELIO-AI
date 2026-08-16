import uuid

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Scene


class SqlAlchemySceneRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, scene: Scene) -> Scene:
        self._session.add(scene)
        self._session.flush()
        return scene

    def get_by_id(self, scene_id: uuid.UUID) -> Scene | None:
        return self._session.get(Scene, scene_id)
