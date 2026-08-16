import uuid

from sqlalchemy.orm import Session

from app.infrastructure.database.models import SceneModel


class SqlAlchemySceneRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, scene: SceneModel) -> SceneModel:
        self._session.add(scene)
        self._session.flush()
        return scene

    def get_by_id(self, scene_id: uuid.UUID) -> SceneModel | None:
        return self._session.get(SceneModel, scene_id)
