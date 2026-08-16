import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.database.models import DetectedObject


class SqlAlchemyObjectRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_many(self, detected_objects: list[DetectedObject]) -> list[DetectedObject]:
        self._session.add_all(detected_objects)
        self._session.flush()
        return detected_objects

    def list_by_scene_id(self, scene_id: uuid.UUID) -> list[DetectedObject]:
        stmt = select(DetectedObject).where(DetectedObject.scene_id == scene_id)
        return list(self._session.scalars(stmt))
