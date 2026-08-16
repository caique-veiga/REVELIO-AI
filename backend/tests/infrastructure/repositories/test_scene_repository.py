import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Scene
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


def test_add_persists_scene_and_assigns_id(
    db_session: Session, scene_factory: Callable[..., Scene]
) -> None:
    repository = SqlAlchemySceneRepository(db_session)

    scene = repository.add(scene_factory())

    assert isinstance(scene.id, uuid.UUID)
    assert scene.image_filename == "photo.jpg"


def test_get_by_id_returns_persisted_scene(
    db_session: Session, scene_factory: Callable[..., Scene]
) -> None:
    repository = SqlAlchemySceneRepository(db_session)
    scene = repository.add(scene_factory())

    found = repository.get_by_id(scene.id)

    assert found is not None
    assert found.id == scene.id


def test_get_by_id_returns_none_when_not_found(db_session: Session) -> None:
    repository = SqlAlchemySceneRepository(db_session)

    assert repository.get_by_id(uuid.uuid4()) is None
