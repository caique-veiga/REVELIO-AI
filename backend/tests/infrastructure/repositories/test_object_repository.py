from collections.abc import Callable

from sqlalchemy.orm import Session

from app.infrastructure.database.models import DetectedObject, SceneModel
from app.infrastructure.repositories.object_repository import SqlAlchemyObjectRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


def test_add_many_persists_all_detected_objects(
    db_session: Session,
    scene_factory: Callable[..., SceneModel],
    detected_object_factory: Callable[..., DetectedObject],
) -> None:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyObjectRepository(db_session)

    objects = repository.add_many(
        [
            detected_object_factory(scene.id, class_name="person"),
            detected_object_factory(scene.id, class_name="chair", class_id=56),
        ]
    )

    assert all(obj.id is not None for obj in objects)


def test_list_by_scene_id_returns_only_objects_from_that_scene(
    db_session: Session,
    scene_factory: Callable[..., SceneModel],
    detected_object_factory: Callable[..., DetectedObject],
) -> None:
    scene_one = SqlAlchemySceneRepository(db_session).add(scene_factory())
    scene_two = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyObjectRepository(db_session)

    repository.add_many([detected_object_factory(scene_one.id, class_name="person")])
    repository.add_many([detected_object_factory(scene_two.id, class_name="chair")])

    objects = repository.list_by_scene_id(scene_one.id)

    assert [obj.class_name for obj in objects] == ["person"]
