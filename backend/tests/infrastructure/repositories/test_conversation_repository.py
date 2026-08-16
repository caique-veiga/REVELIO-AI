import uuid
from collections.abc import Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.infrastructure.database.models import Conversation, SceneModel, User
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


def test_add_persists_conversation_linked_to_scene(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyConversationRepository(db_session)

    conversation = repository.add(Conversation(user_id=user.id, scene_id=scene.id))

    assert isinstance(conversation.id, uuid.UUID)
    assert conversation.scene_id == scene.id


def test_get_by_scene_id_returns_the_associated_conversation(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyConversationRepository(db_session)
    conversation = repository.add(Conversation(user_id=user.id, scene_id=scene.id))

    found = repository.get_by_scene_id(scene.id)

    assert found is not None
    assert found.id == conversation.id


def test_get_by_scene_id_returns_none_when_scene_has_no_conversation(
    db_session: Session, scene_factory: Callable[..., SceneModel]
) -> None:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyConversationRepository(db_session)

    assert repository.get_by_scene_id(scene.id) is None


def test_a_scene_cannot_have_more_than_one_conversation(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    repository = SqlAlchemyConversationRepository(db_session)
    repository.add(Conversation(user_id=user.id, scene_id=scene.id))

    with pytest.raises(IntegrityError):
        repository.add(Conversation(user_id=user.id, scene_id=scene.id))
