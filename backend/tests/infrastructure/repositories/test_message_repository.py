import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.domain.entities.message_role import MessageRole
from app.infrastructure.database.models import Conversation, Message, SceneModel, User
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.message_repository import SqlAlchemyMessageRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


def _make_conversation(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> Conversation:
    scene = SqlAlchemySceneRepository(db_session).add(scene_factory())
    return SqlAlchemyConversationRepository(db_session).add(
        Conversation(user_id=user.id, scene_id=scene.id)
    )


def test_add_persists_message_linked_to_conversation(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    conversation = _make_conversation(db_session, user, scene_factory)
    repository = SqlAlchemyMessageRepository(db_session)

    message = repository.add(
        Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="O que tem na minha frente?",
        )
    )

    assert isinstance(message.id, uuid.UUID)
    assert message.role == MessageRole.USER


def test_list_by_conversation_id_returns_messages_in_chronological_order(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    conversation = _make_conversation(db_session, user, scene_factory)
    repository = SqlAlchemyMessageRepository(db_session)

    repository.add(
        Message(conversation_id=conversation.id, role=MessageRole.USER, content="Pergunta")
    )
    repository.add(
        Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content="Resposta",
            model_name="qwen3.5:4b",
            latency_ms=850,
        )
    )

    messages = repository.list_by_conversation_id(conversation.id)

    assert [message.role for message in messages] == [MessageRole.USER, MessageRole.ASSISTANT]


def test_list_by_conversation_id_does_not_leak_messages_from_other_conversations(
    db_session: Session, user: User, scene_factory: Callable[..., SceneModel]
) -> None:
    conversation_one = _make_conversation(db_session, user, scene_factory)
    conversation_two = _make_conversation(db_session, user, scene_factory)
    repository = SqlAlchemyMessageRepository(db_session)

    repository.add(
        Message(conversation_id=conversation_one.id, role=MessageRole.USER, content="Cena 1")
    )
    repository.add(
        Message(conversation_id=conversation_two.id, role=MessageRole.USER, content="Cena 2")
    )

    messages = repository.list_by_conversation_id(conversation_one.id)

    assert [message.content for message in messages] == ["Cena 1"]
