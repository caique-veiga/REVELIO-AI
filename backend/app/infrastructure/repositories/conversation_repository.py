import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.database.models import Conversation


class SqlAlchemyConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, conversation: Conversation) -> Conversation:
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        return self._session.get(Conversation, conversation_id)

    def get_by_scene_id(self, scene_id: uuid.UUID) -> Conversation | None:
        stmt = select(Conversation).where(Conversation.scene_id == scene_id)
        return self._session.scalars(stmt).one_or_none()
