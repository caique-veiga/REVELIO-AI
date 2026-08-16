import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.database.models import Message


class SqlAlchemyMessageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, message: Message) -> Message:
        self._session.add(message)
        self._session.flush()
        return message

    def list_by_conversation_id(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return list(self._session.scalars(stmt))
