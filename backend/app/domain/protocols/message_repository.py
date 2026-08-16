import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.infrastructure.database.models import Message


class MessageRepository(Protocol):
    def add(self, message: "Message") -> "Message": ...

    def list_by_conversation_id(self, conversation_id: uuid.UUID) -> list["Message"]: ...
