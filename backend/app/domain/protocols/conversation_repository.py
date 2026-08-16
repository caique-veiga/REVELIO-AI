import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.infrastructure.database.models import Conversation


class ConversationRepository(Protocol):
    def add(self, conversation: "Conversation") -> "Conversation": ...

    def get_by_id(self, conversation_id: uuid.UUID) -> "Conversation | None": ...

    def get_by_scene_id(self, scene_id: uuid.UUID) -> "Conversation | None": ...
