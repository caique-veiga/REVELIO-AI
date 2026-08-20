from dataclasses import dataclass

from app.domain.entities.message_role import MessageRole


@dataclass(frozen=True)
class ConversationMessage:
    role: MessageRole
    content: str
