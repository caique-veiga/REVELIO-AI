import uuid
from datetime import datetime

from pydantic import BaseModel

from app.domain.entities.conversation_answer import ConversationAnswer
from app.domain.entities.message_role import MessageRole
from app.infrastructure.database.models import Conversation, Message


class AskQuestionRequest(BaseModel):
    content: str


class ReferencedObjectSchema(BaseModel):
    object_id: uuid.UUID
    class_name: str


class AskQuestionResponse(BaseModel):
    answer: str
    scene_id: uuid.UUID
    referenced_objects: list[ReferencedObjectSchema]

    @classmethod
    def from_domain(cls, conversation_answer: ConversationAnswer) -> "AskQuestionResponse":
        return cls(
            answer=conversation_answer.answer,
            scene_id=conversation_answer.scene_id,
            referenced_objects=[
                ReferencedObjectSchema(object_id=ref.object_id, class_name=ref.class_name)
                for ref in conversation_answer.referenced_objects
            ],
        )


class MessageSchema(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime
    model_name: str | None = None
    latency_ms: int | None = None

    @classmethod
    def from_domain(cls, message: Message) -> "MessageSchema":
        return cls(
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            model_name=message.model_name,
            latency_ms=message.latency_ms,
        )


class ConversationSchema(BaseModel):
    conversation_id: uuid.UUID
    scene_id: uuid.UUID
    messages: list[MessageSchema]

    @classmethod
    def from_rows(cls, conversation: Conversation, messages: list[Message]) -> "ConversationSchema":
        return cls(
            conversation_id=conversation.id,
            scene_id=conversation.scene_id,
            messages=[MessageSchema.from_domain(message) for message in messages],
        )
