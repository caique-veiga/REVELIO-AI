import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationAnswer:
    answer: str
    scene_id: uuid.UUID
