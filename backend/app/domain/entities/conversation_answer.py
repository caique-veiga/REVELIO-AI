import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencedObject:
    object_id: uuid.UUID
    class_name: str


@dataclass(frozen=True)
class ConversationAnswer:
    answer: str
    scene_id: uuid.UUID
    referenced_objects: list[ReferencedObject]
