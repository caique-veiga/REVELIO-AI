import uuid

from pydantic import BaseModel


class SceneCreationResponse(BaseModel):
    scene_id: uuid.UUID
    conversation_id: uuid.UUID
    status: str
