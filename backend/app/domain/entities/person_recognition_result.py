import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterPersonResult:
    success: bool
    message: str
    person_id: uuid.UUID | None = None


@dataclass(frozen=True)
class IdentifiedPerson:
    name: str | None
    relationship: str | None
    similarity: float
    horizontal_position: str


@dataclass(frozen=True)
class IdentifyPersonsResult:
    success: bool
    message: str
    identified_count: int
    total_faces: int
