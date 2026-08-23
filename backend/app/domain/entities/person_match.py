import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class PersonCandidate:
    person_id: uuid.UUID
    name: str
    relationship: str


@dataclass(frozen=True)
class PersonMatch:
    person_id: uuid.UUID
    name: str
    relationship: str
    similarity: float
