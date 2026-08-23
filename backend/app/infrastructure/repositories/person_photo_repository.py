import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.person_match import PersonCandidate
from app.infrastructure.database.models import Person, PersonPhoto


class SqlAlchemyPersonPhotoRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, person_photo: PersonPhoto) -> PersonPhoto:
        self._session.add(person_photo)
        self._session.flush()
        return person_photo

    def list_embeddings_by_user(
        self, user_id: uuid.UUID
    ) -> list[tuple[PersonCandidate, np.ndarray]]:
        stmt = (
            select(Person.id, Person.name, Person.relationship_label, PersonPhoto.photo_embedding)
            .join(PersonPhoto, PersonPhoto.person_id == Person.id)
            .where(Person.user_id == user_id)
        )
        rows = self._session.execute(stmt).all()
        return [
            (
                PersonCandidate(person_id=person_id, name=name, relationship=relationship_label),
                np.frombuffer(embedding, dtype=np.float64),
            )
            for person_id, name, relationship_label, embedding in rows
        ]
