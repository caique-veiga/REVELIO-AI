import uuid

from sqlalchemy.orm import Session

from app.infrastructure.database.models import Person


class SqlAlchemyPersonRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, person: Person) -> Person:
        self._session.add(person)
        self._session.flush()
        return person

    def get_by_id(self, person_id: uuid.UUID) -> Person | None:
        return self._session.get(Person, person_id)
