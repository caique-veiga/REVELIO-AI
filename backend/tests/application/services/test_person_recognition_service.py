from unittest.mock import MagicMock

import numpy as np
from sqlalchemy.orm import Session

from app.application.services.person_recognition_service import PersonRecognitionService
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.face_encoding import FaceEncoding
from app.domain.protocols.face_encoder import FaceEncoder
from app.domain.services.face_matcher import FaceMatcher
from app.domain.services.position_analyzer import PositionAnalyzer
from app.infrastructure.database.models import Person, PersonPhoto, User


def _make_service(db_session: Session, face_encoder: MagicMock) -> PersonRecognitionService:
    return PersonRecognitionService(
        session=db_session,
        face_encoder=face_encoder,
        position_analyzer=PositionAnalyzer(),
        face_matcher=FaceMatcher(threshold=0.5),
    )


def _face(
    x1: int = 0, y1: int = 0, x2: int = 100, y2: int = 100, vector: list[float] | None = None
) -> FaceEncoding:
    return FaceEncoding(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        embedding=np.array(vector or [1.0, 0.0, 0.0]),
    )


def test_register_person_single_person_success(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = [_face()]
    service = _make_service(db_session, face_encoder)

    result = service.register_person(
        user_id=user.id,
        image=b"fake-jpeg",
        photo_storage_key="2026/08/22/scene.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    assert result.success is True
    assert "Maria" in result.message
    assert result.person_id is not None

    person = db_session.get(Person, result.person_id)
    assert person is not None
    assert person.name == "Maria"
    assert person.relationship_label == "minha irmã"


def test_register_person_multiple_people_error(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = [_face(), _face(x1=200, x2=300)]
    service = _make_service(db_session, face_encoder)

    result = service.register_person(
        user_id=user.id,
        image=b"fake-jpeg",
        photo_storage_key="key.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    assert result.success is False
    assert "2 rostos" in result.message
    assert db_session.query(Person).count() == 0


def test_register_person_no_people_error(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = []
    service = _make_service(db_session, face_encoder)

    result = service.register_person(
        user_id=user.id,
        image=b"fake-jpeg",
        photo_storage_key="key.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    assert result.success is False
    assert db_session.query(Person).count() == 0


def test_register_person_embedding_saved_correctly(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    embedding = np.array([0.1, 0.2, 0.3])
    face_encoder.detect_and_encode.return_value = [_face(vector=embedding.tolist())]
    service = _make_service(db_session, face_encoder)

    result = service.register_person(
        user_id=user.id,
        image=b"fake-jpeg",
        photo_storage_key="key.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    photo = db_session.query(PersonPhoto).filter_by(person_id=result.person_id).one()
    stored_embedding = np.frombuffer(photo.photo_embedding, dtype=np.float64)
    assert np.allclose(stored_embedding, embedding)
    assert photo.face_roi == {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    assert photo.is_primary is True


def test_identify_persons_finds_registered_person(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = [_face(vector=[1.0, 0.0, 0.0])]
    service = _make_service(db_session, face_encoder)
    service.register_person(
        user_id=user.id,
        image=b"x",
        photo_storage_key="k.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    result = service.identify_persons(
        user_id=user.id, image=b"x", image_width=300, image_height=200
    )

    assert result.identified_count == 1
    assert result.total_faces == 1
    assert "Maria" in result.message
    assert "minha irmã" in result.message


def test_identify_persons_all_unknown(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = [_face(vector=[1.0, 0.0, 0.0])]
    service = _make_service(db_session, face_encoder)
    # ninguém cadastrado

    result = service.identify_persons(
        user_id=user.id, image=b"x", image_width=300, image_height=200
    )

    assert result.identified_count == 0
    assert result.total_faces == 1
    assert "desconhecida" in result.message


def test_identify_persons_multiple_people_mixed(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = [_face(vector=[1.0, 0.0, 0.0])]
    service = _make_service(db_session, face_encoder)
    service.register_person(
        user_id=user.id,
        image=b"x",
        photo_storage_key="k.jpg",
        name="Maria",
        relationship="minha irmã",
    )

    # Segunda pergunta: uma cara conhecida (Maria) + uma desconhecida
    face_encoder.detect_and_encode.return_value = [
        _face(x1=0, x2=100, vector=[1.0, 0.0, 0.0]),
        _face(x1=250, x2=300, vector=[0.0, 1.0, 0.0]),
    ]

    result = service.identify_persons(
        user_id=user.id, image=b"x", image_width=300, image_height=200
    )

    assert result.total_faces == 2
    assert result.identified_count == 1
    assert "Maria" in result.message
    assert "desconhecida" in result.message


def test_identify_persons_no_faces_detected(db_session: Session, user: User) -> None:
    face_encoder = MagicMock(spec=FaceEncoder)
    face_encoder.detect_and_encode.return_value = []
    service = _make_service(db_session, face_encoder)

    result = service.identify_persons(
        user_id=user.id, image=b"x", image_width=300, image_height=200
    )

    assert result.total_faces == 0
    assert result.identified_count == 0
    assert result.success is True
