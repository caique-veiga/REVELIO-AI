import logging
import uuid

from sqlalchemy.orm import Session

from app.domain.entities.person_recognition_result import (
    IdentifiedPerson,
    IdentifyPersonsResult,
    RegisterPersonResult,
)
from app.domain.protocols.face_encoder import FaceEncoder
from app.domain.services.face_matcher import FaceMatcher
from app.domain.services.position_analyzer import PositionAnalyzer
from app.infrastructure.database.models import Person, PersonPhoto
from app.infrastructure.repositories.person_photo_repository import SqlAlchemyPersonPhotoRepository
from app.infrastructure.repositories.person_repository import SqlAlchemyPersonRepository

logger = logging.getLogger(__name__)

_HORIZONTAL_POSITION_PT = {
    "left": "à esquerda",
    "center": "ao centro",
    "right": "à direita",
}


class PersonRecognitionService:
    """Orquestra cadastro e identificação de pessoas por reconhecimento facial.

    Reaproveita a imagem já salva da Scene atual (não salva uma cópia nova)
    — quem chama passa `photo_storage_key` da Scene em curso.

    Não usa o YOLO como filtro: o detector genérico do COCO (yolov8n) errava
    negativo em selfies reais (ângulo/enquadramento de foto tirada com a
    câmera frontal), bloqueando cadastro/identificação mesmo com uma pessoa
    de verdade na foto. Quem decide se há um rosto é só o FaceEncoder —
    aceitar um falso-positivo ocasional (ex. cachorro) é a troca aceita.
    """

    def __init__(
        self,
        session: Session,
        face_encoder: FaceEncoder,
        position_analyzer: PositionAnalyzer,
        face_matcher: FaceMatcher,
    ) -> None:
        self._session = session
        self._face_encoder = face_encoder
        self._position_analyzer = position_analyzer
        self._face_matcher = face_matcher
        self._person_repository = SqlAlchemyPersonRepository(session)
        self._person_photo_repository = SqlAlchemyPersonPhotoRepository(session)

    def register_person(
        self,
        *,
        user_id: uuid.UUID,
        image: bytes,
        photo_storage_key: str,
        name: str,
        relationship: str,
    ) -> RegisterPersonResult:
        faces = self._face_encoder.detect_and_encode(image)

        if len(faces) == 0:
            return RegisterPersonResult(
                success=False,
                message=(
                    "Não detectei nenhum rosto na foto. Tire uma foto mais clara e tente novamente."
                ),
            )
        if len(faces) > 1:
            return RegisterPersonResult(
                success=False,
                message=(
                    f"Detectei {len(faces)} rostos na foto. "
                    "Tire uma foto com apenas uma pessoa para cadastro."
                ),
            )

        face = faces[0]
        try:
            person = self._person_repository.add(
                Person(user_id=user_id, name=name, relationship_label=relationship)
            )
            self._person_photo_repository.add(
                PersonPhoto(
                    person_id=person.id,
                    photo_storage_key=photo_storage_key,
                    photo_embedding=face.embedding.astype("float64").tobytes(),
                    face_roi={
                        "x1": face.bbox.x1,
                        "y1": face.bbox.y1,
                        "x2": face.bbox.x2,
                        "y2": face.bbox.y2,
                    },
                    is_primary=True,
                )
            )
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        logger.info(
            "person registered name=%s relationship=%s user_id=%s person_id=%s",
            name,
            relationship,
            user_id,
            person.id,
        )
        return RegisterPersonResult(
            success=True,
            message=(
                f"Cadastrei {name} como {relationship}. "
                'Você pode perguntar "quem é essa pessoa?" depois.'
            ),
            person_id=person.id,
        )

    def identify_persons(
        self, *, user_id: uuid.UUID, image: bytes, image_width: int, image_height: int
    ) -> IdentifyPersonsResult:
        faces = self._face_encoder.detect_and_encode(image)

        if len(faces) == 0:
            return IdentifyPersonsResult(
                success=True,
                message="Não vejo nenhuma pessoa na cena.",
                identified_count=0,
                total_faces=0,
            )

        candidates = self._person_photo_repository.list_embeddings_by_user(user_id)

        identified: list[IdentifiedPerson] = []
        for face in faces:
            position = self._position_analyzer.analyze(face.bbox, image_width, image_height)
            match = self._face_matcher.find_best_match(face.embedding, candidates)
            if match is not None:
                identified.append(
                    IdentifiedPerson(
                        name=match.name,
                        relationship=match.relationship,
                        similarity=match.similarity,
                        horizontal_position=position.horizontal.value,
                    )
                )
            else:
                identified.append(
                    IdentifiedPerson(
                        name=None,
                        relationship=None,
                        similarity=0.0,
                        horizontal_position=position.horizontal.value,
                    )
                )

        identified_count = sum(1 for person in identified if person.name is not None)
        message = self._format_message(identified)

        logger.info(
            "identify_persons user_id=%s faces=%d identified=%d",
            user_id,
            len(faces),
            identified_count,
        )
        return IdentifyPersonsResult(
            success=True,
            message=message,
            identified_count=identified_count,
            total_faces=len(faces),
        )

    @staticmethod
    def _format_message(identified: list[IdentifiedPerson]) -> str:
        parts = [
            f"{person.name} ({person.relationship}) "
            f"{_HORIZONTAL_POSITION_PT[person.horizontal_position]}"
            if person.name is not None
            else f"uma pessoa desconhecida {_HORIZONTAL_POSITION_PT[person.horizontal_position]}"
            for person in identified
        ]
        return "Vejo " + ", ".join(parts) + "."
