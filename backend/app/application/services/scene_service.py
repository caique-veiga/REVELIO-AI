import dataclasses
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.scene import Scene
from app.domain.protocols.color_analyzer import ColorAnalyzer
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.object_detector import ObjectDetector
from app.domain.services.position_analyzer import PositionAnalyzer
from app.domain.services.scene_builder import SceneBuilder
from app.infrastructure.database.models import Conversation, DetectedObject, SceneModel, User
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.object_repository import SqlAlchemyObjectRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


class SceneService:
    """Orquestra o fluxo completo de criação de uma cena:

        validar imagem -> criar scene_id -> salvar imagem -> YOLO ->
        PositionAnalyzer -> ColorAnalyzer -> SceneBuilder -> PostgreSQL
        (Scene, Conversation, DetectedObjects)

    Cada chamada a `create_scene` sempre cria uma nova Scene e uma nova
    Conversation — nunca reaproveita uma conversation existente.
    """

    def __init__(
        self,
        session: Session,
        image_storage: ImageStorage,
        object_detector: ObjectDetector,
        position_analyzer: PositionAnalyzer,
        color_analyzer: ColorAnalyzer,
        scene_builder: SceneBuilder,
        model_metadata: ModelMetadata,
    ) -> None:
        self._session = session
        self._image_storage = image_storage
        self._object_detector = object_detector
        self._position_analyzer = position_analyzer
        self._color_analyzer = color_analyzer
        self._scene_builder = scene_builder
        self._model_metadata = model_metadata
        self._scene_repository = SqlAlchemySceneRepository(session)
        self._conversation_repository = SqlAlchemyConversationRepository(session)
        self._object_repository = SqlAlchemyObjectRepository(session)

    def create_scene(self, filename: str, content: bytes) -> Scene:
        scene_id = uuid.uuid4()

        stored_image = self._image_storage.save(scene_id, filename, content)

        detections = self._detect_and_enrich(content, stored_image.width, stored_image.height)

        scene = self._scene_builder.build(
            image=stored_image, model=self._model_metadata, detections=detections
        )

        scene_row = self._scene_repository.add(
            SceneModel(
                id=scene_id,
                image_storage_key=stored_image.storage_key,
                image_filename=stored_image.filename,
                image_mime_type=stored_image.mime_type,
                image_width=stored_image.width,
                image_height=stored_image.height,
                image_size_bytes=stored_image.size_bytes,
                image_hash=stored_image.sha256,
            )
        )

        user = self._get_or_create_default_user()
        conversation_row = self._conversation_repository.add(
            Conversation(user_id=user.id, scene_id=scene_row.id)
        )

        self._object_repository.add_many(
            [self._to_detected_object_row(scene_row.id, detection) for detection in detections]
        )

        return dataclasses.replace(
            scene, scene_id=scene_row.id, conversation_id=conversation_row.id
        )

    def _detect_and_enrich(self, content: bytes, width: int, height: int) -> list[Detection]:
        detections = self._object_detector.detect(content)

        enriched: list[Detection] = []
        for detection in detections:
            position = self._position_analyzer.analyze(detection.bbox, width, height)
            color = self._color_analyzer.analyze(content, detection.bbox)
            enriched.append(dataclasses.replace(detection, position=position, color=color))

        return enriched

    def _get_or_create_default_user(self) -> User:
        # Ainda não há autenticação/gestão de usuários (fora de escopo desta
        # etapa) — reutiliza um único usuário padrão como dono provisório de
        # todas as conversations, até existir um fluxo real de identidade.
        user = self._session.execute(select(User).limit(1)).scalar_one_or_none()
        if user is None:
            user = User()
            self._session.add(user)
            self._session.flush()
        return user

    @staticmethod
    def _to_detected_object_row(scene_id: uuid.UUID, detection: Detection) -> DetectedObject:
        assert detection.position is not None
        assert detection.color is not None

        return DetectedObject(
            scene_id=scene_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox_x1=detection.bbox.x1,
            bbox_y1=detection.bbox.y1,
            bbox_x2=detection.bbox.x2,
            bbox_y2=detection.bbox.y2,
            position_horizontal=detection.position.horizontal.value,
            position_vertical=detection.position.vertical.value,
            position_region=detection.position.region.value,
            color_name=detection.color.name.value,
            color_r=detection.color.rgb[0],
            color_g=detection.color.rgb[1],
            color_b=detection.color.rgb[2],
            color_confidence=detection.color.confidence,
        )
