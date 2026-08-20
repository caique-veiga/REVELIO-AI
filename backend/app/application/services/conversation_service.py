import uuid

from sqlalchemy.orm import Session

from app.api.schemas.scene_schema import SceneSchema
from app.application.services.prompt_composer import PromptComposer
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName, ColorResult
from app.domain.entities.conversation_answer import ConversationAnswer, ReferencedObject
from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.detection import Detection
from app.domain.entities.message_role import MessageRole
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.position import HorizontalPosition, Position, Region, VerticalPosition
from app.domain.entities.scene import Scene
from app.domain.entities.stored_image import StoredImage
from app.domain.protocols.conversation_repository import ConversationNotFoundError
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.infrastructure.database.models import Conversation, DetectedObject, Message, SceneModel
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.message_repository import SqlAlchemyMessageRepository
from app.infrastructure.repositories.object_repository import SqlAlchemyObjectRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository


class ConversationService:
    """Orquestra uma pergunta sobre a cena atual de uma conversation:

        recuperar conversation/scene/histórico -> montar contexto
        (SYSTEM PROMPT + IMAGE + SCENE JSON + HISTORY + PERGUNTA) -> VLM ->
        salvar user message + assistant message -> retornar resposta

    Nunca envia mensagens de outra conversation para a VLM — o histórico é
    sempre filtrado por `conversation_id` (regra fundamental de isolamento
    entre cenas, CLAUDE_CONTEXT.md §4).
    """

    def __init__(
        self,
        session: Session,
        image_storage: ImageStorage,
        vision_language_model: VisionLanguageModel,
        model_metadata: ModelMetadata,
        prompt_composer: PromptComposer,
    ) -> None:
        self._session = session
        self._image_storage = image_storage
        self._vision_language_model = vision_language_model
        self._model_metadata = model_metadata
        self._prompt_composer = prompt_composer
        self._conversation_repository = SqlAlchemyConversationRepository(session)
        self._message_repository = SqlAlchemyMessageRepository(session)
        self._object_repository = SqlAlchemyObjectRepository(session)
        self._scene_repository = SqlAlchemySceneRepository(session)

    def ask(self, conversation_id: uuid.UUID, question: str) -> ConversationAnswer:
        conversation = self._get_conversation_or_raise(conversation_id)

        scene_row = self._scene_repository.get_by_id(conversation.scene_id)
        assert scene_row is not None  # garantido pela FK conversations.scene_id

        detected_object_rows = self._object_repository.list_by_scene_id(scene_row.id)
        scene = self._build_scene(scene_row, detected_object_rows, conversation.id)
        scene_json = SceneSchema.from_domain(scene).model_dump(mode="json", by_alias=True)

        image_bytes = self._image_storage.get(scene_row.image_storage_key)

        history_rows = self._message_repository.list_by_conversation_id(conversation_id)
        history = [ConversationMessage(role=row.role, content=row.content) for row in history_rows]

        system_prompt, prompt_version = self._prompt_composer.build(question)

        self._message_repository.add(
            Message(conversation_id=conversation_id, role=MessageRole.USER, content=question)
        )

        vlm_response = self._vision_language_model.ask(
            image=image_bytes,
            scene_json=scene_json,
            system_prompt=system_prompt,
            conversation_history=history,
            question=question,
        )

        self._message_repository.add(
            Message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=vlm_response.text,
                model_name=vlm_response.model,
                prompt_version=prompt_version,
                latency_ms=round(vlm_response.duration_ms),
            )
        )

        referenced_objects = self._find_referenced_objects(
            detected_object_rows, question, vlm_response.text
        )

        return ConversationAnswer(
            answer=vlm_response.text,
            scene_id=scene_row.id,
            referenced_objects=referenced_objects,
        )

    def get_conversation(self, conversation_id: uuid.UUID) -> tuple[Conversation, list[Message]]:
        conversation = self._get_conversation_or_raise(conversation_id)
        messages = self._message_repository.list_by_conversation_id(conversation_id)
        return conversation, messages

    def _get_conversation_or_raise(self, conversation_id: uuid.UUID) -> Conversation:
        conversation = self._conversation_repository.get_by_id(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation não encontrada: {conversation_id}")
        return conversation

    def _build_scene(
        self,
        scene_row: SceneModel,
        detected_object_rows: list[DetectedObject],
        conversation_id: uuid.UUID,
    ) -> Scene:
        # O nome/task/dataset do modelo de detecção não é persistido por Scene
        # (ver ETAPA 08) — reutiliza a configuração atual, assumindo que não
        # muda entre o momento da captura e o momento da pergunta.
        image = StoredImage(
            storage_key=scene_row.image_storage_key,
            filename=scene_row.image_filename,
            mime_type=scene_row.image_mime_type,
            size_bytes=scene_row.image_size_bytes,
            width=scene_row.image_width,
            height=scene_row.image_height,
            sha256=scene_row.image_hash or "",
        )

        return Scene(
            image=image,
            model=self._model_metadata,
            objects=[self._to_detection(row) for row in detected_object_rows],
            scene_id=scene_row.id,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _to_detection(row: DetectedObject) -> Detection:
        return Detection(
            class_id=row.class_id,
            class_name=row.class_name,
            confidence=row.confidence,
            bbox=BoundingBox(x1=row.bbox_x1, y1=row.bbox_y1, x2=row.bbox_x2, y2=row.bbox_y2),
            object_id=row.id,
            position=Position(
                horizontal=HorizontalPosition(row.position_horizontal),
                vertical=VerticalPosition(row.position_vertical),
                region=Region(row.position_region),
            ),
            color=ColorResult(
                name=ColorName(row.color_name),
                rgb=(row.color_r, row.color_g, row.color_b),
                confidence=row.color_confidence,
            ),
        )

    @staticmethod
    def _find_referenced_objects(
        detected_object_rows: list[DetectedObject], question: str, answer: str
    ) -> list[ReferencedObject]:
        # Heurística simples e deliberadamente limitada: considera "referenciado"
        # todo objeto detectado cuja classe apareça, como substring, na pergunta
        # ou na resposta. Não é NLP/entity-linking real (fora de escopo).
        text = f"{question} {answer}".lower()
        return [
            ReferencedObject(object_id=row.id, class_name=row.class_name)
            for row in detected_object_rows
            if row.class_name.lower() in text
        ]
