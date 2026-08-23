import logging
import uuid

from sqlalchemy.orm import Session

from app.api.schemas.scene_schema import SceneSchema
from app.application.services.person_recognition_service import PersonRecognitionService
from app.application.services.person_tools import (
    IDENTIFY_PERSONS_TOOL,
    PERSON_TOOLS,
    REGISTER_PERSON_TOOL,
)
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
from app.domain.entities.tool_call import ToolCall
from app.domain.protocols.conversation_repository import ConversationNotFoundError
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.tool_calling_vision_language_model import ToolCallingVisionLanguageModel
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.infrastructure.database.models import Conversation, DetectedObject, Message, SceneModel
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.message_repository import SqlAlchemyMessageRepository
from app.infrastructure.repositories.object_repository import SqlAlchemyObjectRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository

logger = logging.getLogger(__name__)


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
        person_recognition_service: PersonRecognitionService,
        skip_yolo_pipeline: bool = False,
    ) -> None:
        self._session = session
        self._image_storage = image_storage
        self._vision_language_model = vision_language_model
        self._model_metadata = model_metadata
        self._prompt_composer = prompt_composer
        self._person_recognition_service = person_recognition_service
        self._skip_yolo_pipeline = skip_yolo_pipeline
        self._conversation_repository = SqlAlchemyConversationRepository(session)
        self._message_repository = SqlAlchemyMessageRepository(session)
        self._object_repository = SqlAlchemyObjectRepository(session)
        self._scene_repository = SqlAlchemySceneRepository(session)

        # Capacidade opcional detectada estruturalmente (Protocol
        # @runtime_checkable) — só o Gemini a implementa. Nenhum mock de
        # teste que usa `spec=VisionLanguageModel` satisfaz isso, então o
        # fluxo de tool calling nunca ativa em testes existentes sem que
        # eles peçam explicitamente.
        self._tool_calling_vlm: ToolCallingVisionLanguageModel | None = (
            vision_language_model
            if isinstance(vision_language_model, ToolCallingVisionLanguageModel)
            else None
        )

    def ask(self, conversation_id: uuid.UUID, question: str) -> ConversationAnswer:
        conversation = self._get_conversation_or_raise(conversation_id)

        scene_row = self._scene_repository.get_by_id(conversation.scene_id)
        assert scene_row is not None  # garantido pela FK conversations.scene_id

        detected_object_rows = self._object_repository.list_by_scene_id(scene_row.id)
        image_bytes = self._image_storage.get(scene_row.image_storage_key)

        history_rows = self._message_repository.list_by_conversation_id(conversation_id)
        history = [ConversationMessage(role=row.role, content=row.content) for row in history_rows]

        try:
            self._message_repository.add(
                Message(conversation_id=conversation_id, role=MessageRole.USER, content=question)
            )

            if self._tool_calling_vlm is not None:
                answer_text, model_name, prompt_version, latency_ms = self._ask_with_tool_calling(
                    scene_row=scene_row,
                    conversation=conversation,
                    history=history,
                    image_bytes=image_bytes,
                    question=question,
                )
            else:
                scene_json = self._build_scene_json(
                    scene_row, detected_object_rows, conversation.id
                )
                system_prompt, prompt_version = self._prompt_composer.build(question)
                vlm_response = self._vision_language_model.ask(
                    image=image_bytes,
                    scene_json=scene_json,
                    system_prompt=system_prompt,
                    conversation_history=history,
                    question=question,
                )
                answer_text = vlm_response.text
                model_name = vlm_response.model
                latency_ms = round(vlm_response.duration_ms)

            self._message_repository.add(
                Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=answer_text,
                    model_name=model_name,
                    prompt_version=prompt_version,
                    latency_ms=latency_ms,
                )
            )

            # Commit explícito antes de retornar: se a VLM falhar, a
            # pergunta do usuário registrada acima também é revertida (o
            # turno inteiro é uma única unidade de trabalho).
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        referenced_objects = self._find_referenced_objects(
            detected_object_rows, question, answer_text
        )

        return ConversationAnswer(
            answer=answer_text,
            scene_id=scene_row.id,
            referenced_objects=referenced_objects,
        )

    def _build_scene_json(
        self,
        scene_row: SceneModel,
        detected_object_rows: list[DetectedObject],
        conversation_id: uuid.UUID,
    ) -> dict[str, object]:
        # YOLO desabilitado para o Gemini (ver SceneService) — sem
        # DetectedObjects não há nada de útil pra montar aqui, e o Gemini
        # descreve a cena diretamente pela imagem (economiza processamento).
        if self._skip_yolo_pipeline:
            return {}
        scene = self._build_scene(scene_row, detected_object_rows, conversation_id)
        return SceneSchema.from_domain(scene).model_dump(mode="json", by_alias=True)

    def _ask_with_tool_calling(
        self,
        *,
        scene_row: SceneModel,
        conversation: Conversation,
        history: list[ConversationMessage],
        image_bytes: bytes,
        question: str,
    ) -> tuple[str, str, str, int]:
        assert self._tool_calling_vlm is not None
        tool_system_prompt = self._prompt_composer.build_tool_system_prompt()

        response = self._tool_calling_vlm.ask_with_tools(
            image=image_bytes,
            system_prompt=tool_system_prompt,
            conversation_history=history,
            question=question,
            tools=PERSON_TOOLS,
        )
        latency_ms = round(response.duration_ms)

        if response.tool_call is None:
            assert response.text  # garantido pelo VLM: nunca None sem tool_call
            return response.text, response.model, "tool_calling_v1", latency_ms

        tool_call = response.tool_call
        logger.info(
            "tool call detected name=%s conversation_id=%s", tool_call.name, conversation.id
        )
        result_message = self._dispatch_tool_call(tool_call, scene_row, conversation, image_bytes)
        return result_message, response.model, f"tool_calling_v1:{tool_call.name}", latency_ms

    def _dispatch_tool_call(
        self,
        tool_call: ToolCall,
        scene_row: SceneModel,
        conversation: Conversation,
        image_bytes: bytes,
    ) -> str:
        if tool_call.name == REGISTER_PERSON_TOOL:
            name = str(tool_call.arguments.get("name", "")).strip()
            relationship = str(tool_call.arguments.get("relationship", "")).strip()
            if not name or not relationship:
                return "Não entendi o nome ou o relacionamento da pessoa. Pode repetir?"

            register_result = self._person_recognition_service.register_person(
                user_id=conversation.user_id,
                image=image_bytes,
                photo_storage_key=scene_row.image_storage_key,
                name=name,
                relationship=relationship,
            )
            return register_result.message

        if tool_call.name == IDENTIFY_PERSONS_TOOL:
            identify_result = self._person_recognition_service.identify_persons(
                user_id=conversation.user_id,
                image=image_bytes,
                image_width=scene_row.image_width,
                image_height=scene_row.image_height,
            )
            return identify_result.message

        logger.warning("unknown tool call name=%s", tool_call.name)
        return "Não consegui executar essa ação."

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
