import logging
import uuid

from sqlalchemy.orm import Session

from app.application.services.person_recognition_service import PersonRecognitionService
from app.application.services.person_tools import (
    IDENTIFY_PERSONS_TOOL,
    PERSON_TOOLS,
    REGISTER_PERSON_TOOL,
)
from app.application.services.prompt_composer import PromptComposer
from app.domain.entities.conversation_answer import ConversationAnswer
from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.message_role import MessageRole
from app.domain.entities.tool_call import ToolCall
from app.domain.protocols.conversation_repository import ConversationNotFoundError
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.infrastructure.database.models import Conversation, Message, SceneModel
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.message_repository import SqlAlchemyMessageRepository
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """Orquestra uma pergunta sobre a cena atual de uma conversation:

        recuperar conversation/scene/histórico -> montar contexto
        (SYSTEM PROMPT + IMAGE + HISTORY + PERGUNTA + TOOLS) -> VLM ->
        salvar user message + assistant message -> retornar resposta

    Pipeline unificado: Ollama e Gemini recebem exatamente a mesma coisa —
    imagem + histórico + pergunta + as tools de reconhecimento de pessoas
    (register_person/identify_persons). Nenhum recebe Scene JSON
    pré-processado (YOLO removido — ver PROMPT "Unificar Comportamento
    Gemini/Ollama"). O próprio modelo decide se responde direto ou chama
    uma tool.

    Nunca envia mensagens de outra conversation para a VLM — o histórico é
    sempre filtrado por `conversation_id` (regra fundamental de isolamento
    entre cenas, CLAUDE_CONTEXT.md §4).
    """

    def __init__(
        self,
        session: Session,
        image_storage: ImageStorage,
        vision_language_model: VisionLanguageModel,
        prompt_composer: PromptComposer,
        person_recognition_service: PersonRecognitionService,
    ) -> None:
        self._session = session
        self._image_storage = image_storage
        self._vision_language_model = vision_language_model
        self._prompt_composer = prompt_composer
        self._person_recognition_service = person_recognition_service
        self._conversation_repository = SqlAlchemyConversationRepository(session)
        self._message_repository = SqlAlchemyMessageRepository(session)
        self._scene_repository = SqlAlchemySceneRepository(session)

    def ask(self, conversation_id: uuid.UUID, question: str) -> ConversationAnswer:
        conversation = self._get_conversation_or_raise(conversation_id)

        scene_row = self._scene_repository.get_by_id(conversation.scene_id)
        assert scene_row is not None  # garantido pela FK conversations.scene_id
        image_bytes = self._image_storage.get(scene_row.image_storage_key)

        history_rows = self._message_repository.list_by_conversation_id(conversation_id)
        history = [ConversationMessage(role=row.role, content=row.content) for row in history_rows]

        try:
            self._message_repository.add(
                Message(conversation_id=conversation_id, role=MessageRole.USER, content=question)
            )

            system_prompt = self._prompt_composer.build_system_prompt()
            response = self._vision_language_model.ask(
                image=image_bytes,
                system_prompt=system_prompt,
                conversation_history=history,
                question=question,
                tools=PERSON_TOOLS,
            )
            latency_ms = round(response.duration_ms)

            if response.tool_call is None:
                assert response.text  # garantido pela VLM: nunca None sem tool_call
                answer_text = response.text
                prompt_version = "tool_calling_v1"
            else:
                tool_call = response.tool_call
                logger.info(
                    "tool call detected name=%s conversation_id=%s",
                    tool_call.name,
                    conversation.id,
                )
                answer_text = self._dispatch_tool_call(
                    tool_call, scene_row, conversation, image_bytes
                )
                prompt_version = f"tool_calling_v1:{tool_call.name}"

            self._message_repository.add(
                Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=answer_text,
                    model_name=response.model,
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

        return ConversationAnswer(answer=answer_text, scene_id=scene_row.id)

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
