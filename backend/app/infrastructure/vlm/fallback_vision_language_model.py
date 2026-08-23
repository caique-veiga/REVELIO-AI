import logging

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.tool_call import ToolCallResponse, ToolDefinition
from app.domain.protocols.vision_language_model import VisionLanguageModel, VisionLanguageModelError

logger = logging.getLogger(__name__)


class FallbackVisionLanguageModel:
    """Tenta `primary` e, se falhar, tenta `fallback` — sem mais níveis.

    Implementa o mesmo Protocol `VisionLanguageModel` que as implementações
    concretas, então ConversationService continua dependendo só da
    interface e não sabe que existe um fallback por trás dela. Isso vale
    também para tool calling: se o Ollama/qwen retornar algo inesperado
    (tool call malformado, resposta vazia), cai automaticamente para o
    Gemini, do mesmo jeito que já fazia para respostas de texto simples.
    """

    def __init__(self, primary: VisionLanguageModel, fallback: VisionLanguageModel) -> None:
        self._primary = primary
        self._fallback = fallback

    def health_check(self) -> None:
        self._fallback.health_check()

    def ask(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse:
        try:
            return self._primary.ask(
                image=image,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                question=question,
                tools=tools,
            )
        except VisionLanguageModelError as exc:
            logger.warning("primary vlm failed (%s), falling back: %s", type(exc).__name__, exc)
            return self._fallback.ask(
                image=image,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                question=question,
                tools=tools,
            )
