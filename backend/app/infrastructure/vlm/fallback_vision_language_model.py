import logging

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.tool_call import ToolCallResponse, ToolDefinition
from app.domain.entities.vlm_response import VLMResponse
from app.domain.protocols.tool_calling_vision_language_model import ToolCallingVisionLanguageModel
from app.domain.protocols.vision_language_model import VisionLanguageModel, VisionLanguageModelError

logger = logging.getLogger(__name__)


class FallbackVisionLanguageModel:
    """Tenta `primary` e, se falhar, tenta `fallback` — sem mais níveis.

    Implementa o mesmo Protocol `VisionLanguageModel` que as implementações
    concretas, então ConversationService continua dependendo só da
    interface e não sabe que existe um fallback por trás dela.

    `ask_with_tools` é sempre delegado direto para `fallback` (o Gemini): o
    Ollama/qwen3.5 não participa de tool calling nesta etapa — isso faz este
    wrapper satisfazer `ToolCallingVisionLanguageModel` estruturalmente,
    independente de o Ollama estar habilitado ou não.
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
        scene_json: dict[str, object],
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
    ) -> VLMResponse:
        try:
            return self._primary.ask(
                image=image,
                scene_json=scene_json,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                question=question,
            )
        except VisionLanguageModelError as exc:
            logger.warning("primary vlm failed (%s), falling back: %s", type(exc).__name__, exc)
            return self._fallback.ask(
                image=image,
                scene_json=scene_json,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                question=question,
            )

    def ask_with_tools(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse:
        assert isinstance(self._fallback, ToolCallingVisionLanguageModel)
        return self._fallback.ask_with_tools(
            image=image,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            question=question,
            tools=tools,
        )
