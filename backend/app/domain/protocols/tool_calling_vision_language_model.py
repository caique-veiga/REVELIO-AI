from typing import Protocol, runtime_checkable

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.tool_call import ToolCallResponse, ToolDefinition


@runtime_checkable
class ToolCallingVisionLanguageModel(Protocol):
    """Capacidade adicional de VLM: detectar intenção e chamar uma tool.

    Só o Gemini implementa isso nesta etapa — o Ollama/qwen3.5 não participa
    de tool calling (fora de escopo). `ConversationService` detecta a
    capacidade via `isinstance` (Protocol `@runtime_checkable`), então nada
    muda para quem só implementa `VisionLanguageModel` puro.

    Não recebe `scene_json`: o fluxo de tool calling é exclusivo do Gemini,
    que já descreve a cena diretamente pela imagem (ver PROMPT "Gemini
    Fallback Sem JSON YOLO + Face Recognition").
    """

    def ask_with_tools(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse: ...
