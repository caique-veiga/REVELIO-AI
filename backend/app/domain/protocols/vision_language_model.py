from typing import Protocol

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.vlm_response import VLMResponse


class VisionLanguageModelError(Exception):
    """Erro base para falhas de comunicação com a VLM."""


class OllamaUnavailableError(VisionLanguageModelError):
    """O Ollama em si não pôde ser alcançado (conexão, timeout, erro de transporte)."""


class ModelUnavailableError(VisionLanguageModelError):
    """O Ollama está acessível, mas o modelo configurado não está disponível nele."""


class EmptyModelResponseError(VisionLanguageModelError):
    """O provider respondeu com sucesso, mas não veio nenhum texto de resposta.

    Causa observada no Ollama (ETAPA 13.1): modelos com "thinking" (ex.
    qwen3.5) podem consumir todo o orçamento de geração (`num_ctx`)
    raciocinando internamente antes de escrever a resposta final — se isso
    acontece, a geração é cortada (`done_reason=length`) antes do modelo
    sequer começar a escrever `content`. Não deve ser mascarado com um texto
    padrão; quem chama decide a política de retry/fallback.
    """


class VisionProviderUnavailableError(VisionLanguageModelError):
    """O provider (ex. Gemini) recusou ou não pôde atender a requisição:
    conexão recusada, chave de API inválida, rate limit ou erro 5xx.
    """


class VisionProviderTimeoutError(VisionLanguageModelError):
    """O provider não respondeu dentro do timeout configurado."""


class VisionLanguageModel(Protocol):
    def health_check(self) -> None:
        """Verifica se o Ollama está acessível e se o modelo configurado existe.

        Levanta OllamaUnavailableError ou ModelUnavailableError quando algo não
        está pronto; não retorna nada quando está tudo certo.
        """
        ...

    def ask(
        self,
        *,
        image: bytes,
        scene_json: dict[str, object],
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
    ) -> VLMResponse: ...
