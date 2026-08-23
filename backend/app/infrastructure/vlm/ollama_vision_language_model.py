import base64
import json
import logging
import time

import httpx

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.tool_call import ToolCall, ToolCallResponse, ToolDefinition
from app.domain.protocols.vision_language_model import (
    EmptyModelResponseError,
    ModelUnavailableError,
    OllamaUnavailableError,
    VisionLanguageModelError,
)

logger = logging.getLogger(__name__)


def _model_matches(configured: str, available: list[str]) -> bool:
    if configured in available:
        return True
    configured_name = configured.split(":", 1)[0]
    return any(name.split(":", 1)[0] == configured_name for name in available)


class OllamaVisionLanguageModel:
    """Implementação de VisionLanguageModel usando a API HTTP do Ollama.

    Não assume que o Ollama roda na mesma máquina — o endereço vem sempre de
    `base_url` (configurado externamente via OLLAMA_BASE_URL).

    Usa o suporte nativo a tool calling da API do Ollama (`tools` no
    payload, `message.tool_calls` na resposta) — ver PROMPT "Unificar
    Comportamento Gemini/Ollama". Não há garantia de que todo modelo vision
    rodando localmente suporte tool calling de forma tão confiável quanto o
    Gemini; um tool call malformado ou ausente quando esperado vira
    VisionLanguageModelError, e quem chama (FallbackVisionLanguageModel) cai
    para o Gemini automaticamente.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        num_ctx: int = 8192,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._num_ctx = num_ctx
        self._client = client if client is not None else httpx.Client()

    def health_check(self) -> None:
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=self._timeout_seconds)
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(f"Ollama inacessível em {self._base_url}") from exc

        if response.status_code != httpx.codes.OK:
            raise OllamaUnavailableError(
                f"Ollama respondeu HTTP {response.status_code} em {self._base_url}"
            )

        available = [entry["name"] for entry in response.json().get("models", [])]
        if not _model_matches(self._model, available):
            raise ModelUnavailableError(
                f"Modelo '{self._model}' não encontrado no Ollama (disponíveis: {available})"
            )

    def ask(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse:
        self.health_check()

        image_b64 = base64.b64encode(image).decode("ascii")

        messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": entry.role.value, "content": entry.content} for entry in conversation_history
        )
        messages.append({"role": "user", "content": question, "images": [image_b64]})

        # `num_ctx` explícito: sem isso o Ollama usa um padrão pequeno
        # (observado: 4096 tokens no total prompt+geração). Modelos com
        # "thinking" (qwen3.5) podem gastar todo esse orçamento raciocinando
        # e nunca chegar a escrever `content` — ver EmptyModelResponseError.
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        started = time.monotonic()
        last_error: Exception | None = None
        attempts = self._max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/api/chat", json=payload, timeout=self._timeout_seconds
                )
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                logger.warning(
                    "ollama request failed (transient) model=%s attempt=%d/%d error=%s",
                    self._model,
                    attempt,
                    attempts,
                    type(exc).__name__,
                )
                continue
            except httpx.HTTPStatusError as exc:
                duration_ms = (time.monotonic() - started) * 1000
                logger.error(
                    "ollama request failed model=%s status=http_error code=%d duration_ms=%.1f",
                    self._model,
                    exc.response.status_code,
                    duration_ms,
                )
                raise VisionLanguageModelError(
                    f"Ollama retornou HTTP {exc.response.status_code}"
                ) from exc

            duration_ms = (time.monotonic() - started) * 1000
            data = response.json()
            answer, tool_call = self._parse_message(data)
            done_reason = data.get("done_reason")

            if tool_call is None and not (answer or "").strip():
                logger.error(
                    "ollama returned empty content model=%s done_reason=%s "
                    "eval_count=%s prompt_eval_count=%s duration_ms=%.1f",
                    self._model,
                    done_reason,
                    data.get("eval_count"),
                    data.get("prompt_eval_count"),
                    duration_ms,
                )
                raise EmptyModelResponseError(
                    f"Ollama retornou content vazio (done_reason={done_reason})"
                )

            logger.info(
                "ollama request succeeded model=%s duration_ms=%.1f attempt=%d "
                "done_reason=%s tool_call=%s",
                self._model,
                duration_ms,
                attempt,
                done_reason,
                tool_call.name if tool_call else None,
            )
            return ToolCallResponse(
                text=answer, tool_call=tool_call, model=self._model, duration_ms=duration_ms
            )

        duration_ms = (time.monotonic() - started) * 1000
        logger.error(
            "ollama request unavailable model=%s duration_ms=%.1f attempts=%d",
            self._model,
            duration_ms,
            attempts,
        )
        raise OllamaUnavailableError(
            f"Ollama inacessível em {self._base_url} após {attempts} tentativas"
        ) from last_error

    @staticmethod
    def _parse_message(data: dict[str, object]) -> tuple[str | None, ToolCall | None]:
        message = data.get("message")
        if not isinstance(message, dict):
            return None, None

        tool_call: ToolCall | None = None
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            if name:
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_call = ToolCall(
                    name=name, arguments=arguments if isinstance(arguments, dict) else {}
                )

        content = message.get("content")
        text = content if isinstance(content, str) and content.strip() else None
        return text, tool_call
