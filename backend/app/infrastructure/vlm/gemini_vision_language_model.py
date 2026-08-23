import base64
import json
import logging
import time
from typing import cast

import httpx

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.tool_call import ToolCall, ToolCallResponse, ToolDefinition
from app.domain.entities.vlm_response import VLMResponse
from app.domain.protocols.vision_language_model import (
    EmptyModelResponseError,
    VisionLanguageModelError,
    VisionProviderTimeoutError,
    VisionProviderUnavailableError,
)
from app.infrastructure.vlm.image_optimizer import optimize_image

logger = logging.getLogger(__name__)

_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiVisionLanguageModel:
    """Implementação de VisionLanguageModel usando a API HTTP do Gemini
    (`generateContent`).

    Provider de fallback: usado quando o Ollama está desabilitado ou
    indisponível (ver FallbackVisionLanguageModel). Não faz nenhuma
    tentativa de fallback adicional por si só — se esta chamada falhar,
    quem chamou decide o que fazer.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_seconds: float = 30.0,
        image_max_dimension: int = 768,
        image_jpeg_quality: int = 85,
        image_enable_optimization: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._image_max_dimension = image_max_dimension
        self._image_jpeg_quality = image_jpeg_quality
        self._image_enable_optimization = image_enable_optimization
        self._client = client if client is not None else httpx.Client()

    def health_check(self) -> None:
        try:
            response = self._client.get(
                f"{self._base_url}/v1beta/models/{self._model}",
                params={"key": self._api_key},
                timeout=self._timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise VisionProviderUnavailableError(f"Gemini inacessível em {self._base_url}") from exc

        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise VisionProviderUnavailableError("Gemini API key inválida")
        if response.status_code != httpx.codes.OK:
            raise VisionProviderUnavailableError(
                f"Gemini respondeu HTTP {response.status_code} ao verificar "
                f"o modelo '{self._model}'"
            )

    def ask(
        self,
        *,
        image: bytes,
        scene_json: dict[str, object],
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
    ) -> VLMResponse:
        image_b64 = self._prepare_image(image)
        scene_context = json.dumps(scene_json, ensure_ascii=False) if scene_json else None
        # Sem Scene JSON (YOLO desabilitado para o Gemini): a imagem sozinha
        # já basta — ver PROMPT "Gemini Fallback Sem JSON YOLO".
        question_text = (
            f"Scene JSON:\n{scene_context}\n\nPergunta: {question}" if scene_context else question
        )

        contents = self._build_contents(conversation_history, question_text, image_b64)
        payload: dict[str, object] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
        }

        data, duration_ms = self._send(payload)
        answer, _tool_call = self._parse_parts(
            data
        )  # ask() nunca envia tools, então não deveria vir

        if not (answer or "").strip():
            logger.error(
                "gemini returned empty content model=%s duration_ms=%.1f",
                self._model,
                duration_ms,
            )
            raise EmptyModelResponseError(f"Gemini retornou content vazio (model={self._model})")
        assert answer is not None

        usage = cast(dict[str, object], data.get("usageMetadata", {}))
        logger.info(
            "gemini request succeeded model=%s duration_ms=%.1f prompt_tokens=%s output_tokens=%s",
            self._model,
            duration_ms,
            usage.get("promptTokenCount"),
            usage.get("candidatesTokenCount"),
        )
        return VLMResponse(text=answer, model=self._model, duration_ms=duration_ms)

    def ask_with_tools(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse:
        image_b64 = self._prepare_image(image)
        contents = self._build_contents(conversation_history, question, image_b64)
        payload: dict[str, object] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in tools
                    ]
                }
            ],
        }

        data, duration_ms = self._send(payload)
        answer, tool_call = self._parse_parts(data)

        if tool_call is None and not (answer or "").strip():
            logger.error(
                "gemini (tools) returned empty content model=%s duration_ms=%.1f",
                self._model,
                duration_ms,
            )
            raise EmptyModelResponseError(f"Gemini retornou content vazio (model={self._model})")

        logger.info(
            "gemini (tools) request succeeded model=%s duration_ms=%.1f tool_call=%s",
            self._model,
            duration_ms,
            tool_call.name if tool_call else None,
        )
        return ToolCallResponse(
            text=answer, tool_call=tool_call, model=self._model, duration_ms=duration_ms
        )

    def _prepare_image(self, image: bytes) -> str:
        image_to_send = image
        if self._image_enable_optimization:
            image_to_send = optimize_image(
                image,
                max_dimension=self._image_max_dimension,
                jpeg_quality=self._image_jpeg_quality,
            )
        return base64.b64encode(image_to_send).decode("ascii")

    @staticmethod
    def _build_contents(
        conversation_history: list[ConversationMessage], question_text: str, image_b64: str
    ) -> list[dict[str, object]]:
        contents: list[dict[str, object]] = [
            {"role": _ROLE_MAP.get(entry.role.value, "user"), "parts": [{"text": entry.content}]}
            for entry in conversation_history
        ]
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"text": question_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        )
        return contents

    def _send(self, payload: dict[str, object]) -> tuple[dict[str, object], float]:
        logger.info("gemini request started model=%s", self._model)
        started = time.monotonic()
        try:
            response = self._client.post(
                f"{self._base_url}/v1beta/models/{self._model}:generateContent",
                params={"key": self._api_key},
                json=payload,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning("gemini request timed out model=%s", self._model)
            raise VisionProviderTimeoutError(
                f"Gemini não respondeu em {self._timeout_seconds}s"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("gemini request failed model=%s error=%s", self._model, type(exc).__name__)
            raise VisionProviderUnavailableError(
                f"Não foi possível alcançar o Gemini: {exc}"
            ) from exc

        duration_ms = (time.monotonic() - started) * 1000

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise VisionProviderUnavailableError("Gemini API rate limited (429)")
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise VisionProviderUnavailableError("Gemini API key inválida (401)")
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise VisionProviderUnavailableError(
                f"Gemini indisponível (HTTP {response.status_code})"
            )
        if response.status_code != httpx.codes.OK:
            raise VisionLanguageModelError(
                f"Gemini retornou HTTP {response.status_code}: {response.text}"
            )

        result: dict[str, object] = response.json()
        return result, duration_ms

    @staticmethod
    def _parse_parts(data: dict[str, object]) -> tuple[str | None, ToolCall | None]:
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []  # type: ignore[index]

        tool_call: ToolCall | None = None
        text_chunks: list[str] = []
        for part in parts:
            if "functionCall" in part:
                function_call = part["functionCall"]
                tool_call = ToolCall(
                    name=function_call["name"], arguments=function_call.get("args", {})
                )
            elif "text" in part:
                text_chunks.append(part["text"])

        text = "".join(text_chunks) if text_chunks else None
        return text, tool_call
