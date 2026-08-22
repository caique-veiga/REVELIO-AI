import base64
import json
import logging
import time

import httpx

from app.domain.entities.conversation_message import ConversationMessage
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
        image_to_send = image
        original_size = len(image)
        if self._image_enable_optimization:
            image_to_send = optimize_image(
                image,
                max_dimension=self._image_max_dimension,
                jpeg_quality=self._image_jpeg_quality,
            )
        image_b64 = base64.b64encode(image_to_send).decode("ascii")
        scene_context = json.dumps(scene_json, ensure_ascii=False)

        contents: list[dict[str, object]] = [
            {"role": _ROLE_MAP.get(entry.role.value, "user"), "parts": [{"text": entry.content}]}
            for entry in conversation_history
        ]
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"text": f"Scene JSON:\n{scene_context}\n\nPergunta: {question}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        )

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
        }

        logger.info(
            "gemini request started model=%s original_bytes=%d sent_bytes=%d",
            self._model,
            original_size,
            len(image_to_send),
        )
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

        data = response.json()
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        answer = "".join(part.get("text", "") for part in parts)
        usage = data.get("usageMetadata", {})

        if not answer.strip():
            logger.error(
                "gemini returned empty content model=%s finish_reason=%s duration_ms=%.1f",
                self._model,
                candidates[0].get("finishReason") if candidates else None,
                duration_ms,
            )
            raise EmptyModelResponseError(f"Gemini retornou content vazio (model={self._model})")

        logger.info(
            "gemini request succeeded model=%s duration_ms=%.1f prompt_tokens=%s output_tokens=%s",
            self._model,
            duration_ms,
            usage.get("promptTokenCount"),
            usage.get("candidatesTokenCount"),
        )
        return VLMResponse(text=answer, model=self._model, duration_ms=duration_ms)
