from unittest.mock import MagicMock

import pytest

from app.domain.entities.vlm_response import VLMResponse
from app.domain.protocols.vision_language_model import (
    OllamaUnavailableError,
    VisionLanguageModel,
    VisionProviderUnavailableError,
)
from app.infrastructure.vlm.fallback_vision_language_model import FallbackVisionLanguageModel


def _ask(vlm: FallbackVisionLanguageModel) -> VLMResponse:
    return vlm.ask(
        image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q"
    )


def test_ask_uses_primary_when_it_succeeds() -> None:
    primary = MagicMock(spec=VisionLanguageModel)
    primary.ask.return_value = VLMResponse(text="ok", model="qwen3.5:4b", duration_ms=10.0)
    fallback = MagicMock(spec=VisionLanguageModel)

    vlm = FallbackVisionLanguageModel(primary=primary, fallback=fallback)
    response = _ask(vlm)

    assert response.model == "qwen3.5:4b"
    fallback.ask.assert_not_called()


def test_ask_falls_back_when_primary_fails() -> None:
    primary = MagicMock(spec=VisionLanguageModel)
    primary.ask.side_effect = OllamaUnavailableError("ollama indisponível")
    fallback = MagicMock(spec=VisionLanguageModel)
    fallback.ask.return_value = VLMResponse(
        text="ok via gemini", model="gemini-2.5-flash-lite", duration_ms=20.0
    )

    vlm = FallbackVisionLanguageModel(primary=primary, fallback=fallback)
    response = _ask(vlm)

    assert response.model == "gemini-2.5-flash-lite"
    fallback.ask.assert_called_once()


def test_ask_raises_when_both_primary_and_fallback_fail() -> None:
    primary = MagicMock(spec=VisionLanguageModel)
    primary.ask.side_effect = OllamaUnavailableError("ollama indisponível")
    fallback = MagicMock(spec=VisionLanguageModel)
    fallback.ask.side_effect = VisionProviderUnavailableError("gemini indisponível")

    vlm = FallbackVisionLanguageModel(primary=primary, fallback=fallback)
    with pytest.raises(VisionProviderUnavailableError):
        _ask(vlm)


def test_health_check_delegates_to_fallback() -> None:
    primary = MagicMock(spec=VisionLanguageModel)
    fallback = MagicMock(spec=VisionLanguageModel)

    vlm = FallbackVisionLanguageModel(primary=primary, fallback=fallback)
    vlm.health_check()

    fallback.health_check.assert_called_once()
    primary.health_check.assert_not_called()
