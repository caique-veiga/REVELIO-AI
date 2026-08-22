import os

import pytest

from app.config.settings import get_settings
from app.infrastructure.vlm.ollama_vision_language_model import OllamaVisionLanguageModel

RUN_INTEGRATION_TEST = os.environ.get("OLLAMA_INTEGRATION_TEST", "false").lower() == "true"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION_TEST,
    reason="OLLAMA_INTEGRATION_TEST não é 'true' — pulando teste contra o PC GPU real.",
)


def test_ask_against_real_ollama_instance(jpeg_bytes: bytes) -> None:
    settings = get_settings()
    vlm = OllamaVisionLanguageModel(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        max_retries=settings.ollama_max_retries,
    )

    vlm.health_check()

    response = vlm.ask(
        image=jpeg_bytes,
        scene_json={"scene_id": "integration-test", "objects": []},
        system_prompt="Você é um assistente visual objetivo e conciso.",
        conversation_history=[],
        question="Descreva brevemente o que você vê.",
    )

    assert response.text
    assert response.model == settings.ollama_model
    assert response.duration_ms > 0
