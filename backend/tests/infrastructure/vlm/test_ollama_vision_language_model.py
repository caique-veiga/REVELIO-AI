import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.message_role import MessageRole
from app.domain.protocols.vision_language_model import (
    EmptyModelResponseError,
    ModelUnavailableError,
    OllamaUnavailableError,
    VisionLanguageModelError,
)
from app.infrastructure.vlm.ollama_vision_language_model import OllamaVisionLanguageModel

BASE_URL = "http://100.118.123.0:11434"
MODEL = "qwen3.5:4b"


def make_vlm(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: object
) -> OllamaVisionLanguageModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaVisionLanguageModel(base_url=BASE_URL, model=MODEL, client=client, **kwargs)  # type: ignore[arg-type]


def tags_response(models: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": name} for name in models]})


def test_health_check_passes_when_model_is_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return tags_response([MODEL])

    make_vlm(handler).health_check()


def test_health_check_raises_ollama_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OllamaUnavailableError):
        make_vlm(handler).health_check()


def test_health_check_raises_model_unavailable_when_model_not_pulled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return tags_response(["llama3:8b"])

    with pytest.raises(ModelUnavailableError):
        make_vlm(handler).health_check()


def test_ask_sends_image_scene_json_history_and_question() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])

        body = json.loads(request.content)
        captured.update(body)
        return httpx.Response(200, json={"message": {"content": "há uma cadeira à sua frente"}})

    vlm = make_vlm(handler)
    response = vlm.ask(
        image=b"fake-jpeg-bytes",
        scene_json={"scene_id": "abc", "objects": []},
        system_prompt="Você é um assistente visual.",
        conversation_history=[ConversationMessage(role=MessageRole.USER, content="oi")],
        question="o que tem na minha frente?",
    )

    assert response.text == "há uma cadeira à sua frente"
    assert response.model == MODEL
    assert response.duration_ms >= 0

    assert captured["model"] == MODEL
    assert captured["stream"] is False
    options = cast(dict[str, object], captured["options"])
    assert cast(int, options["num_ctx"]) > 0
    messages = cast(list[dict[str, object]], captured["messages"])
    assert messages[0] == {"role": "system", "content": "Você é um assistente visual."}
    assert messages[1] == {"role": "user", "content": "oi"}
    last_message = messages[-1]
    assert "o que tem na minha frente?" in cast(str, last_message["content"])
    assert "abc" in cast(str, last_message["content"])
    assert last_message["images"]


def test_ask_raises_error_on_http_error_without_retrying() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        call_count += 1
        return httpx.Response(500, json={"error": "internal error"})

    vlm = make_vlm(handler, max_retries=2)
    with pytest.raises(VisionLanguageModelError):
        vlm.ask(
            image=b"x",
            scene_json={},
            system_prompt="s",
            conversation_history=[],
            question="q",
        )

    assert call_count == 1


def test_ask_retries_limited_times_on_connection_error_then_raises() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        call_count += 1
        raise httpx.ConnectError("connection refused", request=request)

    vlm = make_vlm(handler, max_retries=2)
    with pytest.raises(OllamaUnavailableError):
        vlm.ask(
            image=b"x",
            scene_json={},
            system_prompt="s",
            conversation_history=[],
            question="q",
        )

    assert call_count == 3


def test_ask_succeeds_after_one_transient_failure() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        call_count += 1
        if call_count == 1:
            raise httpx.TimeoutException("timed out", request=request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    vlm = make_vlm(handler, max_retries=2)
    response = vlm.ask(
        image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q"
    )

    assert response.text == "ok"
    assert call_count == 2


def test_ask_raises_empty_model_response_error_when_content_is_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(
            200,
            json={
                "message": {"content": "", "thinking": "raciocínio interno bem longo..."},
                "done": True,
                "done_reason": "length",
                "eval_count": 1218,
                "prompt_eval_count": 2878,
            },
        )

    vlm = make_vlm(handler)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_empty_model_response_error_when_content_is_only_whitespace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(200, json={"message": {"content": "   \n"}, "done_reason": "stop"})

    vlm = make_vlm(handler)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")
