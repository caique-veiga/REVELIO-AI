import json
from collections.abc import Callable
from typing import cast

import httpx
import pytest

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.message_role import MessageRole
from app.domain.entities.tool_call import ToolDefinition
from app.domain.protocols.vision_language_model import (
    EmptyModelResponseError,
    ModelUnavailableError,
    OllamaUnavailableError,
    VisionLanguageModelError,
)
from app.infrastructure.vlm.ollama_vision_language_model import OllamaVisionLanguageModel

BASE_URL = "http://100.118.123.0:11434"
MODEL = "qwen3.5:4b"

_A_TOOL = ToolDefinition(
    name="register_person",
    description="Cadastra uma pessoa",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)


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


def test_ask_sends_image_history_and_question_without_tools() -> None:
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
        system_prompt="Você é um assistente visual.",
        conversation_history=[ConversationMessage(role=MessageRole.USER, content="oi")],
        question="o que tem na minha frente?",
        tools=[],
    )

    assert response.text == "há uma cadeira à sua frente"
    assert response.tool_call is None
    assert response.model == MODEL
    assert response.duration_ms >= 0

    assert captured["model"] == MODEL
    assert captured["stream"] is False
    assert "tools" not in captured
    options = cast(dict[str, object], captured["options"])
    assert cast(int, options["num_ctx"]) > 0
    messages = cast(list[dict[str, object]], captured["messages"])
    assert messages[0] == {"role": "system", "content": "Você é um assistente visual."}
    assert messages[1] == {"role": "user", "content": "oi"}
    last_message = messages[-1]
    assert last_message["content"] == "o que tem na minha frente?"
    assert last_message["images"]


def test_ask_sends_tools_field_when_tools_given() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "olá"}})

    vlm = make_vlm(handler)
    vlm.ask(image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL])

    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "register_person",
                "description": "Cadastra uma pessoa",
                "parameters": _A_TOOL.parameters,
            },
        }
    ]


def test_ask_returns_tool_call_when_model_calls_function() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "register_person",
                                "arguments": {"name": "Maria", "relationship": "minha irmã"},
                            }
                        }
                    ],
                }
            },
        )

    vlm = make_vlm(handler)
    response = vlm.ask(
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "register_person"
    assert response.tool_call.arguments == {"name": "Maria", "relationship": "minha irmã"}
    assert response.text is None


def test_ask_parses_tool_call_arguments_given_as_json_string() -> None:
    """Alguns modelos retornam `arguments` como string JSON em vez de objeto
    (formato OpenAI-like) — o parsing deve aceitar ambos."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "identify_persons",
                                "arguments": "{}",
                            }
                        }
                    ],
                }
            },
        )

    vlm = make_vlm(handler)
    response = vlm.ask(
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "identify_persons"
    assert response.tool_call.arguments == {}


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
        vlm.ask(image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[])

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
        vlm.ask(image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[])

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
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[]
    )

    assert response.text == "ok"
    assert call_count == 2


def test_ask_raises_empty_model_response_error_when_content_and_tool_calls_are_empty() -> None:
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
        vlm.ask(image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[])


def test_ask_raises_empty_model_response_error_when_content_is_only_whitespace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(200, json={"message": {"content": "   \n"}, "done_reason": "stop"})

    vlm = make_vlm(handler)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask(image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[])


def test_ask_ignores_tool_call_missing_a_function_name() -> None:
    """Um tool call malformado (sem `name`) não deve ser tratado como
    válido — cai para o texto (se houver) ou para EmptyModelResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response([MODEL])
        return httpx.Response(
            200,
            json={"message": {"content": "", "tool_calls": [{"function": {"arguments": {}}}]}},
        )

    vlm = make_vlm(handler)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask(
            image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
        )
