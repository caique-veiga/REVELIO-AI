import json
from collections.abc import Callable

import httpx
import pytest

from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.message_role import MessageRole
from app.domain.entities.tool_call import ToolDefinition
from app.domain.protocols.vision_language_model import (
    EmptyModelResponseError,
    VisionLanguageModelError,
    VisionProviderTimeoutError,
    VisionProviderUnavailableError,
)
from app.infrastructure.vlm.gemini_vision_language_model import GeminiVisionLanguageModel

_A_TOOL = ToolDefinition(
    name="register_person",
    description="Cadastra uma pessoa",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)

BASE_URL = "https://generativelanguage.googleapis.com"
MODEL = "gemini-2.5-flash-lite"
API_KEY = "test-api-key"


def make_vlm(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: object
) -> GeminiVisionLanguageModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiVisionLanguageModel(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        client=client,
        **kwargs,  # type: ignore[arg-type]
    )


def _generate_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"},
            ],
            "usageMetadata": {"promptTokenCount": 640, "candidatesTokenCount": 92},
        },
    )


def test_health_check_passes_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1beta/models/{MODEL}"
        return httpx.Response(200, json={"name": MODEL})

    make_vlm(handler).health_check()


def test_health_check_raises_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    with pytest.raises(VisionProviderUnavailableError):
        make_vlm(handler).health_check()


def test_health_check_raises_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(VisionProviderUnavailableError):
        make_vlm(handler).health_check()


def test_ask_sends_system_instruction_history_image_and_question() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url.params["key"] == API_KEY
        return _generate_response("há uma cadeira à sua frente")

    vlm = make_vlm(handler, image_enable_optimization=False)
    response = vlm.ask(
        image=b"fake-jpeg-bytes",
        scene_json={"scene_id": "abc", "objects": []},
        system_prompt="Você é um assistente visual.",
        conversation_history=[
            ConversationMessage(role=MessageRole.USER, content="oi"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="olá"),
        ],
        question="o que tem na minha frente?",
    )

    assert response.text == "há uma cadeira à sua frente"
    assert response.model == MODEL
    assert response.duration_ms >= 0

    assert captured["system_instruction"] == {"parts": [{"text": "Você é um assistente visual."}]}
    contents = captured["contents"]
    assert isinstance(contents, list)
    assert contents[0] == {"role": "user", "parts": [{"text": "oi"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "olá"}]}
    last_turn = contents[-1]
    assert last_turn["role"] == "user"
    text_part, image_part = last_turn["parts"]
    assert "o que tem na minha frente?" in text_part["text"]
    assert "abc" in text_part["text"]
    assert image_part["inline_data"]["mime_type"] == "image/jpeg"


def test_ask_optimizes_image_by_default() -> None:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2000, 1000), color=(1, 2, 3)).save(buffer, format="PNG")
    large_image = buffer.getvalue()

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _generate_response("ok")

    vlm = make_vlm(handler, image_max_dimension=768, image_jpeg_quality=85)
    vlm.ask(
        image=large_image,
        scene_json={},
        system_prompt="s",
        conversation_history=[],
        question="q",
    )

    contents = captured["contents"]
    assert isinstance(contents, list)
    image_part = contents[-1]["parts"][1]
    sent_bytes = image_part["inline_data"]["data"]
    import base64

    decoded = base64.b64decode(sent_bytes)
    assert len(decoded) < len(large_image)


def test_ask_raises_empty_model_response_error_when_no_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_visionprovider_unavailable_on_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionProviderUnavailableError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_visionprovider_unavailable_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionProviderUnavailableError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_visionprovider_unavailable_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionProviderUnavailableError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_visionprovider_timeout_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionProviderTimeoutError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_visionprovider_unavailable_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionProviderUnavailableError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_raises_generic_error_on_unexpected_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, json={"error": "teapot"})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(VisionLanguageModelError):
        vlm.ask(image=b"x", scene_json={}, system_prompt="s", conversation_history=[], question="q")


def test_ask_with_tools_sends_function_declarations() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return _generate_response("olá")

    vlm = make_vlm(handler, image_enable_optimization=False)
    vlm.ask_with_tools(
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
    )

    tools_payload = captured["tools"]
    assert tools_payload == [
        {
            "function_declarations": [
                {
                    "name": "register_person",
                    "description": "Cadastra uma pessoa",
                    "parameters": _A_TOOL.parameters,
                }
            ]
        }
    ]


def test_ask_with_tools_returns_direct_text_when_no_function_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _generate_response("vejo uma cadeira")

    vlm = make_vlm(handler, image_enable_optimization=False)
    response = vlm.ask_with_tools(
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
    )

    assert response.text == "vejo uma cadeira"
    assert response.tool_call is None
    assert response.model == MODEL


def test_ask_with_tools_returns_tool_call_when_model_calls_function() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "register_person",
                                        "args": {"name": "Maria", "relationship": "minha irmã"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    vlm = make_vlm(handler, image_enable_optimization=False)
    response = vlm.ask_with_tools(
        image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "register_person"
    assert response.tool_call.arguments == {"name": "Maria", "relationship": "minha irmã"}


def test_ask_with_tools_raises_empty_model_response_error_when_nothing_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    vlm = make_vlm(handler, image_enable_optimization=False)
    with pytest.raises(EmptyModelResponseError):
        vlm.ask_with_tools(
            image=b"x", system_prompt="s", conversation_history=[], question="q", tools=[_A_TOOL]
        )
