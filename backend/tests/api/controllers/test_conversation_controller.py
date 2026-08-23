import uuid
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.face_encoding import FaceEncoding
from app.domain.entities.message_role import MessageRole
from app.domain.entities.tool_call import ToolCall, ToolCallResponse
from app.infrastructure.database.models import Message


def create_scene(
    api_client: TestClient, jpeg_bytes: bytes, filename: str = "photo.jpg"
) -> dict[str, object]:
    response = api_client.post(
        "/api/v1/scenes", files={"file": (filename, jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 201
    result: dict[str, object] = response.json()
    return result


def _text_response(
    text: str, model: str = "qwen3.5:4b", duration_ms: float = 100.0
) -> ToolCallResponse:
    return ToolCallResponse(text=text, tool_call=None, model=model, duration_ms=duration_ms)


def test_ask_question_returns_answer_and_scene_id(
    api_client: TestClient, fake_vision_language_model: MagicMock, jpeg_bytes: bytes
) -> None:
    scene = create_scene(api_client, jpeg_bytes)
    fake_vision_language_model.ask.return_value = _text_response("A mochila é azul.")

    response = api_client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Qual a cor da mochila?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "A mochila é azul."
    assert payload["scene_id"] == scene["scene_id"]


def test_multiple_messages_persist_in_order(
    api_client: TestClient, fake_vision_language_model: MagicMock, jpeg_bytes: bytes
) -> None:
    scene = create_scene(api_client, jpeg_bytes)
    conversation_id = scene["conversation_id"]

    fake_vision_language_model.ask.return_value = _text_response("resposta 1")
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "pergunta 1"}
    )

    fake_vision_language_model.ask.return_value = _text_response("resposta 2")
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "pergunta 2"}
    )

    response = api_client.get(f"/api/v1/conversations/{conversation_id}")

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages] == [
        "pergunta 1",
        "resposta 1",
        "pergunta 2",
        "resposta 2",
    ]


def test_get_conversation_returns_scene_id_and_empty_messages_initially(
    api_client: TestClient, jpeg_bytes: bytes
) -> None:
    scene = create_scene(api_client, jpeg_bytes)

    response = api_client.get(f"/api/v1/conversations/{scene['conversation_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == scene["conversation_id"]
    assert payload["scene_id"] == scene["scene_id"]
    assert payload["messages"] == []


def test_ask_question_on_nonexistent_conversation_returns_404(api_client: TestClient) -> None:
    response = api_client.post(
        f"/api/v1/conversations/{uuid.uuid4()}/messages", json={"content": "oi"}
    )
    assert response.status_code == 404


def test_get_nonexistent_conversation_returns_404(api_client: TestClient) -> None:
    response = api_client.get(f"/api/v1/conversations/{uuid.uuid4()}")
    assert response.status_code == 404


def test_conversation_history_is_isolated_between_scenes(
    api_client: TestClient,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
    png_bytes: bytes,
) -> None:
    scene_a = create_scene(api_client, jpeg_bytes, filename="a.jpg")
    fake_vision_language_model.ask.return_value = _text_response("resposta A")
    api_client.post(
        f"/api/v1/conversations/{scene_a['conversation_id']}/messages",
        json={"content": "pergunta A"},
    )

    scene_b = create_scene(api_client, png_bytes, filename="b.png")
    assert scene_a["conversation_id"] != scene_b["conversation_id"]

    fake_vision_language_model.ask.return_value = _text_response("resposta B")
    api_client.post(
        f"/api/v1/conversations/{scene_b['conversation_id']}/messages",
        json={"content": "pergunta B"},
    )

    last_call_kwargs = fake_vision_language_model.ask.call_args_list[-1].kwargs
    assert last_call_kwargs["conversation_history"] == []

    conversation_a = api_client.get(f"/api/v1/conversations/{scene_a['conversation_id']}").json()
    conversation_b = api_client.get(f"/api/v1/conversations/{scene_b['conversation_id']}").json()
    assert [m["content"] for m in conversation_a["messages"]] == ["pergunta A", "resposta A"]
    assert [m["content"] for m in conversation_b["messages"]] == ["pergunta B", "resposta B"]


def test_follow_up_question_has_sufficient_history_context(
    api_client: TestClient, fake_vision_language_model: MagicMock, jpeg_bytes: bytes
) -> None:
    scene = create_scene(api_client, jpeg_bytes)
    conversation_id = scene["conversation_id"]

    fake_vision_language_model.ask.return_value = _text_response("Vejo uma mochila.")
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "O que estou vendo?"}
    )

    fake_vision_language_model.ask.return_value = _text_response("A mochila é azul.")
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Qual a cor da mochila?"},
    )

    fake_vision_language_model.ask.return_value = _text_response("Ela está à esquerda.")
    response = api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "E onde ela está?"}
    )

    assert response.status_code == 200

    third_call_kwargs = fake_vision_language_model.ask.call_args_list[-1].kwargs
    history = third_call_kwargs["conversation_history"]
    assert [message.content for message in history] == [
        "O que estou vendo?",
        "Vejo uma mochila.",
        "Qual a cor da mochila?",
        "A mochila é azul.",
    ]
    assert third_call_kwargs["question"] == "E onde ela está?"


def test_prompt_version_is_tool_calling_v1_for_a_direct_answer(
    api_client: TestClient,
    api_db_session: Session,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(api_client, jpeg_bytes)
    fake_vision_language_model.ask.return_value = _text_response("A mochila está à esquerda.")

    api_client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Onde está a mochila?"},
    )

    assistant_message = api_db_session.query(Message).filter_by(role=MessageRole.ASSISTANT).one()
    assert assistant_message.prompt_version == "tool_calling_v1"


def test_register_person_tool_call_persists_person(
    api_client: TestClient,
    fake_vision_language_model: MagicMock,
    fake_face_encoder: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(api_client, jpeg_bytes)

    fake_face_encoder.detect_and_encode.return_value = [
        FaceEncoding(bbox=BoundingBox(x1=1, y1=1, x2=50, y2=50), embedding=np.array([1.0, 0.0]))
    ]
    fake_vision_language_model.ask.return_value = ToolCallResponse(
        text=None,
        tool_call=ToolCall(
            name="register_person", arguments={"name": "Maria", "relationship": "minha irmã"}
        ),
        model="gemini-3.5-flash-lite",
        duration_ms=120.0,
    )

    response = api_client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Cadastra minha irmã Maria"},
    )

    assert response.status_code == 200
    assert "Maria" in response.json()["answer"]

    conversation = api_client.get(f"/api/v1/conversations/{scene['conversation_id']}").json()
    assistant_message = conversation["messages"][-1]
    assert assistant_message["model_name"] == "gemini-3.5-flash-lite"


def test_identify_persons_tool_call_reports_unknown_when_nobody_registered(
    api_client: TestClient,
    fake_vision_language_model: MagicMock,
    fake_face_encoder: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(api_client, jpeg_bytes)

    fake_face_encoder.detect_and_encode.return_value = [
        FaceEncoding(bbox=BoundingBox(x1=1, y1=1, x2=50, y2=50), embedding=np.array([1.0, 0.0]))
    ]
    fake_vision_language_model.ask.return_value = ToolCallResponse(
        text=None,
        tool_call=ToolCall(name="identify_persons", arguments={}),
        model="gemini-3.5-flash-lite",
        duration_ms=90.0,
    )

    response = api_client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Quem está aqui?"},
    )

    assert response.status_code == 200
    assert "desconhecida" in response.json()["answer"]
