import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.detection import Detection
from app.domain.entities.vlm_response import VLMResponse


def make_detection(**overrides: object) -> Detection:
    defaults: dict[str, object] = {
        "class_id": 24,
        "class_name": "mochila",
        "confidence": 0.9,
        "bbox": BoundingBox(x1=1, y1=1, x2=10, y2=10),
    }
    defaults.update(overrides)
    return Detection(**defaults)  # type: ignore[arg-type]


def create_scene(
    api_client: TestClient,
    fake_object_detector: MagicMock,
    image_bytes: bytes,
    detections: list[Detection] | None = None,
    filename: str = "photo.jpg",
) -> dict[str, object]:
    fake_object_detector.detect.return_value = (
        detections if detections is not None else [make_detection()]
    )
    response = api_client.post(
        "/api/v1/scenes", files={"file": (filename, image_bytes, "image/jpeg")}
    )
    assert response.status_code == 201
    result: dict[str, object] = response.json()
    return result


def test_ask_question_returns_answer_scene_id_and_referenced_objects(
    api_client: TestClient,
    fake_object_detector: MagicMock,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(api_client, fake_object_detector, jpeg_bytes)
    fake_vision_language_model.ask.return_value = VLMResponse(
        text="A mochila é azul.", model="qwen3.5:4b", duration_ms=120.0
    )

    response = api_client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Qual a cor da mochila?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "A mochila é azul."
    assert payload["scene_id"] == scene["scene_id"]
    assert len(payload["referenced_objects"]) == 1
    assert payload["referenced_objects"][0]["class_name"] == "mochila"


def test_multiple_messages_persist_in_order(
    api_client: TestClient,
    fake_object_detector: MagicMock,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(api_client, fake_object_detector, jpeg_bytes)
    conversation_id = scene["conversation_id"]

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="resposta 1", model="qwen3.5:4b", duration_ms=100.0
    )
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "pergunta 1"}
    )

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="resposta 2", model="qwen3.5:4b", duration_ms=100.0
    )
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
    api_client: TestClient, fake_object_detector: MagicMock, jpeg_bytes: bytes
) -> None:
    scene = create_scene(api_client, fake_object_detector, jpeg_bytes)

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
    fake_object_detector: MagicMock,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
    png_bytes: bytes,
) -> None:
    scene_a = create_scene(api_client, fake_object_detector, jpeg_bytes, filename="a.jpg")
    fake_vision_language_model.ask.return_value = VLMResponse(
        text="resposta A", model="qwen3.5:4b", duration_ms=50.0
    )
    api_client.post(
        f"/api/v1/conversations/{scene_a['conversation_id']}/messages",
        json={"content": "pergunta A"},
    )

    scene_b = create_scene(api_client, fake_object_detector, png_bytes, filename="b.png")
    assert scene_a["conversation_id"] != scene_b["conversation_id"]

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="resposta B", model="qwen3.5:4b", duration_ms=50.0
    )
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
    api_client: TestClient,
    fake_object_detector: MagicMock,
    fake_vision_language_model: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    scene = create_scene(
        api_client, fake_object_detector, jpeg_bytes, detections=[make_detection()]
    )
    conversation_id = scene["conversation_id"]

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="Vejo uma mochila.", model="qwen3.5:4b", duration_ms=50.0
    )
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "O que estou vendo?"}
    )

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="A mochila é azul.", model="qwen3.5:4b", duration_ms=50.0
    )
    api_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Qual a cor da mochila?"},
    )

    fake_vision_language_model.ask.return_value = VLMResponse(
        text="Ela está à esquerda.", model="qwen3.5:4b", duration_ms=50.0
    )
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
