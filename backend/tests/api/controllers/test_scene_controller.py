import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.infrastructure.database.models import Conversation, SceneModel


def test_create_scene_returns_201_with_expected_fields(
    api_client: TestClient, jpeg_bytes: bytes
) -> None:
    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    )

    assert response.status_code == 201
    payload = response.json()
    assert uuid.UUID(payload["scene_id"])
    assert uuid.UUID(payload["conversation_id"])
    assert payload["status"] == "created"


def test_create_scene_persists_scene_and_conversation(
    api_client: TestClient, api_db_session: Session, jpeg_bytes: bytes
) -> None:
    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    )
    payload = response.json()

    scene_id = uuid.UUID(payload["scene_id"])
    conversation_id = uuid.UUID(payload["conversation_id"])

    scene_row = api_db_session.get(SceneModel, scene_id)
    assert scene_row is not None
    assert scene_row.image_mime_type == "image/jpeg"

    conversation_row = api_db_session.get(Conversation, conversation_id)
    assert conversation_row is not None
    assert conversation_row.scene_id == scene_id


def test_each_request_creates_a_new_scene_and_conversation(
    api_client: TestClient, jpeg_bytes: bytes, png_bytes: bytes
) -> None:
    first = api_client.post(
        "/api/v1/scenes", files={"file": ("photo1.jpg", jpeg_bytes, "image/jpeg")}
    ).json()
    second = api_client.post(
        "/api/v1/scenes", files={"file": ("photo2.png", png_bytes, "image/png")}
    ).json()

    assert first["scene_id"] != second["scene_id"]
    assert first["conversation_id"] != second["conversation_id"]


def test_create_scene_rejects_unsupported_file_extension(
    api_client: TestClient, jpeg_bytes: bytes
) -> None:
    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.gif", jpeg_bytes, "image/gif")}
    )

    assert response.status_code == 400


def test_create_scene_rejects_corrupted_image_content(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/scenes",
        files={"file": ("photo.jpg", b"not a real image", "image/jpeg")},
    )

    assert response.status_code == 400


def test_create_scene_rejects_extension_mismatched_content(
    api_client: TestClient, png_bytes: bytes
) -> None:
    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", png_bytes, "image/jpeg")}
    )

    assert response.status_code == 400
