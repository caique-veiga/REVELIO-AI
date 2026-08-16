import uuid
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.detection import Detection
from app.infrastructure.database.models import Conversation, DetectedObject, SceneModel


def make_detection(**overrides: object) -> Detection:
    defaults: dict[str, object] = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": BoundingBox(x1=1, y1=1, x2=10, y2=10),
    }
    defaults.update(overrides)
    return Detection(**defaults)  # type: ignore[arg-type]


def test_create_scene_returns_201_with_expected_fields(
    api_client: TestClient, fake_object_detector: MagicMock, jpeg_bytes: bytes
) -> None:
    fake_object_detector.detect.return_value = [
        make_detection(class_name="person"),
        make_detection(class_name="chair", class_id=56),
    ]

    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    )

    assert response.status_code == 201
    payload = response.json()
    assert uuid.UUID(payload["scene_id"])
    assert uuid.UUID(payload["conversation_id"])
    assert payload["object_count"] == 2
    assert payload["status"] == "created"


def test_create_scene_with_zero_detections_returns_object_count_zero(
    api_client: TestClient, fake_object_detector: MagicMock, jpeg_bytes: bytes
) -> None:
    fake_object_detector.detect.return_value = []

    response = api_client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    )

    assert response.status_code == 201
    assert response.json()["object_count"] == 0


def test_create_scene_persists_scene_conversation_and_objects(
    api_client: TestClient,
    api_db_session: Session,
    fake_object_detector: MagicMock,
    jpeg_bytes: bytes,
) -> None:
    fake_object_detector.detect.return_value = [make_detection()]

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

    detected_objects = list(
        api_db_session.scalars(select(DetectedObject).where(DetectedObject.scene_id == scene_id))
    )
    assert len(detected_objects) == 1
    assert detected_objects[0].class_name == "person"
    assert detected_objects[0].color_name is not None
    assert detected_objects[0].position_region is not None


def test_each_request_creates_a_new_scene_and_conversation(
    api_client: TestClient, fake_object_detector: MagicMock, jpeg_bytes: bytes, png_bytes: bytes
) -> None:
    fake_object_detector.detect.return_value = []

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
