"""Testes do fluxo de tool calling (Gemini) ponta a ponta via API.

Usa um fake VLM que implementa tanto `VisionLanguageModel` quanto
`ToolCallingVisionLanguageModel` (protocolo `@runtime_checkable`) — ao
contrário do `fake_vision_language_model` compartilhado (que usa
`MagicMock(spec=VisionLanguageModel)` e deliberadamente NÃO tem
`ask_with_tools`), para exercitar o caminho de tool calling de verdade.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import (
    get_color_analyzer,
    get_face_encoder,
    get_image_storage,
    get_object_detector,
    get_skip_yolo_pipeline,
    get_vision_language_model,
)
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.conversation_message import ConversationMessage
from app.domain.entities.face_encoding import FaceEncoding
from app.domain.entities.tool_call import ToolCall, ToolCallResponse, ToolDefinition
from app.domain.protocols.face_encoder import FaceEncoder
from app.domain.protocols.object_detector import ObjectDetector
from app.infrastructure.database.base import Base
from app.infrastructure.database.session import get_db_session
from app.infrastructure.storage.local_image_storage import LocalImageStorage
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer
from app.main import app


class FakeToolCallingVlm:
    """Implementa VisionLanguageModel + ToolCallingVisionLanguageModel de verdade."""

    def __init__(self) -> None:
        self.next_response: ToolCallResponse | None = None

    def health_check(self) -> None:
        return None

    def ask(self, **_: object) -> None:
        raise AssertionError("ask() não deveria ser chamado no fluxo de tool calling")

    def ask_with_tools(
        self,
        *,
        image: bytes,
        system_prompt: str,
        conversation_history: list[ConversationMessage],
        question: str,
        tools: list[ToolDefinition],
    ) -> ToolCallResponse:
        assert self.next_response is not None
        return self.next_response


ToolClientFixture = tuple[TestClient, FakeToolCallingVlm, MagicMock]


@pytest.fixture
def tool_client(tmp_path: Path) -> Generator[ToolClientFixture, None, None]:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_factory()

    def _override_db_session() -> Generator[Session, None, None]:
        try:
            yield session
        except Exception:
            session.rollback()
            raise

    fake_vlm = FakeToolCallingVlm()
    fake_face_encoder = MagicMock(spec=FaceEncoder)

    app.dependency_overrides[get_db_session] = _override_db_session
    # SceneService ainda declara essa dependência, mas com
    # get_skip_yolo_pipeline=True ela nunca é chamada — não precisa de
    # configuração especial.
    app.dependency_overrides[get_object_detector] = lambda: MagicMock(spec=ObjectDetector)
    app.dependency_overrides[get_color_analyzer] = lambda: OpenCVColorAnalyzer()
    app.dependency_overrides[get_image_storage] = lambda: LocalImageStorage(
        root_path=tmp_path, max_size_bytes=10_485_760
    )
    app.dependency_overrides[get_vision_language_model] = lambda: fake_vlm
    app.dependency_overrides[get_skip_yolo_pipeline] = lambda: True
    app.dependency_overrides[get_face_encoder] = lambda: fake_face_encoder

    try:
        yield TestClient(app), fake_vlm, fake_face_encoder
    finally:
        app.dependency_overrides.clear()
        session.close()
        engine.dispose()


def _create_scene(client: TestClient, jpeg_bytes: bytes) -> dict[str, object]:
    response = client.post(
        "/api/v1/scenes", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 201
    result: dict[str, object] = response.json()
    return result


def test_scene_creation_skips_yolo_when_gemini_active(
    tool_client: ToolClientFixture, jpeg_bytes: bytes
) -> None:
    client, _fake_vlm, _face_encoder = tool_client
    scene = _create_scene(client, jpeg_bytes)
    assert scene["object_count"] == 0


def test_direct_question_without_tool_call_returns_gemini_text(
    tool_client: ToolClientFixture, jpeg_bytes: bytes
) -> None:
    client, fake_vlm, _face_encoder = tool_client
    scene = _create_scene(client, jpeg_bytes)
    fake_vlm.next_response = ToolCallResponse(
        text="Vejo uma cadeira à sua frente.",
        tool_call=None,
        model="gemini-3.5-flash-lite",
        duration_ms=100.0,
    )

    response = client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "O que você vê?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Vejo uma cadeira à sua frente."


def test_register_person_tool_call_persists_person(
    tool_client: ToolClientFixture, jpeg_bytes: bytes
) -> None:
    client, fake_vlm, face_encoder = tool_client
    scene = _create_scene(client, jpeg_bytes)

    face_encoder.detect_and_encode.return_value = [
        FaceEncoding(bbox=BoundingBox(x1=1, y1=1, x2=50, y2=50), embedding=np.array([1.0, 0.0]))
    ]
    fake_vlm.next_response = ToolCallResponse(
        text=None,
        tool_call=ToolCall(
            name="register_person", arguments={"name": "Maria", "relationship": "minha irmã"}
        ),
        model="gemini-3.5-flash-lite",
        duration_ms=120.0,
    )

    response = client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Cadastra minha irmã Maria"},
    )

    assert response.status_code == 200
    assert "Maria" in response.json()["answer"]

    conversation = client.get(f"/api/v1/conversations/{scene['conversation_id']}").json()
    assistant_message = conversation["messages"][-1]
    assert assistant_message["model_name"] == "gemini-3.5-flash-lite"


def test_identify_persons_tool_call_reports_unknown_when_nobody_registered(
    tool_client: ToolClientFixture, jpeg_bytes: bytes
) -> None:
    client, fake_vlm, face_encoder = tool_client
    scene = _create_scene(client, jpeg_bytes)

    face_encoder.detect_and_encode.return_value = [
        FaceEncoding(bbox=BoundingBox(x1=1, y1=1, x2=50, y2=50), embedding=np.array([1.0, 0.0]))
    ]
    fake_vlm.next_response = ToolCallResponse(
        text=None,
        tool_call=ToolCall(name="identify_persons", arguments={}),
        model="gemini-3.5-flash-lite",
        duration_ms=90.0,
    )

    response = client.post(
        f"/api/v1/conversations/{scene['conversation_id']}/messages",
        json={"content": "Quem está aqui?"},
    )

    assert response.status_code == 200
    assert "desconhecida" in response.json()["answer"]
