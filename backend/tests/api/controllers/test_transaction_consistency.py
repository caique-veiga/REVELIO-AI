"""Regressão para o 404 intermitente investigado na Etapa 13.1.

Causa raiz: `get_db_session` fazia commit no código pós-yield, que o
FastAPI só executa depois de a resposta HTTP já ter sido enviada — uma
chamada seguinte rápida o bastante podia chegar antes desse commit.

Ao contrário do `api_client` padrão (que reusa uma única Session em
memória para todas as requisições do teste, mascarando completamente essa
classe de bug), este teste cria uma Session NOVA por requisição — como a
produção faz de verdade — contra um SQLite em arquivo (não em memória),
para que a visibilidade dos dados dependa genuinamente do commit ter
ocorrido, não da identity map de uma sessão compartilhada.
"""

import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import (
    get_color_analyzer,
    get_image_storage,
    get_object_detector,
    get_vision_language_model,
)
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.detection import Detection
from app.domain.protocols.object_detector import ObjectDetector
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.infrastructure.database.base import Base
from app.infrastructure.database.models import Conversation, SceneModel
from app.infrastructure.database.session import get_db_session
from app.infrastructure.repositories.object_repository import SqlAlchemyObjectRepository
from app.infrastructure.storage.local_image_storage import LocalImageStorage
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer
from app.main import app


@dataclass
class ClientAndEngine:
    client: TestClient
    engine: Engine


@pytest.fixture
def per_request_session(tmp_path: Path) -> Generator[ClientAndEngine, None, None]:
    db_path = tmp_path / "transaction_consistency.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _fresh_session_per_request() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    fake_object_detector = MagicMock(spec=ObjectDetector)
    fake_object_detector.detect.return_value = [
        Detection(
            class_id=24,
            class_name="mochila",
            confidence=0.9,
            bbox=BoundingBox(x1=1, y1=1, x2=10, y2=10),
        )
    ]

    app.dependency_overrides[get_db_session] = _fresh_session_per_request
    app.dependency_overrides[get_object_detector] = lambda: fake_object_detector
    app.dependency_overrides[get_color_analyzer] = lambda: OpenCVColorAnalyzer()
    app.dependency_overrides[get_image_storage] = lambda: LocalImageStorage(
        root_path=tmp_path / "images", max_size_bytes=10_485_760
    )
    app.dependency_overrides[get_vision_language_model] = lambda: MagicMock(
        spec=VisionLanguageModel
    )

    try:
        yield ClientAndEngine(client=TestClient(app), engine=engine)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def _tiny_jpeg_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _row_counts(engine: Engine) -> tuple[int, int]:
    with Session(engine) as session:
        scene_count = session.execute(select(func.count()).select_from(SceneModel)).scalar_one()
        conversation_count = session.execute(
            select(func.count()).select_from(Conversation)
        ).scalar_one()
        return scene_count, conversation_count


class TestNoRetryNeededAfterCreate:
    def test_get_conversation_is_visible_immediately_after_scene_creation(
        self, per_request_session: ClientAndEngine
    ) -> None:
        image_bytes = _tiny_jpeg_bytes()
        client = per_request_session.client

        for _ in range(20):
            create_response = client.post(
                "/api/v1/scenes", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
            )
            assert create_response.status_code == 201
            conversation_id = create_response.json()["conversation_id"]

            get_response = client.get(f"/api/v1/conversations/{conversation_id}")

            assert get_response.status_code == 200, (
                "GET imediatamente após o 201 não deveria depender de retry/sleep "
                f"para enxergar a Conversation recém-criada (id={conversation_id})"
            )

    def test_ask_question_is_accepted_immediately_after_scene_creation(
        self, per_request_session: ClientAndEngine
    ) -> None:
        image_bytes = _tiny_jpeg_bytes()
        client = per_request_session.client

        for _ in range(20):
            create_response = client.post(
                "/api/v1/scenes", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
            )
            assert create_response.status_code == 201
            conversation_id = create_response.json()["conversation_id"]

            ask_response = client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json={"content": "pergunta imediata"},
            )

            assert ask_response.status_code != 404, (
                "POST /messages imediatamente após o 201 não deveria retornar 404 "
                f"para a Conversation recém-criada (id={conversation_id})"
            )


class TestRollbackOnPersistenceFailure:
    def test_partial_failure_leaves_no_row_in_scene_or_conversation_tables(
        self, per_request_session: ClientAndEngine
    ) -> None:
        image_bytes = _tiny_jpeg_bytes()
        client = per_request_session.client

        scenes_before, conversations_before = _row_counts(per_request_session.engine)
        assert (scenes_before, conversations_before) == (0, 0)

        with patch.object(
            SqlAlchemyObjectRepository, "add_many", side_effect=RuntimeError("falha simulada")
        ):
            response = client.post(
                "/api/v1/scenes", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
            )
        assert response.status_code == 500

        scenes_after, conversations_after = _row_counts(per_request_session.engine)
        assert (scenes_after, conversations_after) == (0, 0), (
            "Uma falha em DetectedObjects não deveria deixar Scene/Conversation "
            "parcialmente persistidas — a unidade de trabalho inteira deve ser "
            "revertida atomicamente."
        )


def test_rollback_removes_orphaned_image_from_storage(
    per_request_session: ClientAndEngine, tmp_path: Path
) -> None:
    image_bytes = _tiny_jpeg_bytes()
    images_dir = tmp_path / "images"

    with patch.object(
        SqlAlchemyObjectRepository, "add_many", side_effect=RuntimeError("falha simulada")
    ):
        response = per_request_session.client.post(
            "/api/v1/scenes", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
        )
    assert response.status_code == 500

    leftover_files = (
        [p for p in images_dir.rglob("*") if p.is_file()] if images_dir.exists() else []
    )
    assert leftover_files == [], f"Imagem órfã não removida após rollback: {leftover_files}"


def test_scene_id_from_failed_request_is_never_persisted(
    per_request_session: ClientAndEngine,
) -> None:
    image_bytes = _tiny_jpeg_bytes()
    fixed_id = uuid.UUID("11111111-1111-1111-1111-111111111111")

    with (
        patch.object(
            SqlAlchemyObjectRepository, "add_many", side_effect=RuntimeError("falha simulada")
        ),
        patch("uuid.uuid4", return_value=fixed_id),
    ):
        response = per_request_session.client.post(
            "/api/v1/scenes", files={"file": ("photo.jpg", image_bytes, "image/jpeg")}
        )
    assert response.status_code == 500

    get_response = per_request_session.client.get(f"/api/v1/conversations/{fixed_id}")
    assert get_response.status_code == 404
