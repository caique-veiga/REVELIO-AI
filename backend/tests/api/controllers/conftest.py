from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_color_analyzer, get_image_storage, get_object_detector
from app.domain.protocols.object_detector import ObjectDetector
from app.infrastructure.database.base import Base
from app.infrastructure.database.session import get_db_session
from app.infrastructure.storage.local_image_storage import LocalImageStorage
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer
from app.main import app


@pytest.fixture
def api_db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def fake_object_detector() -> MagicMock:
    return MagicMock(spec=ObjectDetector)


@pytest.fixture
def api_client(
    api_db_session: Session, fake_object_detector: ObjectDetector, tmp_path: Path
) -> Generator[TestClient, None, None]:
    def _override_db_session() -> Generator[Session, None, None]:
        try:
            yield api_db_session
            api_db_session.commit()
        except Exception:
            api_db_session.rollback()
            raise

    app.dependency_overrides[get_db_session] = _override_db_session
    app.dependency_overrides[get_object_detector] = lambda: fake_object_detector
    app.dependency_overrides[get_color_analyzer] = lambda: OpenCVColorAnalyzer()
    app.dependency_overrides[get_image_storage] = lambda: LocalImageStorage(
        root_path=tmp_path, max_size_bytes=10_485_760
    )

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
