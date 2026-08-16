import io
import uuid
from collections.abc import Callable, Generator

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.base import Base
from app.infrastructure.database.models import DetectedObject, Scene, User


@pytest.fixture
def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
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
def user(db_session: Session) -> User:
    new_user = User()
    db_session.add(new_user)
    db_session.flush()
    return new_user


@pytest.fixture
def scene_factory() -> Callable[..., Scene]:
    def _make(**overrides: object) -> Scene:
        defaults: dict[str, object] = {
            "image_storage_key": f"data/images/{uuid.uuid4()}.jpg",
            "image_filename": "photo.jpg",
            "image_mime_type": "image/jpeg",
            "image_width": 1920,
            "image_height": 1080,
            "image_size_bytes": 204800,
        }
        defaults.update(overrides)
        return Scene(**defaults)

    return _make


@pytest.fixture
def detected_object_factory() -> Callable[..., DetectedObject]:
    def _make(scene_id: uuid.UUID, **overrides: object) -> DetectedObject:
        defaults: dict[str, object] = {
            "scene_id": scene_id,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.98,
            "bbox_x1": 100,
            "bbox_y1": 200,
            "bbox_x2": 500,
            "bbox_y2": 900,
            "position_horizontal": "center",
            "position_vertical": "middle",
            "position_region": "front-center",
            "color_name": "blue",
            "color_r": 20,
            "color_g": 80,
            "color_b": 180,
            "color_confidence": 0.82,
        }
        defaults.update(overrides)
        return DetectedObject(**defaults)

    return _make
