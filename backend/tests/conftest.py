import io
import uuid
from collections.abc import Callable, Generator

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.base import Base
from app.infrastructure.database.models import SceneModel, User


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
def scene_factory() -> Callable[..., SceneModel]:
    def _make(**overrides: object) -> SceneModel:
        defaults: dict[str, object] = {
            "image_storage_key": f"data/images/{uuid.uuid4()}.jpg",
            "image_filename": "photo.jpg",
            "image_mime_type": "image/jpeg",
            "image_width": 1920,
            "image_height": 1080,
            "image_size_bytes": 204800,
        }
        defaults.update(overrides)
        return SceneModel(**defaults)

    return _make
