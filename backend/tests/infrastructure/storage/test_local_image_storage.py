import hashlib
import uuid
from datetime import date
from pathlib import Path

import pytest

from app.domain.protocols.image_storage import (
    ImageNotFoundError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
)
from app.infrastructure.storage.local_image_storage import LocalImageStorage

DEFAULT_MAX_SIZE = 10_485_760


def make_storage(tmp_path: Path, max_size_bytes: int = DEFAULT_MAX_SIZE) -> LocalImageStorage:
    return LocalImageStorage(root_path=tmp_path, max_size_bytes=max_size_bytes)


def test_save_organizes_file_by_date(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)
    scene_id = uuid.uuid4()
    today = date.today()

    stored = storage.save(scene_id, "photo.jpg", jpeg_bytes)

    expected_key = f"{today:%Y}/{today:%m}/{today:%d}/{scene_id}.jpg"
    assert stored.storage_key == expected_key
    assert (tmp_path / expected_key).is_file()


def test_save_returns_correct_metadata(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)

    stored = storage.save(uuid.uuid4(), "photo.jpg", jpeg_bytes)

    assert stored.filename == "photo.jpg"
    assert stored.mime_type == "image/jpeg"
    assert stored.width == 64
    assert stored.height == 48
    assert stored.size_bytes == len(jpeg_bytes)
    assert stored.sha256 == hashlib.sha256(jpeg_bytes).hexdigest()


def test_save_persists_the_exact_bytes(tmp_path: Path, png_bytes: bytes) -> None:
    storage = make_storage(tmp_path)

    stored = storage.save(uuid.uuid4(), "photo.png", png_bytes)

    assert (tmp_path / stored.storage_key).read_bytes() == png_bytes


def test_get_returns_previously_saved_bytes(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)
    stored = storage.save(uuid.uuid4(), "photo.jpg", jpeg_bytes)

    assert storage.get(stored.storage_key) == jpeg_bytes


def test_get_raises_when_storage_key_does_not_exist(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(ImageNotFoundError):
        storage.get("2026/08/16/does-not-exist.jpg")


def test_exists_reflects_whether_the_image_was_saved(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)
    stored = storage.save(uuid.uuid4(), "photo.jpg", jpeg_bytes)

    assert storage.exists(stored.storage_key) is True
    assert storage.exists("2026/08/16/does-not-exist.jpg") is False


def test_delete_removes_the_file(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)
    stored = storage.save(uuid.uuid4(), "photo.jpg", jpeg_bytes)

    storage.delete(stored.storage_key)

    assert storage.exists(stored.storage_key) is False
    with pytest.raises(ImageNotFoundError):
        storage.get(stored.storage_key)


def test_delete_is_idempotent_when_file_is_already_gone(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    storage.delete("2026/08/16/does-not-exist.jpg")


def test_save_rejects_unsupported_extension(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(UnsupportedImageFormatError):
        storage.save(uuid.uuid4(), "photo.gif", jpeg_bytes)


def test_save_rejects_image_larger_than_configured_limit(tmp_path: Path, jpeg_bytes: bytes) -> None:
    storage = make_storage(tmp_path, max_size_bytes=10)

    with pytest.raises(ImageTooLargeError):
        storage.save(uuid.uuid4(), "photo.jpg", jpeg_bytes)


def test_save_rejects_corrupted_image_content(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(InvalidImageError):
        storage.save(uuid.uuid4(), "photo.jpg", b"not a real image")


def test_save_rejects_extension_that_does_not_match_actual_content(
    tmp_path: Path, png_bytes: bytes
) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(UnsupportedImageFormatError):
        storage.save(uuid.uuid4(), "photo.jpg", png_bytes)


def test_get_rejects_path_traversal_outside_root(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(ValueError, match="Storage key inválida"):
        storage.get("../../etc/passwd")
