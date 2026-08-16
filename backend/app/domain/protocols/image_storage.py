import uuid
from typing import Protocol

from app.domain.entities.stored_image import StoredImage


class ImageStorageError(Exception):
    """Base error for image storage operations."""


class UnsupportedImageFormatError(ImageStorageError):
    """Raised when the image extension or decoded content is not an allowed format."""


class ImageTooLargeError(ImageStorageError):
    """Raised when the image exceeds the configured maximum size."""


class InvalidImageError(ImageStorageError):
    """Raised when the image content cannot be decoded as a valid image."""


class ImageNotFoundError(ImageStorageError):
    """Raised when a storage key does not correspond to a stored image."""


class ImageStorage(Protocol):
    def save(self, scene_id: uuid.UUID, filename: str, content: bytes) -> StoredImage: ...

    def get(self, storage_key: str) -> bytes: ...

    def delete(self, storage_key: str) -> None: ...

    def exists(self, storage_key: str) -> bool: ...
