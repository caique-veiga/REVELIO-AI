import hashlib
import io
import uuid
from datetime import date
from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError as PillowUnidentifiedImageError

from app.domain.entities.stored_image import StoredImage
from app.domain.protocols.image_storage import (
    ImageNotFoundError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
)

_EXTENSION_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

_PILLOW_FORMAT_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


def _inspect_image(content: bytes) -> tuple[str, int, int]:
    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()
    except (PillowUnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Arquivo de imagem corrompido ou ilegível.") from exc

    # verify() invalidates the image object for further use, so it must be reopened.
    with Image.open(io.BytesIO(content)) as image:
        pillow_format = image.format
        width, height = image.size

    mime_type = _PILLOW_FORMAT_MIME_TYPES.get(pillow_format or "")
    if mime_type is None:
        raise UnsupportedImageFormatError(f"Formato de imagem não suportado: {pillow_format}.")

    return mime_type, width, height


def _validate_image(filename: str, content: bytes, max_size_bytes: int) -> tuple[str, int, int]:
    extension = Path(filename).suffix.lower()
    expected_mime_type = _EXTENSION_MIME_TYPES.get(extension)
    if expected_mime_type is None:
        raise UnsupportedImageFormatError(f"Extensão não suportada: {extension or '(nenhuma)'}.")

    if not content:
        raise InvalidImageError("Arquivo de imagem vazio.")

    if len(content) > max_size_bytes:
        raise ImageTooLargeError(
            f"Imagem tem {len(content)} bytes; o máximo permitido é {max_size_bytes} bytes."
        )

    detected_mime_type, width, height = _inspect_image(content)

    if detected_mime_type != expected_mime_type:
        raise UnsupportedImageFormatError(
            f"A extensão '{extension}' não corresponde ao conteúdo real da imagem "
            f"({detected_mime_type})."
        )

    return detected_mime_type, width, height


class LocalImageStorage:
    def __init__(self, root_path: Path | str, max_size_bytes: int) -> None:
        self._root = Path(root_path)
        self._max_size_bytes = max_size_bytes

    def save(self, scene_id: uuid.UUID, filename: str, content: bytes) -> StoredImage:
        mime_type, width, height = _validate_image(filename, content, self._max_size_bytes)
        extension = Path(filename).suffix.lower()

        today = date.today()
        storage_key = f"{today:%Y}/{today:%m}/{today:%d}/{scene_id}{extension}"

        destination = self._resolve(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

        return StoredImage(
            storage_key=storage_key,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(content),
            width=width,
            height=height,
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def get(self, storage_key: str) -> bytes:
        path = self._resolve(storage_key)
        if not path.is_file():
            raise ImageNotFoundError(f"Imagem não encontrada: {storage_key}")
        return path.read_bytes()

    def delete(self, storage_key: str) -> None:
        self._resolve(storage_key).unlink(missing_ok=True)

    def exists(self, storage_key: str) -> bool:
        return self._resolve(storage_key).is_file()

    def _resolve(self, storage_key: str) -> Path:
        root = self._root.resolve()
        candidate = (root / storage_key).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Storage key inválida: {storage_key}")
        return candidate
