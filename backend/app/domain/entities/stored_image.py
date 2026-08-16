from dataclasses import dataclass


@dataclass(frozen=True)
class StoredImage:
    storage_key: str
    filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    sha256: str
