"""Cliente HTTP fino para a API do Revelio AI, usado só pelo benchmark de
avaliação. Fala com a aplicação real (sem mocks) através da API pública —
não importa nada de `backend/app`, então não altera nem depende dos
detalhes internos do pipeline principal.
"""

from pathlib import Path
from typing import Any

import httpx

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class RevelioClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def create_scene(self, image_path: Path) -> dict[str, Any]:
        mime_type = _MIME_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
        with image_path.open("rb") as file:
            response = self._client.post(
                "/api/v1/scenes", files={"file": (image_path.name, file, mime_type)}
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def ask(self, conversation_id: str, content: str) -> dict[str, Any]:
        response = self._client.post(
            f"/api/v1/conversations/{conversation_id}/messages", json={"content": content}
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        response = self._client.get(f"/api/v1/conversations/{conversation_id}")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def close(self) -> None:
        self._client.close()
