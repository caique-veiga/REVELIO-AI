"""Cliente HTTP fino para a API do Revelio AI, usado só pelo benchmark de
avaliação. Fala com a aplicação real (sem mocks) através da API pública —
não importa nada de `backend/app`, então não altera nem depende dos
detalhes internos do pipeline principal.

Sem retries/sleeps para mascarar problemas de consistência: o 404
intermitente que existia aqui era sintoma de um bug real de transação em
`backend/app` (corrigido na ETAPA 13.1, ver `session.py`/`SceneService`/
`ConversationService`). Se ele reaparecer, este cliente deve deixar isso
visível, não escondido atrás de um retry.
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
        """Pergunta algo na conversation. Nunca levanta por erro HTTP/timeout —
        retorna `{"error": {"kind": ..., "status_code": ..., "detail": ...}}`
        para que o runner classifique o resultado (ver AnswerState) e siga
        para a próxima pergunta em vez de abortar o cenário inteiro.
        """
        try:
            response = self._client.post(
                f"/api/v1/conversations/{conversation_id}/messages", json={"content": content}
            )
        except httpx.TimeoutException as exc:
            return {"error": {"kind": "timeout", "status_code": None, "detail": str(exc)}}

        if response.status_code >= 400:
            return {
                "error": {
                    "kind": "http_error",
                    "status_code": response.status_code,
                    "detail": response.text[:500],
                }
            }

        result: dict[str, Any] = response.json()
        return result

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        response = self._client.get(f"/api/v1/conversations/{conversation_id}")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def close(self) -> None:
        self._client.close()
