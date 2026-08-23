from typing import Protocol

from app.domain.entities.face_encoding import FaceEncoding


class FaceEncoder(Protocol):
    def detect_and_encode(self, image: bytes) -> list[FaceEncoding]:
        """Detecta todos os rostos numa imagem e retorna bbox + embedding de cada um.

        Lista vazia significa nenhum rosto detectado — não é um erro.
        """
        ...
