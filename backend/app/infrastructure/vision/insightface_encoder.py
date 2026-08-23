import io
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.face_encoding import FaceEncoding

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class InsightFaceEncoder:
    """Detecção e embedding de rosto via InsightFace (modelo `buffalo_l`, ONNX/CPU).

    O modelo só é carregado (e baixado, na primeira vez — ~280MB via GitHub)
    na primeira chamada real a `detect_and_encode`, não na construção — isso
    evita pagar esse custo em toda requisição HTTP ou em testes que nunca
    chegam a usar reconhecimento facial (ver CLAUDE_CONTEXT.md §17: não
    depender de rede/GPU nos testes unitários).
    """

    def __init__(self, model_name: str = "buffalo_l", detection_size: int = 640) -> None:
        self._model_name = model_name
        self._detection_size = detection_size
        self._app: FaceAnalysis | None = None

    def detect_and_encode(self, image: bytes) -> list[FaceEncoding]:
        app = self._get_app()

        with Image.open(io.BytesIO(image)) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        bgr = rgb[:, :, ::-1]  # InsightFace/OpenCV esperam BGR

        faces = app.get(bgr)
        return [
            FaceEncoding(
                bbox=BoundingBox(
                    x1=int(face.bbox[0]),
                    y1=int(face.bbox[1]),
                    x2=int(face.bbox[2]),
                    y2=int(face.bbox[3]),
                ),
                embedding=self._normalize(np.asarray(face.embedding, dtype=np.float64)),
            )
            for face in faces
        ]

    def _get_app(self) -> "FaceAnalysis":
        if self._app is None:
            from insightface.app import FaceAnalysis

            logger.info(
                "loading InsightFace model=%s detection_size=%d (download on first run)",
                self._model_name,
                self._detection_size,
            )
            app = FaceAnalysis(name=self._model_name, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(self._detection_size, self._detection_size))
            self._app = app
        return self._app

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return np.asarray(embedding / norm, dtype=np.float64)
