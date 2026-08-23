import io
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.infrastructure.vision.insightface_encoder import InsightFaceEncoder


@dataclass
class _FakeFace:
    bbox: np.ndarray
    embedding: np.ndarray


def _jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_detect_and_encode_maps_bbox_and_normalizes_embedding() -> None:
    encoder = InsightFaceEncoder()
    fake_app = MagicMock()
    fake_app.get.return_value = [
        _FakeFace(bbox=np.array([10.0, 20.0, 30.0, 40.0]), embedding=np.array([3.0, 4.0])),
    ]
    encoder._app = fake_app  # evita carregar/baixar o modelo real

    faces = encoder.detect_and_encode(_jpeg_bytes())

    assert len(faces) == 1
    face = faces[0]
    assert (face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2) == (10, 20, 30, 40)
    assert np.isclose(np.linalg.norm(face.embedding), 1.0)
    fake_app.get.assert_called_once()


def test_detect_and_encode_returns_empty_list_when_no_faces() -> None:
    encoder = InsightFaceEncoder()
    fake_app = MagicMock()
    fake_app.get.return_value = []
    encoder._app = fake_app

    faces = encoder.detect_and_encode(_jpeg_bytes())

    assert faces == []


def test_detect_and_encode_passes_bgr_image_to_app() -> None:
    encoder = InsightFaceEncoder()
    fake_app = MagicMock()
    fake_app.get.return_value = []
    encoder._app = fake_app

    encoder.detect_and_encode(_jpeg_bytes(width=4, height=2))

    passed_image = fake_app.get.call_args.args[0]
    assert passed_image.shape == (2, 4, 3)
