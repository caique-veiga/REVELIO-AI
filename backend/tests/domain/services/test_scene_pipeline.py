"""Testa o pipeline completo descrito na Etapa 07:

    Image -> ObjectDetector -> Detection -> PositionAnalyzer ->
    ColorAnalyzer -> SceneBuilder -> Scene -> SceneSchema (JSON)

O ObjectDetector é mockado (é a única dependência pesada/externa — o modelo
YOLO); PositionAnalyzer e OpenCVColorAnalyzer rodam de verdade sobre uma
imagem sintética determinística, exercitando a integração real entre eles.
"""

import dataclasses
import io
import json
import uuid
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.api.schemas.scene_schema import SceneSchema
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.scene import Scene
from app.domain.entities.stored_image import StoredImage
from app.domain.protocols.object_detector import ObjectDetector
from app.domain.services.position_analyzer import PositionAnalyzer
from app.domain.services.scene_builder import SceneBuilder
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer

WIDTH = 200
HEIGHT = 100


def make_synthetic_image() -> bytes:
    """Imagem com metade esquerda vermelha e metade direita azul."""
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[:, : WIDTH // 2] = (255, 0, 0)
    array[:, WIDTH // 2 :] = (0, 0, 255)

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def make_mock_detector() -> ObjectDetector:
    detector = MagicMock(spec=ObjectDetector)
    detector.detect.return_value = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(x1=0, y1=0, x2=WIDTH // 2, y2=HEIGHT),
        ),
        Detection(
            class_id=56,
            class_name="chair",
            confidence=0.80,
            bbox=BoundingBox(x1=WIDTH // 2, y1=0, x2=WIDTH, y2=HEIGHT),
        ),
    ]
    return detector


def run_pipeline(image_bytes: bytes, detector: ObjectDetector) -> Scene:
    position_analyzer = PositionAnalyzer()
    color_analyzer = OpenCVColorAnalyzer()

    raw_detections = detector.detect(image_bytes)

    enriched_detections = []
    for detection in raw_detections:
        position = position_analyzer.analyze(detection.bbox, WIDTH, HEIGHT)
        color = color_analyzer.analyze(image_bytes, detection.bbox)
        enriched_detections.append(dataclasses.replace(detection, position=position, color=color))

    stored_image = StoredImage(
        storage_key="2026/08/16/scene.jpg",
        filename="photo.jpg",
        mime_type="image/jpeg",
        size_bytes=len(image_bytes),
        width=WIDTH,
        height=HEIGHT,
        sha256="a" * 64,
    )
    model_metadata = ModelMetadata(name="yolov8n.pt", task="detect", dataset="COCO")

    return SceneBuilder().build(
        image=stored_image,
        model=model_metadata,
        detections=enriched_detections,
        conversation_id=uuid.uuid4(),
    )


def test_full_pipeline_produces_a_fully_enriched_scene() -> None:
    image_bytes = make_synthetic_image()
    detector = make_mock_detector()

    scene = run_pipeline(image_bytes, detector)

    detector.detect.assert_called_once_with(image_bytes)  # type: ignore[attr-defined]
    assert len(scene.objects) == 2
    assert all(obj.position is not None for obj in scene.objects)
    assert all(obj.color is not None for obj in scene.objects)


def test_full_pipeline_computes_correct_position_per_object() -> None:
    scene = run_pipeline(make_synthetic_image(), make_mock_detector())

    person, chair = scene.objects
    assert person.position is not None and person.position.horizontal.value == "left"
    assert chair.position is not None and chair.position.horizontal.value == "right"


def test_full_pipeline_computes_correct_color_per_object() -> None:
    scene = run_pipeline(make_synthetic_image(), make_mock_detector())

    person, chair = scene.objects
    assert person.color is not None and person.color.name.value == "red"
    assert chair.color is not None and chair.color.name.value == "blue"


def test_full_pipeline_output_serializes_to_valid_scene_json() -> None:
    scene = run_pipeline(make_synthetic_image(), make_mock_detector())

    payload = json.loads(SceneSchema.from_domain(scene).to_json())

    assert payload["scene_id"] == str(scene.scene_id)
    assert len(payload["objects"]) == 2
    assert payload["objects"][0]["class"]["name"] == "person"
    assert payload["objects"][0]["position"]["horizontal"] == "left"
    assert payload["objects"][0]["color"]["name"] == "red"
    assert payload["objects"][1]["class"]["name"] == "chair"
    assert payload["objects"][1]["position"]["horizontal"] == "right"
    assert payload["objects"][1]["color"]["name"] == "blue"


def test_full_pipeline_is_deterministic_given_the_same_detector_output() -> None:
    image_bytes = make_synthetic_image()

    first_scene = run_pipeline(image_bytes, make_mock_detector())
    second_scene = run_pipeline(image_bytes, make_mock_detector())

    first_payload = json.loads(SceneSchema.from_domain(first_scene).to_json())
    second_payload = json.loads(SceneSchema.from_domain(second_scene).to_json())

    # scene_id/object_id são UUIDs novos a cada chamada — comparamos tudo,
    # exceto identidade, para garantir que o *conteúdo* computado é estável.
    for payload in (first_payload, second_payload):
        del payload["scene_id"], payload["conversation_id"]
        for obj in payload["objects"]:
            del obj["object_id"]

    assert first_payload == second_payload
