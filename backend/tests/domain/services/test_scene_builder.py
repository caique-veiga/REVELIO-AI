import uuid

import pytest

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName, ColorResult
from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.position import HorizontalPosition, Position, Region, VerticalPosition
from app.domain.entities.stored_image import StoredImage
from app.domain.services.scene_builder import SceneBuilder

POSITION = Position(
    horizontal=HorizontalPosition.CENTER,
    vertical=VerticalPosition.MIDDLE,
    region=Region.FRONT_CENTER,
)
COLOR = ColorResult(name=ColorName.BLUE, rgb=(20, 80, 180), confidence=0.82)


def make_image(**overrides: object) -> StoredImage:
    defaults: dict[str, object] = {
        "storage_key": "2026/08/16/scene.jpg",
        "filename": "photo.jpg",
        "mime_type": "image/jpeg",
        "size_bytes": 204800,
        "width": 1920,
        "height": 1080,
        "sha256": "a" * 64,
    }
    defaults.update(overrides)
    return StoredImage(**defaults)  # type: ignore[arg-type]


def make_model() -> ModelMetadata:
    return ModelMetadata(name="yolov8n.pt", task="detect", dataset="COCO")


def make_detection(**overrides: object) -> Detection:
    defaults: dict[str, object] = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.98,
        "bbox": BoundingBox(x1=100, y1=200, x2=500, y2=900),
        "position": POSITION,
        "color": COLOR,
    }
    defaults.update(overrides)
    return Detection(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def builder() -> SceneBuilder:
    return SceneBuilder()


def test_builds_a_scene_from_image_model_and_detections(builder: SceneBuilder) -> None:
    image = make_image()
    model = make_model()
    detection = make_detection()

    scene = builder.build(image=image, model=model, detections=[detection])

    assert scene.image == image
    assert scene.model == model
    assert scene.objects == [detection]


def test_scene_gets_a_freshly_generated_id(builder: SceneBuilder) -> None:
    scene = builder.build(image=make_image(), model=make_model(), detections=[])

    assert isinstance(scene.scene_id, uuid.UUID)


def test_conversation_id_defaults_to_none(builder: SceneBuilder) -> None:
    scene = builder.build(image=make_image(), model=make_model(), detections=[])

    assert scene.conversation_id is None


def test_conversation_id_can_be_provided(builder: SceneBuilder) -> None:
    conversation_id = uuid.uuid4()

    scene = builder.build(
        image=make_image(), model=make_model(), detections=[], conversation_id=conversation_id
    )

    assert scene.conversation_id == conversation_id


def test_preserves_the_order_of_detections(builder: SceneBuilder) -> None:
    first = make_detection(class_name="person")
    second = make_detection(class_name="chair")

    scene = builder.build(image=make_image(), model=make_model(), detections=[first, second])

    assert [obj.class_name for obj in scene.objects] == ["person", "chair"]


def test_raises_when_a_detection_has_no_position(builder: SceneBuilder) -> None:
    detection = make_detection(position=None)

    with pytest.raises(ValueError, match="position"):
        builder.build(image=make_image(), model=make_model(), detections=[detection])


def test_raises_when_a_detection_has_no_color(builder: SceneBuilder) -> None:
    detection = make_detection(color=None)

    with pytest.raises(ValueError, match="color"):
        builder.build(image=make_image(), model=make_model(), detections=[detection])
