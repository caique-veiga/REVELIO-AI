import json
import uuid

from app.api.schemas.scene_schema import SceneSchema
from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName, ColorResult
from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.position import HorizontalPosition, Position, Region, VerticalPosition
from app.domain.entities.scene import Scene
from app.domain.entities.stored_image import StoredImage


def make_scene() -> Scene:
    image = StoredImage(
        storage_key="2026/08/16/scene.jpg",
        filename="photo.jpg",
        mime_type="image/jpeg",
        size_bytes=204800,
        width=1920,
        height=1080,
        sha256="a" * 64,
    )
    model = ModelMetadata(name="yolov8n.pt", task="detect", dataset="COCO")
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.98,
        bbox=BoundingBox(x1=100, y1=200, x2=500, y2=900),
        position=Position(
            horizontal=HorizontalPosition.CENTER,
            vertical=VerticalPosition.MIDDLE,
            region=Region.FRONT_CENTER,
        ),
        color=ColorResult(name=ColorName.BLUE, rgb=(20, 80, 180), confidence=0.82),
    )
    return Scene(
        image=image,
        model=model,
        objects=[detection],
        conversation_id=uuid.uuid4(),
    )


def test_json_matches_the_scene_json_shape_from_the_project_context() -> None:
    scene = make_scene()

    payload = json.loads(SceneSchema.from_domain(scene).to_json())

    assert set(payload.keys()) == {"scene_id", "conversation_id", "image", "model", "objects"}
    assert set(payload["image"].keys()) == {"storage_key", "width", "height"}
    assert set(payload["model"].keys()) == {"name", "task", "dataset"}

    (obj,) = payload["objects"]
    assert set(obj.keys()) == {"object_id", "class", "bbox", "position", "color"}
    assert set(obj["class"].keys()) == {"id", "name", "confidence"}
    assert set(obj["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
    assert set(obj["position"].keys()) == {"horizontal", "vertical", "region"}
    assert set(obj["color"].keys()) == {"name", "rgb", "confidence"}


def test_class_key_is_literally_class_not_class_underscore() -> None:
    scene = make_scene()

    payload = json.loads(SceneSchema.from_domain(scene).to_json())

    assert "class" in payload["objects"][0]
    assert "class_" not in payload["objects"][0]


def test_values_round_trip_correctly() -> None:
    scene = make_scene()
    detection = scene.objects[0]
    assert detection.position is not None
    assert detection.color is not None

    payload = json.loads(SceneSchema.from_domain(scene).to_json())
    obj = payload["objects"][0]

    assert payload["scene_id"] == str(scene.scene_id)
    assert payload["conversation_id"] == str(scene.conversation_id)
    assert payload["image"]["storage_key"] == scene.image.storage_key
    assert payload["image"]["width"] == scene.image.width
    assert payload["model"]["name"] == scene.model.name
    assert obj["object_id"] == str(detection.object_id)
    assert obj["class"]["id"] == detection.class_id
    assert obj["class"]["name"] == detection.class_name
    assert obj["class"]["confidence"] == detection.confidence
    assert obj["bbox"] == {"x1": 100, "y1": 200, "x2": 500, "y2": 900}
    assert obj["position"]["horizontal"] == detection.position.horizontal.value
    assert obj["color"]["name"] == detection.color.name.value
    assert obj["color"]["rgb"] == list(detection.color.rgb)


def test_conversation_id_can_be_null() -> None:
    scene = make_scene()
    scene_without_conversation = Scene(
        image=scene.image, model=scene.model, objects=scene.objects, scene_id=scene.scene_id
    )

    payload = json.loads(SceneSchema.from_domain(scene_without_conversation).to_json())

    assert payload["conversation_id"] is None


def test_serialization_is_deterministic_across_calls() -> None:
    scene = make_scene()
    schema = SceneSchema.from_domain(scene)

    assert schema.to_json() == schema.to_json()


def test_objects_preserve_input_order() -> None:
    scene = make_scene()
    second_detection = Detection(
        class_id=56,
        class_name="chair",
        confidence=0.7,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        position=scene.objects[0].position,
        color=scene.objects[0].color,
    )
    two_object_scene = Scene(
        image=scene.image,
        model=scene.model,
        objects=[scene.objects[0], second_detection],
        scene_id=scene.scene_id,
    )

    payload = json.loads(SceneSchema.from_domain(two_object_scene).to_json())

    assert [obj["class"]["name"] for obj in payload["objects"]] == ["person", "chair"]
