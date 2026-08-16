import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities.detection import Detection
from app.domain.entities.scene import Scene


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ClassSchema(BaseModel):
    id: int
    name: str
    confidence: float


class PositionSchema(BaseModel):
    horizontal: str
    vertical: str
    region: str


class ColorSchema(BaseModel):
    name: str
    rgb: tuple[int, int, int]
    confidence: float


class ObjectSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    object_id: uuid.UUID
    class_: ClassSchema = Field(alias="class")
    bbox: BoundingBoxSchema
    position: PositionSchema
    color: ColorSchema

    @classmethod
    def from_domain(cls, detection: Detection) -> "ObjectSchema":
        assert detection.position is not None
        assert detection.color is not None

        return cls(
            object_id=detection.object_id,
            class_=ClassSchema(
                id=detection.class_id,
                name=detection.class_name,
                confidence=detection.confidence,
            ),
            bbox=BoundingBoxSchema(
                x1=detection.bbox.x1,
                y1=detection.bbox.y1,
                x2=detection.bbox.x2,
                y2=detection.bbox.y2,
            ),
            position=PositionSchema(
                horizontal=detection.position.horizontal.value,
                vertical=detection.position.vertical.value,
                region=detection.position.region.value,
            ),
            color=ColorSchema(
                name=detection.color.name.value,
                rgb=detection.color.rgb,
                confidence=detection.color.confidence,
            ),
        )


class ImageSchema(BaseModel):
    storage_key: str
    width: int
    height: int


class ModelSchema(BaseModel):
    name: str
    task: str
    dataset: str


class SceneSchema(BaseModel):
    scene_id: uuid.UUID
    conversation_id: uuid.UUID | None
    image: ImageSchema
    model: ModelSchema
    objects: list[ObjectSchema]

    @classmethod
    def from_domain(cls, scene: Scene) -> "SceneSchema":
        return cls(
            scene_id=scene.scene_id,
            conversation_id=scene.conversation_id,
            image=ImageSchema(
                storage_key=scene.image.storage_key,
                width=scene.image.width,
                height=scene.image.height,
            ),
            model=ModelSchema(
                name=scene.model.name, task=scene.model.task, dataset=scene.model.dataset
            ),
            objects=[ObjectSchema.from_domain(detection) for detection in scene.objects],
        )

    def to_json(self) -> str:
        """Serializa no formato exato do Scene JSON (com a chave `class`)."""
        return self.model_dump_json(by_alias=True, indent=2)
