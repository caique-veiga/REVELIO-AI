import uuid

from app.domain.entities.detection import Detection
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.entities.scene import Scene
from app.domain.entities.stored_image import StoredImage


class SceneBuilder:
    """Assembles a Scene from an image, detector metadata, and detections.

    This is the final step of the pipeline:

        Image -> ObjectDetector -> Detection -> PositionAnalyzer ->
        ColorAnalyzer -> SceneBuilder -> Scene

    It does not call PositionAnalyzer or ColorAnalyzer itself — by the time a
    detection reaches the builder, its `position` and `color` must already be
    set by those earlier pipeline steps.
    """

    def build(
        self,
        image: StoredImage,
        model: ModelMetadata,
        detections: list[Detection],
        conversation_id: uuid.UUID | None = None,
    ) -> Scene:
        for detection in detections:
            if detection.position is None:
                raise ValueError(f"Detection {detection.object_id} não possui position definida.")
            if detection.color is None:
                raise ValueError(f"Detection {detection.object_id} não possui color definida.")

        return Scene(
            image=image,
            model=model,
            objects=detections,
            conversation_id=conversation_id,
        )
