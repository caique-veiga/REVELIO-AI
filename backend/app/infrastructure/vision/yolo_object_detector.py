import io
import uuid
from typing import cast

import numpy as np
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models import YOLO

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.detection import Detection


class YOLOObjectDetector:
    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold

    def detect(self, image: bytes) -> list[Detection]:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        results = self._model.predict(
            source=np.asarray(pil_image),
            conf=self._confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in cast(list[Results], results):
            boxes = result.boxes
            if boxes is None:
                continue

            class_names = result.names
            for index in range(len(boxes)):
                box = boxes[index]
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
                detections.append(
                    Detection(
                        object_id=uuid.uuid4(),
                        class_id=class_id,
                        class_name=class_names[class_id],
                        confidence=float(box.conf[0]),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return detections
