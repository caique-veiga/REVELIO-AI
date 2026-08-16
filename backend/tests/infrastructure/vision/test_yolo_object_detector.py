from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.domain.entities.detection import Detection
from app.infrastructure.vision.yolo_object_detector import YOLOObjectDetector


class FakeBox:
    def __init__(self, class_id: int, confidence: float, xyxy: tuple[int, int, int, int]) -> None:
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [xyxy]


def make_result(names: dict[int, str], boxes: list[FakeBox]) -> SimpleNamespace:
    return SimpleNamespace(names=names, boxes=boxes)


@patch("app.infrastructure.vision.yolo_object_detector.YOLO")
def test_detect_converts_yolo_results_into_domain_detections(
    mock_yolo_cls: MagicMock, jpeg_bytes: bytes
) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.predict.return_value = [
        make_result(
            names={0: "person", 56: "chair"},
            boxes=[
                FakeBox(class_id=0, confidence=0.93, xyxy=(100, 200, 500, 900)),
                FakeBox(class_id=56, confidence=0.61, xyxy=(10, 20, 30, 40)),
            ],
        )
    ]

    detector = YOLOObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    detections = detector.detect(jpeg_bytes)

    assert len(detections) == 2
    assert all(isinstance(detection, Detection) for detection in detections)

    first, second = detections
    assert first.class_id == 0
    assert first.class_name == "person"
    assert first.confidence == 0.93
    assert (first.bbox.x1, first.bbox.y1, first.bbox.x2, first.bbox.y2) == (100, 200, 500, 900)

    assert second.class_id == 56
    assert second.class_name == "chair"


@patch("app.infrastructure.vision.yolo_object_detector.YOLO")
def test_detect_returns_empty_list_when_nothing_is_detected(
    mock_yolo_cls: MagicMock, jpeg_bytes: bytes
) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.predict.return_value = [make_result(names={}, boxes=[])]

    detector = YOLOObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)

    assert detector.detect(jpeg_bytes) == []


@patch("app.infrastructure.vision.yolo_object_detector.YOLO")
def test_detect_passes_confidence_threshold_to_the_model(
    mock_yolo_cls: MagicMock, jpeg_bytes: bytes
) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.predict.return_value = [make_result(names={}, boxes=[])]

    detector = YOLOObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.75)
    detector.detect(jpeg_bytes)

    _, kwargs = mock_model.predict.call_args
    assert kwargs["conf"] == 0.75


@patch("app.infrastructure.vision.yolo_object_detector.YOLO")
def test_detect_assigns_a_unique_object_id_per_detection(
    mock_yolo_cls: MagicMock, jpeg_bytes: bytes
) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_model.predict.return_value = [
        make_result(
            names={0: "person"},
            boxes=[
                FakeBox(class_id=0, confidence=0.9, xyxy=(0, 0, 1, 1)),
                FakeBox(class_id=0, confidence=0.8, xyxy=(2, 2, 3, 3)),
            ],
        )
    ]

    detector = YOLOObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    detections = detector.detect(jpeg_bytes)

    object_ids = {detection.object_id for detection in detections}
    assert len(object_ids) == 2


@patch("app.infrastructure.vision.yolo_object_detector.YOLO")
def test_constructor_loads_the_model_from_the_given_path(mock_yolo_cls: MagicMock) -> None:
    YOLOObjectDetector(model_path="custom-weights.pt", confidence_threshold=0.5)

    mock_yolo_cls.assert_called_once_with("custom-weights.pt")
