from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName, ColorResult
from app.domain.entities.detection import Detection
from app.domain.entities.position import HorizontalPosition, Position, Region, VerticalPosition

BBOX = BoundingBox(x1=0, y1=0, x2=10, y2=10)


def test_detection_has_no_position_by_default() -> None:
    detection = Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBOX)

    assert detection.position is None


def test_detection_can_carry_a_position() -> None:
    position = Position(
        horizontal=HorizontalPosition.CENTER,
        vertical=VerticalPosition.MIDDLE,
        region=Region.FRONT_CENTER,
    )

    detection = Detection(
        class_id=0, class_name="person", confidence=0.9, bbox=BBOX, position=position
    )

    assert detection.position == position


def test_detection_has_no_color_by_default() -> None:
    detection = Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBOX)

    assert detection.color is None


def test_detection_can_carry_a_color() -> None:
    color = ColorResult(name=ColorName.BLUE, rgb=(20, 80, 180), confidence=0.82)

    detection = Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBOX, color=color)

    assert detection.color == color
