import pytest

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.position import HorizontalPosition, Region, VerticalPosition
from app.domain.services.position_analyzer import PositionAnalyzer

WIDTH = 1920
HEIGHT = 1080


@pytest.fixture
def analyzer() -> PositionAnalyzer:
    return PositionAnalyzer()


def bbox_around(center_x: int, center_y: int, half_size: int = 5) -> BoundingBox:
    return BoundingBox(
        x1=center_x - half_size,
        y1=center_y - half_size,
        x2=center_x + half_size,
        y2=center_y + half_size,
    )


# --- os nove cantos/regiões do grid 3x3 --------------------------------------


@pytest.mark.parametrize(
    ("center_x", "center_y", "horizontal", "vertical", "region"),
    [
        (100, 100, HorizontalPosition.LEFT, VerticalPosition.TOP, Region.UPPER_LEFT),
        (960, 100, HorizontalPosition.CENTER, VerticalPosition.TOP, Region.UPPER_CENTER),
        (1800, 100, HorizontalPosition.RIGHT, VerticalPosition.TOP, Region.UPPER_RIGHT),
        (100, 540, HorizontalPosition.LEFT, VerticalPosition.MIDDLE, Region.FRONT_LEFT),
        (960, 540, HorizontalPosition.CENTER, VerticalPosition.MIDDLE, Region.FRONT_CENTER),
        (1800, 540, HorizontalPosition.RIGHT, VerticalPosition.MIDDLE, Region.FRONT_RIGHT),
        (100, 980, HorizontalPosition.LEFT, VerticalPosition.BOTTOM, Region.LOWER_LEFT),
        (960, 980, HorizontalPosition.CENTER, VerticalPosition.BOTTOM, Region.LOWER_CENTER),
        (1800, 980, HorizontalPosition.RIGHT, VerticalPosition.BOTTOM, Region.LOWER_RIGHT),
    ],
)
def test_maps_center_to_expected_grid_cell(
    analyzer: PositionAnalyzer,
    center_x: int,
    center_y: int,
    horizontal: HorizontalPosition,
    vertical: VerticalPosition,
    region: Region,
) -> None:
    position = analyzer.analyze(bbox_around(center_x, center_y), WIDTH, HEIGHT)

    assert position.horizontal == horizontal
    assert position.vertical == vertical
    assert position.region == region


# --- centro exato da imagem --------------------------------------------------


def test_bbox_covering_the_whole_image_is_front_center(analyzer: PositionAnalyzer) -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=WIDTH, y2=HEIGHT)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.horizontal == HorizontalPosition.CENTER
    assert position.vertical == VerticalPosition.MIDDLE
    assert position.region == Region.FRONT_CENTER


# --- bordas / limiares exatos entre faixas -----------------------------------


def test_center_exactly_at_left_third_boundary_is_center(analyzer: PositionAnalyzer) -> None:
    third = WIDTH / 3
    bbox = BoundingBox(x1=int(third), y1=0, x2=int(third), y2=0)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.horizontal == HorizontalPosition.CENTER


def test_center_exactly_at_right_third_boundary_is_right(analyzer: PositionAnalyzer) -> None:
    two_thirds = int(2 * WIDTH / 3)
    bbox = BoundingBox(x1=two_thirds, y1=0, x2=two_thirds, y2=0)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.horizontal == HorizontalPosition.RIGHT


def test_center_exactly_at_top_third_boundary_is_middle(analyzer: PositionAnalyzer) -> None:
    third = int(HEIGHT / 3)
    bbox = BoundingBox(x1=0, y1=third, x2=0, y2=third)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.vertical == VerticalPosition.MIDDLE


def test_center_exactly_at_bottom_third_boundary_is_bottom(analyzer: PositionAnalyzer) -> None:
    two_thirds = int(2 * HEIGHT / 3)
    bbox = BoundingBox(x1=0, y1=two_thirds, x2=0, y2=two_thirds)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.vertical == VerticalPosition.BOTTOM


def test_bbox_touching_the_top_left_pixel_is_upper_left(analyzer: PositionAnalyzer) -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=0, y2=0)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.region == Region.UPPER_LEFT


def test_bbox_touching_the_bottom_right_pixel_is_lower_right(analyzer: PositionAnalyzer) -> None:
    bbox = BoundingBox(x1=WIDTH, y1=HEIGHT, x2=WIDTH, y2=HEIGHT)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.region == Region.LOWER_RIGHT


# --- objetos pequenos e grandes -----------------------------------------------


def test_tiny_object_in_a_corner_is_positioned_precisely(analyzer: PositionAnalyzer) -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=2, y2=2)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.region == Region.UPPER_LEFT


def test_large_object_off_center_follows_its_center_not_its_extent(
    analyzer: PositionAnalyzer,
) -> None:
    # Ocupa quase a imagem inteira, mas o centro está deslocado para a direita.
    bbox = BoundingBox(x1=1200, y1=0, x2=WIDTH, y2=HEIGHT)

    position = analyzer.analyze(bbox, WIDTH, HEIGHT)

    assert position.horizontal == HorizontalPosition.RIGHT


# --- diferentes resoluções ----------------------------------------------------


@pytest.mark.parametrize(("width", "height"), [(100, 100), (640, 480), (1920, 1080), (4000, 3000)])
def test_bbox_covering_the_left_third_is_left_regardless_of_resolution(
    analyzer: PositionAnalyzer, width: int, height: int
) -> None:
    bbox = BoundingBox(x1=0, y1=0, x2=width // 4, y2=height)

    position = analyzer.analyze(bbox, width, height)

    assert position.horizontal == HorizontalPosition.LEFT


@pytest.mark.parametrize(("width", "height"), [(100, 100), (640, 480), (1920, 1080), (4000, 3000)])
def test_bbox_at_the_exact_center_is_front_center_regardless_of_resolution(
    analyzer: PositionAnalyzer, width: int, height: int
) -> None:
    position = analyzer.analyze(bbox_around(width // 2, height // 2), width, height)

    assert position.region == Region.FRONT_CENTER


def test_odd_resolution_does_not_crash_and_produces_a_valid_region(
    analyzer: PositionAnalyzer,
) -> None:
    position = analyzer.analyze(bbox_around(320, 240), 641, 481)

    assert position.region in list(Region)
