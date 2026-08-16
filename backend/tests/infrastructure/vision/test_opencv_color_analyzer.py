import io

import numpy as np
import pytest
from PIL import Image

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer

WIDTH = 80
HEIGHT = 80
FULL_BBOX = BoundingBox(x1=0, y1=0, x2=WIDTH, y2=HEIGHT)


@pytest.fixture
def analyzer() -> OpenCVColorAnalyzer:
    return OpenCVColorAnalyzer()


def solid_color_image_bytes(rgb: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (WIDTH, HEIGHT), color=rgb).save(buffer, format="PNG")
    return buffer.getvalue()


# Cor RGB canônica para cada bucket, escolhida e verificada manualmente contra
# as regras de classificação (ver módulo opencv_color_analyzer.py).
CANONICAL_COLORS: dict[ColorName, tuple[int, int, int]] = {
    ColorName.BLACK: (0, 0, 0),
    ColorName.WHITE: (255, 255, 255),
    ColorName.GRAY: (128, 128, 128),
    ColorName.RED: (255, 0, 0),
    ColorName.ORANGE: (255, 128, 0),
    ColorName.YELLOW: (255, 255, 0),
    ColorName.GREEN: (0, 255, 0),
    ColorName.CYAN: (0, 255, 255),
    ColorName.BLUE: (0, 0, 255),
    ColorName.PURPLE: (128, 0, 255),
    ColorName.PINK: (255, 192, 203),
    ColorName.BROWN: (139, 69, 19),
}


@pytest.mark.parametrize(("expected_name", "rgb"), list(CANONICAL_COLORS.items()))
def test_classifies_each_canonical_color_correctly(
    analyzer: OpenCVColorAnalyzer, expected_name: ColorName, rgb: tuple[int, int, int]
) -> None:
    result = analyzer.analyze(solid_color_image_bytes(rgb), FULL_BBOX)

    assert result.name == expected_name


def test_solid_color_image_has_full_confidence(analyzer: OpenCVColorAnalyzer) -> None:
    result = analyzer.analyze(solid_color_image_bytes((255, 0, 0)), FULL_BBOX)

    assert result.confidence == pytest.approx(1.0)


def test_confidence_is_always_between_zero_and_one(analyzer: OpenCVColorAnalyzer) -> None:
    result = analyzer.analyze(solid_color_image_bytes((0, 255, 0)), FULL_BBOX)

    assert 0.0 <= result.confidence <= 1.0


def test_rgb_components_are_valid_byte_values(analyzer: OpenCVColorAnalyzer) -> None:
    result = analyzer.analyze(solid_color_image_bytes((0, 0, 255)), FULL_BBOX)

    assert all(0 <= component <= 255 for component in result.rgb)


def test_returned_rgb_is_close_to_the_solid_source_color(analyzer: OpenCVColorAnalyzer) -> None:
    result = analyzer.analyze(solid_color_image_bytes((0, 200, 0)), FULL_BBOX)

    assert result.rgb[1] > result.rgb[0]
    assert result.rgb[1] > result.rgb[2]


def test_small_foreground_square_wins_over_a_different_background() -> None:
    # Fundo azul preenchendo a imagem, com um quadrado vermelho concentrado no
    # centro do bbox — testa que o recorte com inset favorece o primeiro plano
    # central em vez do fundo que toca as bordas do bbox.
    array = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    array[:, :] = (0, 0, 255)  # azul (RGB) preenchendo tudo
    margin = 20
    array[margin : HEIGHT - margin, margin : WIDTH - margin] = (255, 0, 0)  # vermelho no centro

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")

    analyzer = OpenCVColorAnalyzer(inset_ratio=0.3)
    result = analyzer.analyze(buffer.getvalue(), FULL_BBOX)

    assert result.name == ColorName.RED


def test_bbox_is_clamped_to_image_bounds_without_crashing(analyzer: OpenCVColorAnalyzer) -> None:
    oversized_bbox = BoundingBox(x1=-50, y1=-50, x2=WIDTH + 50, y2=HEIGHT + 50)

    result = analyzer.analyze(solid_color_image_bytes((255, 255, 0)), oversized_bbox)

    assert result.name == ColorName.YELLOW


def test_tiny_bbox_skips_inset_and_still_works(analyzer: OpenCVColorAnalyzer) -> None:
    tiny_bbox = BoundingBox(x1=10, y1=10, x2=12, y2=12)

    result = analyzer.analyze(solid_color_image_bytes((0, 255, 255)), tiny_bbox)

    assert result.name == ColorName.CYAN


def test_invalid_image_bytes_raise_a_clear_error(analyzer: OpenCVColorAnalyzer) -> None:
    with pytest.raises(ValueError, match="decodificar"):
        analyzer.analyze(b"not an image", FULL_BBOX)
