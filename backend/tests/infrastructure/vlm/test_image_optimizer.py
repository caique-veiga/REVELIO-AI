from io import BytesIO

from PIL import Image

from app.infrastructure.vlm.image_optimizer import optimize_image


def _png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_optimize_image_downscales_above_max_dimension() -> None:
    original = _png_bytes(2000, 1000)

    optimized = optimize_image(original, max_dimension=768, jpeg_quality=85)

    with Image.open(BytesIO(optimized)) as image:
        assert image.format == "JPEG"
        assert max(image.width, image.height) <= 768
        assert image.width / image.height == 2000 / 1000


def test_optimize_image_reduces_size_for_large_image() -> None:
    original = _png_bytes(2000, 1000)

    optimized = optimize_image(original, max_dimension=768, jpeg_quality=85)

    assert len(optimized) < len(original)


def test_optimize_image_does_not_upscale_small_image() -> None:
    original = _png_bytes(100, 50)

    optimized = optimize_image(original, max_dimension=768, jpeg_quality=85)

    with Image.open(BytesIO(optimized)) as image:
        assert image.width == 100
        assert image.height == 50
