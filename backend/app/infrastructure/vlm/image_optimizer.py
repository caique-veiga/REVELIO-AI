import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def optimize_image(content: bytes, max_dimension: int, jpeg_quality: int) -> bytes:
    """Reduz dimensão e recodifica como JPEG antes de enviar a um provider remoto.

    Usado apenas na chamada à VLM (não na imagem persistida em disco) — o
    objetivo é reduzir custo/latência de upload, não a qualidade guardada
    para o histórico da Scene.
    """
    with Image.open(io.BytesIO(content)) as source:
        rgb_image = source.convert("RGB")

    if max(rgb_image.width, rgb_image.height) > max_dimension:
        rgb_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=jpeg_quality)
    optimized = buffer.getvalue()
    width, height = rgb_image.width, rgb_image.height

    reduction_pct = 100 * (1 - len(optimized) / len(content)) if content else 0.0
    logger.info(
        "image optimization original_bytes=%d optimized_bytes=%d reduction_pct=%.0f "
        "width=%d height=%d",
        len(content),
        len(optimized),
        reduction_pct,
        width,
        height,
    )
    return optimized
