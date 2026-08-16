"""Manually run the YOLO object detector against a real image.

Uses YOLO_MODEL and YOLO_CONFIDENCE_THRESHOLD from the environment/.env
(see Settings). On first use, ultralytics downloads the pretrained COCO
weights automatically if they are not already cached locally — see the
"Detecção de objetos (YOLO)" section in the README for details.

Usage:
    uv run python scripts/test_detection.py path/to/image.jpg
"""

import sys
from pathlib import Path

from app.config.settings import get_settings
from app.infrastructure.vision.yolo_object_detector import YOLOObjectDetector


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/test_detection.py <path-to-image>")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Arquivo não encontrado: {image_path}")
        raise SystemExit(1)

    settings = get_settings()
    print(
        f"Carregando modelo '{settings.yolo_model}' (limiar de confiança: "
        f"{settings.yolo_confidence_threshold})..."
    )
    detector = YOLOObjectDetector(
        model_path=settings.yolo_model,
        confidence_threshold=settings.yolo_confidence_threshold,
    )

    detections = detector.detect(image_path.read_bytes())

    if not detections:
        print("Nenhum objeto detectado.")
        return

    print(f"\n{len(detections)} objeto(s) detectado(s):\n")
    for detection in detections:
        bbox = detection.bbox
        print(
            f"- {detection.class_name} (class_id={detection.class_id}) "
            f"confiança={detection.confidence:.2f} "
            f"bbox=({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})"
        )


if __name__ == "__main__":
    main()
