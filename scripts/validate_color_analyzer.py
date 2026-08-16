"""Real end-to-end validation of OpenCVColorAnalyzer against real YOLO detections.

For every image in the given directory: runs the real YOLO detector (no
mocks), crops each detected region from the original image, classifies its
dominant color with OpenCVColorAnalyzer, saves the crop to
`<images-directory>/output/{class_name}_{n}.jpg`, and writes a summary of
every detection + its color to `<images-directory>/output/results.json`.

Usage:
    uv run python scripts/validate_color_analyzer.py <path-to-images-directory>
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from app.config.settings import get_settings
from app.domain.entities.bounding_box import BoundingBox
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer
from app.infrastructure.vision.yolo_object_detector import YOLOObjectDetector

ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_DIR_NAME = "output"


def discover_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS
    )


def clamp_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
    x1 = max(0, min(bbox.x1, width - 1))
    y1 = max(0, min(bbox.y1, height - 1))
    x2 = max(x1 + 1, min(bbox.x2, width))
    y2 = max(y1 + 1, min(bbox.y2, height))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/validate_color_analyzer.py <path-to-images-directory>")
        raise SystemExit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"ERRO: diretório não encontrado: {directory}")
        raise SystemExit(1)

    images = discover_images(directory)
    if not images:
        print(f"ERRO: nenhuma imagem encontrada em '{directory}'")
        raise SystemExit(1)

    output_dir = directory / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    print(f"Carregando modelo '{settings.yolo_model}'...")
    detector = YOLOObjectDetector(
        model_path=settings.yolo_model,
        confidence_threshold=settings.yolo_confidence_threshold,
    )
    color_analyzer = OpenCVColorAnalyzer()
    print("Modelo carregado.\n")

    class_counters: Counter[str] = Counter()
    results: list[dict[str, object]] = []
    images_failed = 0

    for path in images:
        try:
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as probe:
                width, height = probe.size
                rgb_image = probe.convert("RGB")

            detections = detector.detect(content)
        except Exception as exc:
            images_failed += 1
            print(f"{path.name}: ERRO ao processar — {exc}")
            continue

        print(f"{path.name}: {len(detections)} objeto(s) detectado(s)")

        for detection in detections:
            class_counters[detection.class_name] += 1
            number = class_counters[detection.class_name]
            crop_filename = f"{detection.class_name}_{number}.jpg"

            bbox = clamp_bbox(detection.bbox, width, height)
            rgb_image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2)).save(output_dir / crop_filename)

            color = color_analyzer.analyze(content, detection.bbox)

            results.append(
                {
                    "image": path.name,
                    "crop_filename": crop_filename,
                    "class_name": detection.class_name,
                    "class_id": detection.class_id,
                    "detection_confidence": round(detection.confidence, 4),
                    "bbox": {
                        "x1": detection.bbox.x1,
                        "y1": detection.bbox.y1,
                        "x2": detection.bbox.x2,
                        "y2": detection.bbox.y2,
                    },
                    "color": {
                        "name": color.name.value,
                        "rgb": list(color.rgb),
                        "confidence": round(color.confidence, 4),
                    },
                }
            )

            print(
                f"  - {crop_filename}: cor={color.name.value} "
                f"(confiança={color.confidence:.2f}, rgb={color.rgb})"
            )

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "images_processed": len(images) - images_failed,
                "images_failed": images_failed,
                "total_detections": len(results),
                "detections": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{len(results)} detecção(ões) salva(s) em '{output_dir}/' (crops + {results_path.name})."
    )


if __name__ == "__main__":
    main()
