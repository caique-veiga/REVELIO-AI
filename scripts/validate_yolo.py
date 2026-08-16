"""Real end-to-end validation of YOLOObjectDetector against real images.

Loads the actual YOLO model (no mocks, no synthetic results) and runs
inference over every image directly inside a given directory, printing one
JSON block per image plus a final aggregate summary.

Usage:
    uv run python scripts/validate_yolo.py <path-to-images-directory>
"""

import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from app.config.settings import get_settings
from app.domain.entities.detection import Detection
from app.infrastructure.vision.yolo_object_detector import YOLOObjectDetector

ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEPARATOR = "=" * 60


def discover_images(directory: Path) -> list[Path]:
    """List image files directly inside `directory` (non-recursive)."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS
    )


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_bbox(detection: Detection, width: int, height: int) -> list[str]:
    """Return a list of human-readable problems with the bbox, if any."""
    bbox = detection.bbox
    problems: list[str] = []

    if not bbox.x1 < bbox.x2:
        problems.append(f"x1 ({bbox.x1}) não é menor que x2 ({bbox.x2})")
    if not bbox.y1 < bbox.y2:
        problems.append(f"y1 ({bbox.y1}) não é menor que y2 ({bbox.y2})")
    for name, value, limit in (
        ("x1", bbox.x1, width),
        ("x2", bbox.x2, width),
        ("y1", bbox.y1, height),
        ("y2", bbox.y2, height),
    ):
        if not 0 <= value <= limit:
            problems.append(f"{name} ({value}) fora dos limites [0, {limit}]")

    return problems


def build_detection_dict(detection: Detection) -> dict[str, object]:
    bbox = detection.bbox
    return {
        "object_id": str(detection.object_id),
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(detection.confidence, 4),
        "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2},
    }


def build_classes_summary(detections: list[Detection]) -> dict[str, int]:
    counts = Counter(detection.class_name for detection in detections)
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def process_image(
    detector: YOLOObjectDetector, path: Path
) -> tuple[dict[str, object], list[Detection], float]:
    content = path.read_bytes()

    with Image.open(io.BytesIO(content)) as probe:
        width, height = probe.size

    started_at = time.perf_counter()
    detections = detector.detect(content)
    inference_time_ms = (time.perf_counter() - started_at) * 1000

    for detection in detections:
        for problem in validate_bbox(detection, width, height):
            print(f"⚠ aviso: {detection.class_name} ({detection.object_id}) — {problem}")

    result: dict[str, object] = {
        "image": {"filename": path.name, "width": width, "height": height},
        "detections": [build_detection_dict(detection) for detection in detections],
        "summary": {
            "total_objects": len(detections),
            "classes": build_classes_summary(detections),
        },
        "inference_time_ms": round(inference_time_ms, 2),
    }
    return result, detections, inference_time_ms


def print_header(filename: str) -> None:
    print(SEPARATOR)
    print(f"IMAGE: {filename}")
    print(SEPARATOR)


def load_detector() -> YOLOObjectDetector:
    settings = get_settings()
    device = detect_device()
    print(
        f"Carregando modelo '{settings.yolo_model}' "
        f"(limiar de confiança: {settings.yolo_confidence_threshold}, device: {device})..."
    )
    try:
        detector = YOLOObjectDetector(
            model_path=settings.yolo_model,
            confidence_threshold=settings.yolo_confidence_threshold,
        )
    except Exception as exc:
        print(f"ERRO FATAL: falha ao carregar o modelo YOLO '{settings.yolo_model}': {exc}")
        print(
            "Verifique sua conexão com a internet (o Ultralytics baixa os pesos "
            "automaticamente no primeiro uso) ou aponte YOLO_MODEL para um arquivo "
            ".pt local já existente."
        )
        raise SystemExit(1) from exc

    print("Modelo carregado com sucesso.\n")
    return detector


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/validate_yolo.py <path-to-images-directory>")
        raise SystemExit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"ERRO: diretório não encontrado: {directory}")
        raise SystemExit(1)

    images = discover_images(directory)
    if not images:
        print(
            f"ERRO: nenhuma imagem encontrada em '{directory}' "
            f"(formatos aceitos: {', '.join(sorted(ACCEPTED_EXTENSIONS))})"
        )
        raise SystemExit(1)

    detector = load_detector()

    images_processed = 0
    images_failed = 0
    total_detections = 0
    classes_detected: Counter[str] = Counter()
    inference_times_ms: list[float] = []

    for path in images:
        print_header(path.name)
        try:
            result, detections, inference_time_ms = process_image(detector, path)
        except Exception as exc:
            images_failed += 1
            print(f"ERRO: falha ao processar '{path.name}': {exc}")
            continue

        print(json.dumps(result, indent=2, ensure_ascii=False))

        images_processed += 1
        total_detections += len(detections)
        classes_detected.update(detection.class_name for detection in detections)
        inference_times_ms.append(inference_time_ms)

    print(SEPARATOR)
    print("FINAL SUMMARY")
    print(SEPARATOR)

    performance: dict[str, object] = {"device": detect_device()}
    if inference_times_ms:
        performance["average_inference_time_ms"] = round(
            sum(inference_times_ms) / len(inference_times_ms), 2
        )
        performance["min_inference_time_ms"] = round(min(inference_times_ms), 2)
        performance["max_inference_time_ms"] = round(max(inference_times_ms), 2)

    final_summary = {
        "images_processed": images_processed,
        "images_failed": images_failed,
        "total_detections": total_detections,
        "classes_detected": dict(
            sorted(classes_detected.items(), key=lambda item: item[1], reverse=True)
        ),
        "performance": performance,
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
