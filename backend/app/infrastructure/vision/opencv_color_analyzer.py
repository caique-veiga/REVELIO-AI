from collections import Counter

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.color_result import ColorName, ColorResult

# Tamanho para o qual o recorte é reduzido antes da classificação: mantém o
# custo constante mesmo para bounding boxes muito grandes, e a média implícita
# do downscale já reduz um pouco o ruído de reflexos/JPEG.
_ANALYSIS_SIZE = 48

# Fração descartada de cada borda do bbox antes da análise, para reduzir a
# chance do fundo (que costuma aparecer nas bordas de um bbox retangular)
# dominar o resultado.
_INSET_RATIO = 0.15

# Regras de classificação, avaliadas em ordem — a primeira que casar "vence"
# por pixel. Mantidas como uma lista simples e ordenável para permitir trocar
# o classificador (limiares, algoritmo, ou até um modelo treinado) sem alterar
# o resto do pipeline (ver módulo docstring da classe).
_BLACK_MAX_VALUE = 45
_WHITE_MIN_VALUE = 200
_GRAYSCALE_MAX_SATURATION = 30
_BROWN_MAX_VALUE = 140
_BROWN_MIN_SATURATION = 60
_PINK_MAX_SATURATION = 120
_PINK_MIN_VALUE = 150


def _build_classification_rules(
    hue: NDArray[np.int32], saturation: NDArray[np.int32], value: NDArray[np.int32]
) -> list[tuple[ColorName, NDArray[np.bool_]]]:
    reddish_hue = (hue <= 10) | (hue >= 160)
    dark_reddish_hue = (hue <= 20) | (hue >= 170)

    return [
        (ColorName.BLACK, value < _BLACK_MAX_VALUE),
        (
            ColorName.WHITE,
            (saturation < _GRAYSCALE_MAX_SATURATION) & (value > _WHITE_MIN_VALUE),
        ),
        (ColorName.GRAY, saturation < _GRAYSCALE_MAX_SATURATION),
        (
            ColorName.BROWN,
            (value < _BROWN_MAX_VALUE) & (saturation >= _BROWN_MIN_SATURATION) & dark_reddish_hue,
        ),
        (
            ColorName.PINK,
            reddish_hue & (saturation < _PINK_MAX_SATURATION) & (value > _PINK_MIN_VALUE),
        ),
        (ColorName.RED, reddish_hue),
        (ColorName.ORANGE, (hue > 10) & (hue <= 20)),
        (ColorName.YELLOW, (hue > 20) & (hue <= 34)),
        (ColorName.GREEN, (hue > 34) & (hue <= 85)),
        (ColorName.CYAN, (hue > 85) & (hue <= 100)),
        (ColorName.BLUE, (hue > 100) & (hue <= 130)),
        (ColorName.PURPLE, (hue > 130) & (hue <= 159)),
    ]


class OpenCVColorAnalyzer:
    """Estima a cor predominante da região de um objeto detectado.

    Esta é apenas "a cor predominante da região detectada" — não tenta
    localizar semanticamente uma parte específica do objeto (ex.: a camisa de
    uma pessoa, uma calça, um sapato). Ver README para limitações conhecidas
    (iluminação, reflexos, fundo, bounding boxes grandes).

    O classificador (regras de matiz/saturação/valor em `_build_classification_rules`)
    é isolado do restante do pipeline de propósito, para poder ser substituído
    futuramente (ex.: por clustering ou um modelo treinado) sem alterar a
    interface pública `ColorAnalyzer`.
    """

    def __init__(
        self, inset_ratio: float = _INSET_RATIO, analysis_size: int = _ANALYSIS_SIZE
    ) -> None:
        self._inset_ratio = inset_ratio
        self._analysis_size = analysis_size

    def analyze(self, image: bytes, bbox: BoundingBox) -> ColorResult:
        bgr_image = self._decode(image)
        crop = self._crop(bgr_image, bbox)
        resized = cv2.resize(
            crop, (self._analysis_size, self._analysis_size), interpolation=cv2.INTER_AREA
        )
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        labels, counts = self._classify_pixels(hsv)
        winner_name, winner_count = counts.most_common(1)[0]
        confidence = winner_count / labels.size

        rgb = self._mean_rgb(resized, labels, winner_name)

        return ColorResult(name=winner_name, rgb=rgb, confidence=confidence)

    def _decode(self, image: bytes) -> MatLike:
        buffer = np.frombuffer(image, dtype=np.uint8)
        bgr_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("Não foi possível decodificar a imagem.")
        return bgr_image

    def _crop(self, bgr_image: MatLike, bbox: BoundingBox) -> MatLike:
        height, width = bgr_image.shape[:2]
        x1 = max(0, min(bbox.x1, width - 1))
        y1 = max(0, min(bbox.y1, height - 1))
        x2 = max(x1 + 1, min(bbox.x2, width))
        y2 = max(y1 + 1, min(bbox.y2, height))

        x1, y1, x2, y2 = self._apply_inset(x1, y1, x2, y2)

        crop = bgr_image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Bounding box vazia após recorte: {bbox}")
        return crop

    def _apply_inset(self, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        inset_x = int((x2 - x1) * self._inset_ratio)
        inset_y = int((y2 - y1) * self._inset_ratio)

        inset_x1, inset_y1 = x1 + inset_x, y1 + inset_y
        inset_x2, inset_y2 = x2 - inset_x, y2 - inset_y

        if inset_x2 <= inset_x1 or inset_y2 <= inset_y1:
            return x1, y1, x2, y2  # bbox pequena demais para o inset: usa a caixa inteira

        return inset_x1, inset_y1, inset_x2, inset_y2

    def _classify_pixels(self, hsv: MatLike) -> tuple[NDArray[np.object_], Counter[ColorName]]:
        hue = hsv[:, :, 0].astype(np.int32)
        saturation = hsv[:, :, 1].astype(np.int32)
        value = hsv[:, :, 2].astype(np.int32)

        assigned = np.zeros(hue.shape, dtype=bool)
        labels: NDArray[np.object_] = np.empty(hue.shape, dtype=object)
        for name, mask in _build_classification_rules(hue, saturation, value):
            apply_mask = mask & ~assigned
            labels[apply_mask] = name
            assigned |= apply_mask

        # Rede de segurança: as faixas de matiz acima cobrem 0-179 sem lacunas,
        # então isto não deveria disparar; existe apenas para nunca deixar um
        # pixel sem rótulo.
        labels[~assigned] = ColorName.GRAY

        counts: Counter[ColorName] = Counter(labels.ravel().tolist())
        return labels, counts

    def _mean_rgb(
        self, bgr_image: MatLike, labels: NDArray[np.object_], winner_name: ColorName
    ) -> tuple[int, int, int]:
        winner_pixels = bgr_image[labels == winner_name]
        mean_bgr = winner_pixels.mean(axis=0)
        return (int(round(mean_bgr[2])), int(round(mean_bgr[1])), int(round(mean_bgr[0])))
