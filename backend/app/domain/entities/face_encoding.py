from dataclasses import dataclass

import numpy as np

from app.domain.entities.bounding_box import BoundingBox


@dataclass(frozen=True, eq=False)
class FaceEncoding:
    """Um rosto detectado numa imagem: onde está (bbox) e seu embedding.

    `eq=False` porque a comparação de dois `np.ndarray` com `==` retorna um
    array, não um bool — o dataclass eq automático quebraria com isso.
    """

    bbox: BoundingBox
    embedding: np.ndarray
