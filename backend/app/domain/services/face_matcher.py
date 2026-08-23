from collections.abc import Iterable

import numpy as np

from app.domain.entities.person_match import PersonCandidate, PersonMatch


class FaceMatcher:
    """Compara um embedding de rosto contra candidatos cadastrados.

    Métrica: similaridade de cosseno — é a métrica correta para os
    embeddings do InsightFace (diferente do dlib/face_recognition, que usa
    distância euclidiana). `threshold` é um score de similaridade
    (1.0 = idêntico), não uma distância.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def find_best_match(
        self,
        embedding: np.ndarray,
        candidates: Iterable[tuple[PersonCandidate, np.ndarray]],
    ) -> PersonMatch | None:
        best: PersonMatch | None = None
        for candidate, candidate_embedding in candidates:
            similarity = self._cosine_similarity(embedding, candidate_embedding)
            if similarity < self._threshold:
                continue
            if best is None or similarity > best.similarity:
                best = PersonMatch(
                    person_id=candidate.person_id,
                    name=candidate.name,
                    relationship=candidate.relationship,
                    similarity=similarity,
                )
        return best

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)
