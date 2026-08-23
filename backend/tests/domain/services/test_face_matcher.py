import uuid

import numpy as np

from app.domain.entities.person_match import PersonCandidate
from app.domain.services.face_matcher import FaceMatcher


def _candidate(name: str = "Maria", relationship: str = "minha irmã") -> PersonCandidate:
    return PersonCandidate(person_id=uuid.uuid4(), name=name, relationship=relationship)


def test_find_best_match_returns_match_above_threshold() -> None:
    matcher = FaceMatcher(threshold=0.5)
    embedding = np.array([1.0, 0.0, 0.0])
    candidate = _candidate()

    match = matcher.find_best_match(embedding, [(candidate, np.array([1.0, 0.0, 0.0]))])

    assert match is not None
    assert match.name == "Maria"
    assert match.relationship == "minha irmã"
    assert match.similarity > 0.99


def test_find_best_match_returns_none_below_threshold() -> None:
    matcher = FaceMatcher(threshold=0.5)
    embedding = np.array([1.0, 0.0, 0.0])
    candidate = _candidate()

    match = matcher.find_best_match(embedding, [(candidate, np.array([0.0, 1.0, 0.0]))])

    assert match is None


def test_find_best_match_returns_none_for_no_candidates() -> None:
    matcher = FaceMatcher(threshold=0.5)
    match = matcher.find_best_match(np.array([1.0, 0.0, 0.0]), [])
    assert match is None


def test_find_best_match_returns_highest_similarity_among_multiple() -> None:
    matcher = FaceMatcher(threshold=0.5)
    embedding = np.array([1.0, 0.0])
    close_candidate = _candidate(name="Maria")
    exact_candidate = _candidate(name="João", relationship="do trabalho")

    match = matcher.find_best_match(
        embedding,
        [
            (close_candidate, np.array([0.9, 0.1])),
            (exact_candidate, np.array([1.0, 0.0])),
        ],
    )

    assert match is not None
    assert match.name == "João"


def test_find_best_match_handles_zero_vector_without_crashing() -> None:
    matcher = FaceMatcher(threshold=0.5)
    match = matcher.find_best_match(np.array([0.0, 0.0]), [(_candidate(), np.array([0.0, 0.0]))])
    assert match is None
