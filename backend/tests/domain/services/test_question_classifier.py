import pytest

from app.domain.entities.question_type import QuestionType
from app.domain.services.question_classifier import QuestionClassifier


@pytest.fixture
def classifier() -> QuestionClassifier:
    return QuestionClassifier()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("O que estou vendo?", QuestionType.GENERAL),
        ("Descreva a cena para mim.", QuestionType.GENERAL),
        ("Onde está a mochila?", QuestionType.SPATIAL),
        ("A cadeira está à esquerda ou à direita?", QuestionType.SPATIAL),
        ("Qual a cor da mochila?", QuestionType.COLOR),
        ("A blusa é colorida?", QuestionType.COLOR),
        ("Tem um cachorro na cena?", QuestionType.OBJECT),
        ("Quantas pessoas você vê?", QuestionType.OBJECT),
        ("O que é aquilo no canto?", QuestionType.OBJECT),
        ("Você tem certeza disso?", QuestionType.UNCERTAINTY),
        ("Será que é mesmo uma mesa?", QuestionType.UNCERTAINTY),
    ],
)
def test_classify_returns_expected_type(
    classifier: QuestionClassifier, question: str, expected: QuestionType
) -> None:
    assert classifier.classify(question) == expected


def test_classify_is_case_insensitive(classifier: QuestionClassifier) -> None:
    assert classifier.classify("ONDE ESTÁ A MOCHILA?") == QuestionType.SPATIAL


def test_uncertainty_takes_priority_over_other_keywords(classifier: QuestionClassifier) -> None:
    assert classifier.classify("Você tem certeza que a cor é azul?") == QuestionType.UNCERTAINTY
