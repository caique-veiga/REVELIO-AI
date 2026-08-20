from app.domain.entities.question_type import QuestionType

_UNCERTAINTY_KEYWORDS = ("certeza", "você acha", "sera que", "será que", "tem como saber")
_SPATIAL_KEYWORDS = (
    "onde",
    "posição",
    "esquerda",
    "direita",
    "em cima",
    "embaixo",
    "atrás",
    "na frente",
    "ao lado",
)
_COLOR_KEYWORDS = ("cor ", "cor?", "colorido", "colorida")
_OBJECT_KEYWORDS = (
    "o que é",
    "que objeto",
    "quais objetos",
    "quantos",
    "quantas",
    "tem um",
    "tem uma",
    "existe um",
    "existe uma",
    "isso",
    "aquilo",
)


class QuestionClassifier:
    """Classifica uma pergunta em um `QuestionType`, por palavras-chave.

    Heurística simples e deliberadamente limitada (sem NLP real) — cobre os
    padrões de pergunta descritos no CLAUDE_CONTEXT.md (posição, cor,
    presença/identidade de objetos, incerteza), com "general" como fallback
    para descrições amplas ("o que estou vendo?").
    """

    def classify(self, question: str) -> QuestionType:
        text = question.strip().lower()

        if any(keyword in text for keyword in _UNCERTAINTY_KEYWORDS):
            return QuestionType.UNCERTAINTY
        if any(keyword in text for keyword in _SPATIAL_KEYWORDS):
            return QuestionType.SPATIAL
        if any(keyword in text for keyword in _COLOR_KEYWORDS):
            return QuestionType.COLOR
        if any(keyword in text for keyword in _OBJECT_KEYWORDS):
            return QuestionType.OBJECT
        return QuestionType.GENERAL
