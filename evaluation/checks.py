"""Checagens simples, baseadas em palavra-chave, para o benchmark de avaliação.

Nenhuma delas é um "LLM judge" nem análise semântica real (por instrução
explícita da etapa) — são heurísticas de texto, documentadas como tal.
Servem para dar um primeiro sinal automático; a leitura humana dos
resultados em evaluation/results/ continua sendo necessária.
"""

import unicodedata
from enum import Enum

_NEGATION_WORDS = ("nao", "nenhum", "nenhuma", "nada de", "sem ")

# Rótulos canônicos com sinônimos PT/EN, usados só pelo benchmark (não pelo
# pipeline principal) para não depender do casamento ingênuo de substring do
# backend (`class_name` em inglês vs. resposta em português — ver ETAPA 13.1,
# achado do checker de contexto do pet_dog: "dog" não aparece em "cachorro").
CANONICAL_LABELS: dict[str, list[str]] = {
    "dog": ["dog", "cachorro", "cao", "cachorrinho"],
    "cat": ["cat", "gato"],
    "banana": ["banana", "bananas"],
    "couch": ["couch", "sofa"],
    "tv": ["tv", "televisao"],
    "knife": ["knife", "faca"],
    "bowl": ["bowl", "tigela", "bacia"],
    "potted plant": ["potted plant", "planta", "vaso de planta"],
    "vase": ["vase", "vaso"],
    "laptop": ["laptop", "notebook"],
    "cell phone": ["cell phone", "celular", "telefone"],
    "bicycle": ["bicycle", "bicicleta"],
}


class AnswerState(Enum):
    """Estado da resposta de uma pergunta, para o benchmark de avaliação.

    Existe para não confundir "resposta vazia" com "alucinação" (achado da
    ETAPA 13.1: o checker antigo classificava content=="" como alucinação
    em perguntas do tipo object_absence, o que está errado — vazio não é
    afirmação nem negação, é ausência de resposta).
    """

    ANSWERED = "answered"
    NEGATED = "negated"
    AFFIRMED = "affirmed"
    EMPTY_RESPONSE = "empty_response"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNEXPECTED = "unexpected"


def _normalize(text: str) -> str:
    text = text.lower()
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def contains_any_keyword(answer: str, keywords: list[str]) -> bool:
    """True se algum dos `keywords` aparece na resposta (sem acento, sem caixa)."""
    normalized_answer = _normalize(answer)
    return any(_normalize(keyword) in normalized_answer for keyword in keywords)


def denies_presence(answer: str) -> bool:
    """True se a resposta parece negar a presença de algo perguntado."""
    return contains_any_keyword(answer, list(_NEGATION_WORDS))


def mentions_object(text: str, canonical_class_name: str) -> bool:
    """True se `text` menciona o objeto, em PT ou EN, via CANONICAL_LABELS.

    Fallback: se `canonical_class_name` não estiver no dicionário, cai para
    checar o próprio nome da classe (comportamento equivalente ao substring
    ingênuo do backend, só que aqui é uma decisão explícita do benchmark).
    """
    aliases = CANONICAL_LABELS.get(canonical_class_name.lower(), [canonical_class_name])
    return contains_any_keyword(text, aliases)


def classify_answer_state(
    answer: str | None,
    error_kind: str | None = None,
    *,
    expects_presence_check: bool = False,
) -> AnswerState:
    """Classifica o resultado de uma pergunta num estado explícito.

    `error_kind` vem do client (`"timeout"`, `"http_error"` etc.) quando a
    chamada HTTP falhou antes de haver qualquer `answer`. `expects_presence_check`
    só deve ser True para perguntas do tipo object_absence — é aí que faz
    sentido distinguir NEGATED de AFFIRMED.
    """
    if error_kind == "timeout":
        return AnswerState.TIMEOUT
    if error_kind is not None:
        return AnswerState.ERROR
    if answer is None:
        return AnswerState.UNEXPECTED
    if not answer.strip():
        return AnswerState.EMPTY_RESPONSE
    if expects_presence_check:
        return AnswerState.NEGATED if denies_presence(answer) else AnswerState.AFFIRMED
    return AnswerState.ANSWERED
