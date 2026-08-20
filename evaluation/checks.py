"""Checagens simples, baseadas em palavra-chave, para o benchmark de avaliação.

Nenhuma delas é um "LLM judge" nem análise semântica real (por instrução
explícita da etapa) — são heurísticas de texto, documentadas como tal.
Servem para dar um primeiro sinal automático; a leitura humana dos
resultados em evaluation/results/ continua sendo necessária.
"""

import unicodedata

_NEGATION_WORDS = ("nao", "nenhum", "nenhuma", "nada de", "sem ")


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
