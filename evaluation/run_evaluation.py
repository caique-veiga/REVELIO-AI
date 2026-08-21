#!/usr/bin/env python3
"""Runner do benchmark de avaliação do Revelio AI.

Para cada cenário em evaluation/scenarios/*.json:
    imagem -> POST /api/v1/scenes (cria Scene + Conversation de verdade)
        -> para cada pergunta: POST /api/v1/conversations/{id}/messages
        -> GET /api/v1/conversations/{id} (recupera model/prompt_version/latency/timestamp
           realmente persistidos)
        -> aplica checagens simples (evaluation/checks.py)
    -> grava tudo em evaluation/results/<timestamp>.json

Não mocka nada: precisa do servidor (uvicorn), do PostgreSQL e do Ollama
com o modelo configurado rodando de verdade. Usa só a API HTTP pública da
aplicação — não importa nada de backend/app, então não altera (nem
depende dos detalhes internos d)o pipeline principal.

Uso:
    uv run python evaluation/run_evaluation.py
    uv run python evaluation/run_evaluation.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from checks import AnswerState, classify_answer_state, contains_any_keyword, mentions_object
from client import RevelioClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_scenarios() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(SCENARIOS_DIR.glob("*.json"))
    ]


def evaluate_question(
    question: dict[str, Any], answer: str | None, error_kind: str | None, previous_answer: str
) -> tuple[AnswerState, dict[str, bool | None]]:
    """Retorna o AnswerState da resposta e as checagens booleanas do tipo.

    Um valor `None` numa checagem significa "não aplicável neste estado"
    (ex.: resposta vazia ou erro de infraestrutura) — bem diferente de
    `False` (resposta chegou, mas errou). Ver AnswerState: uma resposta
    vazia nunca deve virar "alucinação" (achado da ETAPA 13.1).
    """
    question_type = question["type"]
    expected_keywords: list[str] = question.get("expected_keywords", [])
    expects_presence_check = question_type == "object_absence"

    state = classify_answer_state(answer, error_kind, expects_presence_check=expects_presence_check)
    checks: dict[str, bool | None] = {}

    if state in (AnswerState.EMPTY_RESPONSE, AnswerState.ERROR, AnswerState.TIMEOUT):
        if question_type == "general":
            checks["resposta_correta"] = None
        elif question_type == "color":
            checks["cor"] = None
        elif question_type == "spatial":
            checks["posicao"] = None
        elif question_type == "object_absence":
            checks["ausencia_de_objeto"] = None
            checks["alucinacao"] = None
    else:
        assert answer is not None
        if question_type == "general":
            checks["resposta_correta"] = contains_any_keyword(answer, expected_keywords)
        elif question_type == "color":
            checks["cor"] = contains_any_keyword(answer, expected_keywords)
        elif question_type == "spatial":
            checks["posicao"] = contains_any_keyword(answer, expected_keywords)
        elif question_type == "object_absence":
            checks["ausencia_de_objeto"] = state is AnswerState.NEGATED
            checks["alucinacao"] = state is AnswerState.AFFIRMED

    if question.get("checks_context"):
        # Checagem própria do benchmark (não usa o `referenced_objects` do
        # backend, que faz substring ingênuo em inglês e falha para
        # respostas em português — ex. "dog" não aparece em "cachorro",
        # achado da ETAPA 13.1). Usa CANONICAL_LABELS para decidir se a
        # resposta ANTERIOR mencionou o objeto relevante.
        context_object = question.get("context_object")
        if context_object and state not in (AnswerState.ERROR, AnswerState.TIMEOUT):
            checks["contexto"] = mentions_object(previous_answer, context_object)
        else:
            checks["contexto"] = None

    return state, checks


def run_scenario(client: RevelioClient, scenario: dict[str, Any]) -> dict[str, Any] | None:
    image_path = REPO_ROOT / scenario["image"]
    if not image_path.is_file():
        print(f"[ERRO] imagem não encontrada: {image_path}", file=sys.stderr)
        return None

    print(f"== {scenario['scenario_id']} — {scenario['name']} ==")
    scene = client.create_scene(image_path)
    conversation_id = scene["conversation_id"]

    question_results: list[dict[str, Any]] = []
    previous_answer = ""

    for question in scenario["questions"]:
        started = time.monotonic()
        response = client.ask(conversation_id, question["content"])
        client_latency_ms = (time.monotonic() - started) * 1000

        error = response.get("error")
        answer = response.get("answer") if error is None else None
        referenced = (
            [obj["class_name"] for obj in response.get("referenced_objects", [])]
            if error is None
            else []
        )

        state, checks = evaluate_question(
            question, answer, error["kind"] if error else None, previous_answer
        )

        question_results.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["content"],
                "answer": answer,
                "answer_state": state.value,
                "error": error,
                "referenced_objects": referenced,
                "client_latency_ms": round(client_latency_ms, 1),
                "checks": checks,
            }
        )
        print(f"  [{question['id']}] {question['content']!r} -> state={state.value} {answer!r}")
        previous_answer = answer or ""

    conversation = client.get_conversation(conversation_id)
    answer_messages = [m for m in conversation["messages"] if m["role"] == "assistant"]

    # Sem retry: com o bug de transação da ETAPA 13.1 corrigido em
    # backend/app, este GET já deve enxergar todas as mensagens do
    # assistente imediatamente. Se não enxergar, é sinal (não mascarado)
    # de que algo regrediu — por isso o aviso explícito, não um retry.
    if len(answer_messages) < len([q for q in question_results if q["error"] is None]):
        print(
            f"  [AVISO] {scenario['scenario_id']}: só {len(answer_messages)} mensagens do "
            "assistente visíveis no GET — possível regressão da correção de transação "
            "da ETAPA 13.1.",
            file=sys.stderr,
        )

    for question_result, message in zip(
        (q for q in question_results if q["error"] is None), answer_messages, strict=False
    ):
        # .get() propositalmente defensivo: este script fala só com a API
        # pública (nenhum acesso direto ao banco), então não deve quebrar se
        # algum campo não estiver exposto no schema de resposta.
        question_result["model"] = message.get("model_name")
        question_result["prompt_version"] = message.get("prompt_version")
        question_result["latency_ms"] = message.get("latency_ms")
        question_result["timestamp"] = message.get("created_at")

    return {
        "scenario_id": scenario["scenario_id"],
        "name": scenario["name"],
        "image": scenario["image"],
        "scene_id": scene["scene_id"],
        "conversation_id": conversation_id,
        "object_count": scene["object_count"],
        "questions": question_results,
    }


# Nem toda checagem é "True = bom": `alucinacao` é o oposto — True significa
# que o modelo afirmou algo que não deveria. Somar tudo cru (True=ok,
# False=falha) sem considerar isso conta "não houve alucinação" (False,
# correto) como uma falha — foi exatamente esse artefato de contagem, não um
# erro real do modelo, que inflava o placar da ETAPA 13 (ver ETAPA 13.1, achado
# do Part 16). Corrigido aqui, não nos dados/expected_keywords dos cenários.
_INVERTED_CHECKS = {"alucinacao"}


def summarize_checks(scenario_results: list[dict[str, Any]]) -> dict[str, int]:
    named_checks = [
        (name, value)
        for scenario in scenario_results
        for question in scenario["questions"]
        for name, value in question["checks"].items()
    ]
    passed = sum(
        1
        for name, value in named_checks
        if value is not None and value != (name in _INVERTED_CHECKS)
    )
    failed = sum(
        1
        for name, value in named_checks
        if value is not None and value == (name in _INVERTED_CHECKS)
    )
    not_applicable = sum(1 for _, value in named_checks if value is None)
    return {
        "total": len(named_checks),
        "passed": passed,
        "failed": failed,
        "not_applicable": not_applicable,
    }


def summarize_answer_states(scenario_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scenario in scenario_results:
        for question in scenario["questions"]:
            state = question["answer_state"]
            counts[state] = counts.get(state, 0) + 1
    return counts


def run() -> int:
    parser = argparse.ArgumentParser(description="Roda o benchmark de avaliação do Revelio AI.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default=None, help="Caminho do JSON de saída (opcional).")
    args = parser.parse_args()

    scenarios = load_scenarios()
    if not scenarios:
        print("Nenhum cenário encontrado em evaluation/scenarios/.", file=sys.stderr)
        return 1

    client = RevelioClient(base_url=args.base_url)
    run_started_at = datetime.now(UTC)
    scenario_results = []
    try:
        for scenario in scenarios:
            result = run_scenario(client, scenario)
            if result is not None:
                scenario_results.append(result)
    finally:
        client.close()

    checks_summary = summarize_checks(scenario_results)
    answer_states_summary = summarize_answer_states(scenario_results)
    results = {
        "run_started_at": run_started_at.isoformat(),
        "base_url": args.base_url,
        "scenario_count": len(scenario_results),
        "question_count": sum(len(s["questions"]) for s in scenario_results),
        "checks_summary": checks_summary,
        "answer_states_summary": answer_states_summary,
        "scenarios": scenario_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"{run_started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResultados: {output_path}")
    print(
        f"Checks: {checks_summary['passed']} ok / {checks_summary['failed']} falharam / "
        f"{checks_summary['not_applicable']} não aplicável / {checks_summary['total']} total"
    )
    print(f"Estados de resposta: {answer_states_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
