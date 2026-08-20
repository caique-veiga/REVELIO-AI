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

from checks import contains_any_keyword, denies_presence
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
    question: dict[str, Any], answer: str, previous_referenced: list[str]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    question_type = question["type"]
    expected_keywords: list[str] = question.get("expected_keywords", [])

    if question_type == "general":
        checks["resposta_correta"] = contains_any_keyword(answer, expected_keywords)
    elif question_type == "color":
        checks["cor"] = contains_any_keyword(answer, expected_keywords)
    elif question_type == "spatial":
        checks["posicao"] = contains_any_keyword(answer, expected_keywords)
    elif question_type == "object_absence":
        denies = denies_presence(answer)
        checks["ausencia_de_objeto"] = denies
        checks["alucinacao"] = not denies

    if question.get("checks_context"):
        # A pergunta de acompanhamento deve se referir ao mesmo objeto sem
        # precisar nomeá-lo de novo — checagem estrutural (via
        # referenced_objects), não julgamento semântico da resposta.
        checks["contexto"] = bool(previous_referenced)

    return checks


def run_scenario(client: RevelioClient, scenario: dict[str, Any]) -> dict[str, Any] | None:
    image_path = REPO_ROOT / scenario["image"]
    if not image_path.is_file():
        print(f"[ERRO] imagem não encontrada: {image_path}", file=sys.stderr)
        return None

    print(f"== {scenario['scenario_id']} — {scenario['name']} ==")
    scene = client.create_scene(image_path)
    conversation_id = scene["conversation_id"]

    question_results: list[dict[str, Any]] = []
    previous_referenced: list[str] = []

    for question in scenario["questions"]:
        started = time.monotonic()
        response = client.ask(conversation_id, question["content"])
        client_latency_ms = (time.monotonic() - started) * 1000

        answer = response["answer"]
        referenced = [obj["class_name"] for obj in response["referenced_objects"]]
        checks = evaluate_question(question, answer, previous_referenced)

        question_results.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["content"],
                "answer": answer,
                "referenced_objects": referenced,
                "client_latency_ms": round(client_latency_ms, 1),
                "checks": checks,
            }
        )
        print(f"  [{question['id']}] {question['content']!r} -> {answer!r}")
        previous_referenced = referenced

    conversation = client.get_conversation(conversation_id)
    answer_messages = [m for m in conversation["messages"] if m["role"] == "assistant"]
    for question_result, message in zip(question_results, answer_messages, strict=True):
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


def summarize_checks(scenario_results: list[dict[str, Any]]) -> dict[str, int]:
    all_checks = [
        value
        for scenario in scenario_results
        for question in scenario["questions"]
        for value in question["checks"].values()
    ]
    return {
        "total": len(all_checks),
        "passed": sum(1 for value in all_checks if value is True),
        "failed": sum(1 for value in all_checks if value is False),
    }


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
    results = {
        "run_started_at": run_started_at.isoformat(),
        "base_url": args.base_url,
        "scenario_count": len(scenario_results),
        "question_count": sum(len(s["questions"]) for s in scenario_results),
        "checks_summary": checks_summary,
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
        f"Checks: {checks_summary['passed']} ok / {checks_summary['failed']} falharam "
        f"/ {checks_summary['total']} total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
