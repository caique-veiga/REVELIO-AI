# Avaliação do Revelio AI

Benchmark simples e reproduzível do assistente ponta a ponta: imagem real →
`POST /api/v1/scenes` → perguntas reais → `POST
/api/v1/conversations/{id}/messages` → checagens automáticas simples →
resultado em JSON.

Não é um "LLM judge" nem uma métrica sofisticada — são checagens de
palavra-chave, documentadas como heurísticas em `checks.py`. A leitura
humana dos arquivos gerados em `results/` continua sendo necessária para
avaliar a qualidade real das respostas.

## Como funciona

```
evaluation/
    scenarios/          # cenários: imagem + perguntas + palavras-chave esperadas
    checks.py           # checagens simples (palavra-chave, negação)
    client.py           # cliente HTTP fino para a API real (sem mocks)
    run_evaluation.py   # runner: executa todos os cenários e grava o resultado
    results/            # saída em JSON (gerado a cada execução, não versionado)
```

Cada cenário (`evaluation/scenarios/*.json`) descreve:

- `image`: caminho da imagem real (relativo à raiz do projeto);
- `known_objects`: objetos que sabemos que o YOLO detecta nessa imagem
  (referência, vindo de execuções reais anteriores — `validation_yolo_tests/output/results.json`);
- `questions`: lista de perguntas, cada uma com um `type`
  (`general` | `spatial` | `color` | `object_absence`) e, quando aplicável,
  `expected_keywords` (palavras aceitáveis na resposta).

O runner cria uma `Scene`/`Conversation` de verdade para cada imagem, faz
cada pergunta na ordem (mantendo o histórico da conversa, igual um app real
faria), e registra por pergunta:

- `model`, `prompt_version`, `latency_ms`, `timestamp` — vindos de verdade
  do banco (via `GET /conversations/{id}`), não inventados pelo script;
- `answer`, `referenced_objects` — a resposta real do Qwen;
- `checks` — o resultado das checagens simples (ver abaixo).

## As seis dimensões de checagem (tarefa 4)

| Dimensão            | Como é checada                                                              |
|---------------------|------------------------------------------------------------------------------|
| resposta correta    | pergunta `general`: alguma `expected_keyword` aparece na resposta            |
| cor                 | pergunta `color`: a cor esperada aparece na resposta                         |
| posição             | pergunta `spatial`: o termo de posição esperado aparece na resposta          |
| ausência de objeto  | pergunta `object_absence`: a resposta nega a presença (palavra de negação)   |
| alucinação          | pergunta `object_absence`: o inverso da checagem acima (afirmou algo ausente)|
| contexto            | pergunta marcada com `"checks_context": true`: a pergunta seguinte referenciou o mesmo objeto sem precisar nomeá-lo de novo (checagem estrutural via `referenced_objects`, não semântica) |

Todas são heurísticas simples de texto — podem dar falso positivo/negativo
(ex. o Qwen responder em inglês, ou usar um sinônimo fora da lista de
`expected_keywords`). O objetivo é dar um primeiro sinal automático rápido,
não substituir a leitura humana do resultado.

## Como executar

Pré-requisitos (tudo real, nada mockado):

1. Servidor da aplicação rodando: `uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`
2. PostgreSQL rodando (`docker compose -f docker/docker-compose.yml up -d`) com as migrations aplicadas.
3. Ollama com o modelo configurado (`OLLAMA_MODEL`) rodando e acessível em `OLLAMA_BASE_URL`.

Depois:

```bash
uv run python evaluation/run_evaluation.py
```

Por padrão aponta para `http://127.0.0.1:8000`; para apontar para outro
host: `--base-url http://outro-host:8000`. Para escolher onde salvar o
resultado: `--output caminho/personalizado.json`.

O resultado fica em `evaluation/results/<timestamp>.json` e um resumo é
impresso no terminal (quantas checagens passaram/falharam).

## Reprodutibilidade

Cada execução cria cenas e conversas novas (o mesmo comportamento da API
real — cada foto sempre gera uma nova `Scene`/`Conversation`, nunca
reaproveita uma existente), então rodar de novo não deixa estado sujo nem
depende de rodadas anteriores. A única fonte de variação entre execuções é
a própria VLM (Qwen), que pode responder de forma um pouco diferente a cada
vez — por isso as checagens são por palavra-chave (tolerantes a variação de
fraseado), não por igualdade exata de texto.

## Cenários disponíveis

Reaproveitam as imagens reais já usadas em `validation_yolo_tests/` (mesmas
usadas para validar o YOLO e o `OpenCVColorAnalyzer` em etapas anteriores):

- `living_room` — sofá, planta, vaso, TV.
- `desk` — notebook, celular.
- `kitchen_cutting_board` — bananas, faca, tigela.
- `pet_dog` — cachorro (cenário de contexto/follow-up: "que animal é esse?" → "qual a cor **dele**?").

Cobrem parte das categorias descritas no `CLAUDE_CONTEXT.md` §18 (S02
Kitchen, S03 Living Room, S05 Person parcialmente coberto por um animal em
vez de pessoa, S06 Desk). Não cobrem ainda, por falta de imagem real
disponível: cena de rua (S04), ambiguidade proposital (S07), pessoa humana
(S05 completo). Se quiser ampliar a cobertura, me passe (ou peça que eu
peça) imagens reais adicionais — este projeto não gera/sintetiza imagens
para os cenários.
