# Revelio AI

Assistente visual para pessoas cegas ou com baixa visão — protótipo desenvolvido como projeto de
pós-graduação em IA Generativa e LLMs.

## Objetivo

Um aplicativo Android captura uma fotografia e a envia para o backend, que salva a imagem e cria
uma nova `Conversation`. O usuário então faz perguntas sobre a cena; cada pergunta, junto da
imagem e do histórico da conversa, é enviada diretamente a uma VLM multimodal — o modelo analisa a
imagem e decide sozinho como responder: com texto direto, ou chamando uma *tool*
(`register_person`/`identify_persons`) para cadastrar ou reconhecer pessoas por reconhecimento
facial. A resposta é lida em voz alta pelo aplicativo Android via Text-to-Speech.

Pipeline unificado: Ollama (Qwen, local) é tentado primeiro quando habilitado, com fallback
automático para o Gemini Flash-Lite (API) em caso de falha — ambos recebem exatamente a mesma
coisa (imagem + histórico + pergunta + tools), sem nenhum pré-processamento estruturado (JSON de
detecção de objetos) entre a imagem e o modelo.

Cada nova fotografia inicia uma nova conversa — o histórico de uma cena anterior nunca é usado na
cena seguinte.

Esta é a primeira versão do projeto. Funcionalidades como detecção de perigos, depth estimation,
segmentação, OCR, RAG ou microsserviços estão fora de escopo por enquanto — veja
[CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) para o contexto arquitetural completo.

## Arquitetura

O backend é um **modular monolith** em FastAPI, organizado em camadas:

```
Controller -> Application Service -> Domain (interfaces) -> Infrastructure (implementações)
```

```
backend/app/
    api/             # controllers (rotas) e schemas (Pydantic)
    application/      # services de orquestração
    domain/          # entidades, modelos de domínio e protocols (interfaces)
    infrastructure/  # implementações concretas: database, repositories, storage, vision, vlm
    config/          # configuração tipada (Settings)
```

O backend e a máquina com GPU (que roda o Ollama) são hosts diferentes, conectados por uma rede
privada/VPN — o endereço do Ollama é sempre configurável via `OLLAMA_BASE_URL`, nunca hardcoded.

## Stack

- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, PostgreSQL
- Pillow para validação/inspeção de imagens; armazenamento local em `data/images/`
- Ollama + Qwen (tier 1, local) com fallback automático para Gemini Flash-Lite (tier 2, API) — os
  dois suportam tool calling (`register_person`/`identify_persons`)
- InsightFace (modelo `buffalo_l`, CPU/ONNX) para detecção facial + embedding, usado no
  reconhecimento de pessoas
- pytest, Ruff, mypy

Gerenciamento de dependências e ambiente virtual com [uv](https://docs.astral.sh/uv/).

## Como executar

Instale as dependências e crie o ambiente virtual:

```bash
uv sync
```

Copie `.env.example` para `.env` e ajuste os valores conforme seu ambiente:

```bash
cp .env.example .env
```

### Banco de dados

Suba um PostgreSQL de desenvolvimento via Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Aplique as migrations:

```bash
uv run alembic upgrade head
```

Para criar uma nova migration depois de alterar os models (`backend/app/infrastructure/database/models.py`):

```bash
uv run alembic revision --autogenerate -m "descrição da mudança"
```

### Servidor

Suba o servidor de desenvolvimento:

```bash
uv run uvicorn app.main:app --app-dir backend --reload
```

Verifique se está no ar:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### VLM: Ollama + Gemini (pipeline unificado)

`ConversationService` chama sempre o mesmo método — `VisionLanguageModel.ask(image, system_prompt,
conversation_history, question, tools)` — não importa se quem responde é o Ollama ou o Gemini. Não
há Scene JSON nem qualquer outro pré-processamento estruturado: a VLM recebe a imagem crua e decide
sozinha se responde com texto ou chama uma tool.

- `OLLAMA_ENABLED=true` (padrão): tenta o Ollama/Qwen primeiro (`OLLAMA_BASE_URL`,
  `OLLAMA_MODEL`); se falhar (indisponível, timeout, resposta vazia, tool call malformado), cai
  automaticamente para o Gemini.
- `OLLAMA_ENABLED=false`: usa só o Gemini (`GEMINI_API_KEY` obrigatório).

O suporte a tool calling do Ollama depende do modelo configurado — não há garantia de que todo
modelo vision local faça function calling de forma tão confiável quanto o Gemini; qualquer
comportamento inesperado vira erro e aciona o fallback automaticamente.

### Reconhecimento de pessoas

Duas tools ficam sempre disponíveis para a VLM: `register_person` (cadastra uma pessoa a partir de
uma foto com exatamente um rosto) e `identify_persons` (compara os rostos da cena atual contra as
pessoas já cadastradas do usuário). `InsightFaceEncoder` faz detecção + embedding em uma única
chamada; a similaridade usa distância de cosseno (`FaceMatcher`, limiar configurável via
`FACE_MATCH_THRESHOLD`). Não há filtro por classe de objeto antes do reconhecimento facial — decidir
se há um rosto é responsabilidade exclusiva do `FaceEncoder`.

### API

`POST /api/v1/scenes` recebe uma imagem (`multipart/form-data`, campo `file`), valida, salva no
filesystem local e persiste `Scene` + `Conversation` no PostgreSQL — cada chamada sempre cria uma
nova `Scene` e uma nova `Conversation` (nunca reaproveita uma existente). Retorna:

```json
{
  "scene_id": "...",
  "conversation_id": "...",
  "status": "created"
}
```

Erros de imagem inválida/não suportada retornam `400`, imagem grande demais retorna `413`; qualquer
outra falha inesperada retorna `500` (logada no servidor, sem vazar detalhes internos na resposta).

```bash
curl -X POST http://localhost:8000/api/v1/scenes -F "file=@caminho/para/imagem.jpg"
```

`POST /api/v1/conversations/{conversation_id}/messages` envia a pergunta à VLM ativa (Ollama com
fallback para Gemini, ou só Gemini) junto da imagem e do histórico, e retorna a resposta:

```json
{
  "answer": "...",
  "scene_id": "..."
}
```

Como ainda não há autenticação/gestão de usuários, o serviço reutiliza um único usuário padrão
(criado automaticamente na primeira cena) como dono provisório de todas as conversations.

### Testes e qualidade

```bash
uv run pytest
uv run ruff check .
uv run mypy backend/app
```

Os testes de repository rodam contra um SQLite em memória (não é necessário Postgres para
`uv run pytest`); o Postgres real via Docker é usado apenas para rodar a aplicação e as migrations.
