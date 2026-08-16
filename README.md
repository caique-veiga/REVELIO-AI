# Revelio AI

Assistente visual para pessoas cegas ou com baixa visão — protótipo desenvolvido como projeto de
pós-graduação em IA Generativa e LLMs.

## Objetivo

Um aplicativo Android captura uma fotografia e a envia para o backend. O backend detecta objetos
na cena (YOLO treinado em COCO), estima a cor predominante de cada objeto (OpenCV), monta um
Scene JSON estruturado e persiste tudo no PostgreSQL. O usuário então faz perguntas sobre a cena;
cada pergunta, junto da imagem, do Scene JSON e do histórico da conversa, é enviada a uma VLM
(Qwen 3.5 4B via Ollama) e a resposta é lida em voz alta pelo aplicativo Android via Text-to-Speech.

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

- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic
- PostgreSQL (ainda não integrado nesta etapa)
- YOLO (COCO) + OpenCV para visão computacional (ainda não integrado nesta etapa)
- Ollama + Qwen 3.5 4B para a VLM (ainda não integrado nesta etapa)
- pytest, Ruff, mypy

Gerenciamento de dependências e ambiente virtual com [uv](https://docs.astral.sh/uv/).

## Como executar

Instale as dependências e crie o ambiente virtual:

```bash
uv sync
```

Suba o servidor de desenvolvimento:

```bash
uv run uvicorn app.main:app --app-dir backend --reload
```

Verifique se está no ar:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

Copie `.env.example` para `.env` e ajuste os valores conforme seu ambiente:

```bash
cp .env.example .env
```

### Testes e qualidade

```bash
uv run pytest
uv run ruff check .
uv run mypy backend/app
```

## Status atual

Fundação do backend: estrutura modular, configuração tipada por environment variables e endpoint
`GET /health`. YOLO, PostgreSQL, Ollama, armazenamento de imagens e o aplicativo Android ainda não
foram implementados.
