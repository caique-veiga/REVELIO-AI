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

- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, PostgreSQL
- Pillow para validação/inspeção de imagens; armazenamento local em `data/images/`
- YOLO (Ultralytics, pré-treinado em COCO) para detecção de objetos, rodando em CPU
- OpenCV para estimativa de cor (ainda não integrado nesta etapa)
- Ollama + Qwen 3.5 4B para a VLM (ainda não integrado nesta etapa)
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

### Detecção de objetos (YOLO)

O modelo é configurável via `YOLO_MODEL` (padrão: `yolov8n.pt`, a variante *nano* do YOLOv8 —
pequena o bastante para rodar em CPU) e o limiar de confiança via `YOLO_CONFIDENCE_THRESHOLD`
(padrão: `0.5`). O modelo **não é treinado** nesta versão — apenas inferência com pesos
pré-treinados no COCO (80 classes).

Ao instanciar `YOLOObjectDetector`, a biblioteca `ultralytics` baixa automaticamente os pesos
correspondentes (ex.: `yolov8n.pt`) do repositório oficial na primeira execução, cacheando o
arquivo localmente (por padrão, no diretório de trabalho atual). Não é necessário baixar nada
manualmente para desenvolvimento — só é preciso acesso à internet na primeira execução. Para usar
pesos já baixados (ou um modelo próprio), aponte `YOLO_MODEL` para o caminho do arquivo `.pt`.

Para testar a detecção contra uma imagem real:

```bash
uv run python scripts/test_detection.py caminho/para/imagem.jpg
```

### Testes e qualidade

```bash
uv run pytest
uv run ruff check .
uv run mypy backend/app
```

Os testes de repository rodam contra um SQLite em memória (não é necessário Postgres para
`uv run pytest`); o Postgres real via Docker é usado apenas para rodar a aplicação e as migrations.

## Status atual

Fundação do backend (estrutura modular, configuração tipada, `GET /health`), persistência
PostgreSQL: models SQLAlchemy 2.x (`User`, `Device`, `Conversation`, `Scene`, `DetectedObject`,
`Message`), migration inicial via Alembic e repositories (`SceneRepository`,
`ConversationRepository`, `MessageRepository`, `ObjectRepository`) testados. Armazenamento local de
imagens: `ImageStorage` (protocol) e `LocalImageStorage` — organiza os arquivos por data
(`data/images/AAAA/MM/DD/{scene_id}.jpg`), valida extensão/MIME/tamanho máximo/integridade e
calcula SHA-256; a imagem nunca é salva como BYTEA no banco. Detecção de objetos: `ObjectDetector`
(protocol) e `YOLOObjectDetector` — recebe bytes de imagem e retorna uma lista de `Detection`
(`object_id`, `class_id`, `class_name`, `confidence`, `bbox`), sem vazar tipos da Ultralytics para
o domínio. Posição, cor, Scene Builder, Ollama e o aplicativo Android ainda não foram
implementados.
