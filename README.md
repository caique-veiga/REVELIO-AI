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
- OpenCV (HSV) para estimativa da cor predominante de cada objeto detectado
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

### Cor predominante (OpenCV)

`OpenCVColorAnalyzer` recebe os bytes da imagem original e um `BoundingBox`, recorta a região,
converte para HSV e classifica a cor predominante em um de 12 buckets (`black`, `white`, `gray`,
`red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple`, `pink`, `brown`) usando limiares de
matiz/saturação/valor. O resultado é **apenas** "a cor predominante da região detectada" — não
tenta localizar semanticamente uma parte do objeto (ex. a camisa de uma pessoa).

Para reduzir a chance do fundo dominar o resultado, o recorte usado na análise descarta uma faixa
das bordas do bbox (`inset_ratio`, padrão 15%) antes de classificar; a cor vencedora é escolhida
por votação de pixels (não pela média simples de toda a região), e o `confidence` retornado é a
fração de pixels do recorte que caiu no bucket vencedor.

**Limitações conhecidas** (herdadas de qualquer classificação de cor por pixel, sem correção de
cena):
- **Iluminação**: sombras fortes ou luz muito quente/fria deslocam a cor percebida (ex. um objeto
  branco sob luz amarela pode ser lido como `orange`/`yellow`).
- **Reflexos**: superfícies brilhantes/metálicas geram destaques quase brancos que competem com a
  cor real do objeto.
- **Fundo**: o mitigador de inset ajuda, mas não elimina o problema — bboxes que abraçam mal o
  objeto (comum em objetos irregulares/finos, como o `knife` observado na validação) ainda
  misturam pixels de fundo.
- **Bounding boxes grandes**: quanto maior a caixa, maior a chance de conter múltiplas cores reais
  (ex. um `couch` com estampa, ou uma pessoa com roupas de cores diferentes) — o resultado é
  sempre "a cor que mais aparece", não "a cor de cada parte".

O classificador (regras de matiz/saturação/valor em `_build_classification_rules`) foi isolado do
resto do pipeline de propósito, para permitir troca futura (ex. clustering, modelo treinado) sem
alterar a interface `ColorAnalyzer`.

Para validar detecção + cor contra imagens reais (gera crops e um `results.json`):

```bash
uv run python scripts/validate_color_analyzer.py caminho/para/diretorio-de-imagens
# saída em <diretorio-de-imagens>/output/
```

### API

`POST /api/v1/scenes` recebe uma imagem (`multipart/form-data`, campo `file`) e executa o fluxo
completo: valida a imagem, salva no filesystem local, roda YOLO, calcula posição e cor de cada
objeto, monta a `Scene` e persiste `Scene`, `Conversation` e os `DetectedObject`s no PostgreSQL —
cada chamada sempre cria uma nova `Scene` e uma nova `Conversation` (nunca reaproveita uma
existente). Retorna:

```json
{
  "scene_id": "...",
  "conversation_id": "...",
  "object_count": 3,
  "status": "created"
}
```

Erros de imagem inválida/não suportada retornam `400`, imagem grande demais retorna `413`; qualquer
outra falha inesperada retorna `500` (logada no servidor, sem vazar detalhes internos na resposta).

```bash
curl -X POST http://localhost:8000/api/v1/scenes -F "file=@caminho/para/imagem.jpg"
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
o domínio; validado com imagens reais via `scripts/validate_yolo.py`. Posição espacial:
`PositionAnalyzer` (domain service, sem dependência externa) — mapeia o centro do bbox em um grid
3x3 (`horizontal`: left/center/right, `vertical`: top/middle/bottom, `region`: combinação das duas,
ex. `front-center`), sem inferir distância, GPS ou profundidade; `Detection.position` é opcional
(preenchido por um passo separado, não pelo próprio detector). Cor predominante: `ColorAnalyzer`
(protocol) e `OpenCVColorAnalyzer` — recorta o bbox, classifica em HSV (12 cores) e retorna
`ColorResult` (`name`, `rgb`, `confidence`); validado com imagens reais via
`scripts/validate_color_analyzer.py`. Scene JSON: `SceneBuilder` (domain service) monta um `Scene`
(`scene_id`, `conversation_id`, `image`, `model`, `objects`) a partir de detecções já enriquecidas
com posição e cor — valida que cada `Detection` tem `position`/`color` antes de montar a cena, mas
não chama `PositionAnalyzer`/`ColorAnalyzer` ele mesmo (isso é papel do futuro `SceneService`).
`SceneSchema` (Pydantic, em `api/schemas/`) serializa exatamente o formato descrito na seção 11 do
contexto, incluindo a chave `class` (reservada em Python — mapeada via `Field(alias="class")`).
Pipeline completo (`ObjectDetector` mockado + `PositionAnalyzer`/`OpenCVColorAnalyzer` reais +
`SceneBuilder` + `SceneSchema`) coberto por testes de integração. API: `SceneService`
(`application/services/`) orquestra o fluxo ponta a ponta (imagem → YOLO → posição → cor →
`SceneBuilder` → PostgreSQL) e `POST /api/v1/scenes` (`SceneController`) o expõe via HTTP, com
injeção de dependências em `api/dependencies.py` e tratamento de erros mapeando exceções de domínio
para status HTTP. `GET /api/v1/scenes/{scene_id}`, conversas/mensagens, Ollama e o aplicativo
Android ainda não foram implementados.
