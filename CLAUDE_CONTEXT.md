# PROJETO — ASSISTENTE VISUAL PARA PESSOAS CEGAS

## 1. OBJETIVO

Estamos desenvolvendo um projeto de pós-graduação em IA Generativa e LLMs.

O objetivo é construir um protótipo funcional de um assistente visual para pessoas cegas ou com baixa visão.

O sistema recebe uma fotografia capturada por um aplicativo Android e:

1. armazena a imagem no computador que possui a GPU;
2. utiliza um detector de objetos YOLO pré-treinado no dataset COCO;
3. extrai:
   - classe do objeto;
   - confiança;
   - bounding box;
   - posição aproximada na imagem;
4. utiliza OpenCV para estimar a cor predominante dos objetos;
5. constrói um Scene JSON estruturado;
6. persiste os dados no PostgreSQL;
7. permite ao usuário fazer perguntas sobre a cena;
8. envia para uma VLM através do Ollama:
   - imagem;
   - Scene JSON;
   - histórico da conversa;
   - pergunta atual;
9. recebe uma resposta textual;
10. salva a pergunta e a resposta;
11. o aplicativo Android lê a resposta utilizando Text-to-Speech.

IMPORTANTE:

Esta é a primeira versão do projeto.

NÃO implementar ainda:

- detecção de perigos;
- Safety Engine;
- depth estimation;
- segmentação;
- OCR;
- RAG;
- Qdrant;
- RabbitMQ;
- microsserviços;
- ESP32;
- wearable;
- autenticação complexa;
- cloud storage;
- infraestrutura cloud;
- agentes autônomos.

A arquitetura, entretanto, deve ser preparada para permitir essas extensões futuramente.

---

# 2. STACK DEFINIDA

Backend:

- Python 3.12
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- PostgreSQL
- pytest
- Ruff
- mypy quando fizer sentido

Computer Vision:

- YOLO pré-treinado em COCO
- OpenCV
- NumPy
- Pillow quando necessário

Generative AI:

- Ollama
- Qwen3.5 4B
- VLM multimodal compatível com Ollama

Mobile:

- Android
- Kotlin
- Jetpack Compose, se adequado
- Camera
- Microphone
- Text-to-Speech

Storage:

- armazenamento local no mesmo PC que executa a GPU.

Inicialmente:

data/images/

Não utilizar MinIO nesta primeira versão.

---

# 3. ARQUITETURA

A aplicação deve ser inicialmente um MODULAR MONOLITH.

Não criar microsserviços separados.

Estrutura conceitual:

Android
    |
    | HTTP
    v
FastAPI
    |
    v
Scene Service
    |
    +--> Image Storage
    |
    +--> YOLO Detector
    |
    +--> Position Analyzer
    |
    +--> Color Analyzer
    |
    +--> Scene Builder
    |
    v
PostgreSQL

Quando o usuário fizer uma pergunta:

Android
    |
    v
FastAPI
    |
    v
Conversation Service
    |
    +--> Scene JSON
    +--> Conversation History
    +--> Image
    |
    v
Ollama / Qwen3.5 4B
    |
    v
Answer
    |
    v
PostgreSQL
    |
    v
Android TTS

---

# 4. REGRA FUNDAMENTAL SOBRE CONVERSAS

CADA NOVA FOTO CRIA UMA NOVA CONVERSA.

Exemplo:

Foto 1
    -> Conversation 1
       -> pergunta
       -> resposta
       -> pergunta
       -> resposta

Nova foto

Foto 2
    -> Conversation 2
       -> pergunta
       -> resposta

O histórico da Conversation 1 NÃO deve ser utilizado na Conversation 2.

Portanto:

scene 1 <-> conversation 1
scene 2 <-> conversation 2

A conversa existe apenas enquanto a cena correspondente estiver ativa.

No aplicativo:

[CAPTURAR FOTO]

cria:

- nova Scene
- nova Conversation
- limpa o estado de conversa anterior

---

# 5. PRINCÍPIOS DE ARQUITETURA

Não colocar lógica de negócio em controllers.

Não colocar YOLO diretamente nos endpoints.

Não colocar chamadas ao Ollama diretamente nos controllers.

Não colocar SQLAlchemy diretamente nos controllers.

Preferir:

Controller
    ->
Application Service
    ->
Domain abstractions / interfaces
    ->
Infrastructure implementations

Exemplo:

SceneController
    ->
SceneService
    ->
ObjectDetector
    ->
YOLODetector

SceneService
    ->
ColorAnalyzer
    ->
OpenCVColorAnalyzer

ConversationService
    ->
VisionLanguageModel
    ->
OllamaVLM

ConversationService
    ->
ConversationRepository

---

# 6. PADRÕES DE PROJETO

Utilizar padrões apenas quando trouxerem benefício real.

Padrões esperados:

- Repository Pattern
- Service Layer
- Dependency Injection
- Strategy Pattern
- Adapter Pattern
- Factory somente quando houver necessidade
- DTO / Pydantic schemas
- Domain entities

Não criar abstrações artificiais.

O projeto deve permanecer simples.

---

# 7. ESTRUTURA DE DIRETÓRIOS

Preferencialmente:

backend/
    app/
        api/
            controllers/
            schemas/

        application/
            services/

        domain/
            entities/
            models/
            protocols/

        infrastructure/
            database/
            repositories/
            storage/
            vision/
            vlm/

        config/

    tests/

android/

prompts/
    system/
    scene/
    question/

scenarios/

evaluation/

data/
    images/

docker/

docs/

README.md

A estrutura pode ser adaptada se existir uma justificativa técnica melhor.

---

# 8. DETECTOR

Utilizar YOLO pré-treinado para detecção no COCO.

IMPORTANTE:

COCO possui 80 classes de detecção.

Não assumir que COCO possui 1000 classes.

Não treinar o modelo nesta primeira versão.

A finalidade inicial é inferência.

O detector deve retornar uma estrutura interna independente da biblioteca utilizada.

Não permitir que objetos específicos da biblioteca YOLO vazem para o domínio.

Exemplo conceitual:

Detection:
    object_id
    class_name
    class_id
    confidence
    bounding_box

---

# 9. POSIÇÃO

Calcular posição aproximada com base no centro da bounding box.

Horizontal:

- left
- center
- right

Vertical:

- top
- middle
- bottom

Região:

- front-left
- front-center
- front-right
- upper-left
- upper-center
- upper-right
- lower-left
- lower-center
- lower-right

Não inferir distância física.

Não utilizar "perto" ou "longe" como distância real nesta versão.

---

# 10. COR

Utilizar OpenCV.

A cor deve ser considerada:

"cor predominante da região detectada"

e não necessariamente a cor de uma parte semântica específica.

Por exemplo:

person
    -> dominant_color

NÃO assumir automaticamente:

person -> shirt color.

Perguntas como "qual a cor da camisa?" podem ser encaminhadas à VLM e analisadas visualmente.

---

# 11. SCENE JSON

O formato deve seguir aproximadamente:

{
    "scene_id": "...",
    "conversation_id": "...",
    "image": {
        "storage_key": "...",
        "width": 1920,
        "height": 1080
    },
    "model": {
        "name": "...",
        "task": "detect",
        "dataset": "COCO"
    },
    "objects": [
        {
            "object_id": "...",
            "class": {
                "id": 0,
                "name": "person",
                "confidence": 0.98
            },
            "bbox": {
                "x1": 100,
                "y1": 200,
                "x2": 500,
                "y2": 900
            },
            "position": {
                "horizontal": "center",
                "vertical": "middle",
                "region": "front-center"
            },
            "color": {
                "name": "blue",
                "rgb": [20, 80, 180],
                "confidence": 0.82
            }
        }
    ]
}

O formato pode evoluir, mas alterações devem ser justificadas.

---

# 12. BANCO DE DADOS

Utilizar PostgreSQL.

Entidades principais:

User
Device
Conversation
Scene
DetectedObject
Message

Relacionamentos:

User
    |
    +--> Conversations
    |
    +--> Devices

Conversation
    |
    +--> Scene
    |
    +--> Messages

Scene
    |
    +--> DetectedObjects

Messages
    |
    +--> user / assistant

Não armazenar a imagem como BYTEA nesta primeira versão.

Guardar a imagem no filesystem local.

No banco guardar:

- storage_key/path
- filename
- MIME type
- width
- height
- size
- hash quando implementado

---

# 13. CONVERSAÇÃO

Cada Conversation possui:

- id
- scene_id
- created_at
- updated_at

Cada Message possui:

- id
- conversation_id
- role
- content
- created_at
- model_name quando for assistant
- prompt_version quando aplicável
- latency_ms quando aplicável

A VLM recebe:

SYSTEM PROMPT
+
IMAGE
+
SCENE JSON
+
CONVERSATION HISTORY
+
CURRENT QUESTION

---

# 14. PROMPT DA VLM

A VLM deve ser instruída a:

- ajudar uma pessoa cega ou com baixa visão;
- responder de forma curta e natural;
- priorizar a imagem e o Scene JSON;
- não inventar objetos;
- não transformar inferências em fatos;
- indicar incerteza;
- usar esquerda, direita, centro, acima e abaixo;
- não inventar distância;
- responder diretamente à pergunta;
- não repetir toda a descrição se a pergunta for específica.

Não implementar ainda regras de segurança física.

---

# 15. API

Endpoints esperados:

POST /api/v1/conversations

POST /api/v1/scenes

GET /api/v1/scenes/{scene_id}

POST /api/v1/conversations/{conversation_id}/messages

GET /api/v1/conversations/{conversation_id}

A API deve usar Pydantic schemas.

---

# 16. ANDROID

O aplicativo deve ser extremamente simples.

Tela principal:

- botão grande para capturar foto;
- botão para falar/perguntar;
- indicação simples do estado;
- saída por Text-to-Speech.

Não criar uma interface visual complexa.

Fluxo:

CAPTURAR FOTO
    ->
POST /scenes
    ->
nova conversation
    ->
usuário fala pergunta
    ->
POST /messages
    ->
resposta
    ->
TTS

Cada nova foto limpa o estado anterior.

---

# 17. TESTES

Todos os componentes importantes devem possuir testes.

Prioridade:

1. domain
2. position analyzer
3. color analyzer
4. scene builder
5. repositories
6. services
7. API
8. integração com Ollama
9. Android posteriormente

Não depender de GPU nos testes unitários.

Mockar:

YOLO
Ollama
filesystem quando apropriado.

---

# 18. CENÁRIOS

Criar cenários de avaliação:

S01 Office
S02 Kitchen
S03 Living Room
S04 Street
S05 Person
S06 Desk
S07 Unknown Scene
S08 Multi-turn Conversation

Perguntas devem testar:

- descrição geral;
- presença;
- posição;
- cor;
- relações espaciais;
- características;
- perguntas de acompanhamento;
- perguntas sobre objetos inexistentes;
- ambiguidade;
- continuidade de conversa.

---

# 19. QUALIDADE DE CÓDIGO

Código profissional.

Priorizar:

- type hints;
- funções pequenas;
- nomes claros;
- baixo acoplamento;
- alta coesão;
- tratamento de erros;
- logs;
- configuração por environment variables;
- testes;
- documentação mínima necessária.

Não usar comentários óbvios.

Não criar código prematuramente complexo.

---

# 20. CONFIGURAÇÃO

Segredos e configurações devem vir de environment variables.

Exemplo:

DATABASE_URL
OLLAMA_BASE_URL
OLLAMA_MODEL
IMAGE_STORAGE_PATH
YOLO_MODEL
YOLO_CONFIDENCE_THRESHOLD

Nunca colocar credenciais diretamente no código.

Criar .env.example.

Nunca commitar .env.

---

# 21. GIT

Cada etapa deve terminar com:

1. implementação;
2. testes;
3. lint;
4. type checking quando aplicável;
5. atualização de documentação;
6. git diff review;
7. commit seguindo Conventional Commits.

Exemplos:

feat(vision): integrate coco object detector

feat(scene): add spatial position analysis

feat(vlm): integrate ollama vision model

test(scene): add scene builder tests

fix(conversation): isolate history between scenes

---

# 22. REGRA PARA O CLAUDE CODE

Antes de implementar:

1. inspecione o projeto existente;
2. identifique o estado atual;
3. não recrie arquivos desnecessariamente;
4. preserve decisões existentes;
5. procure testes existentes;
6. implemente somente o escopo solicitado;
7. não antecipe funcionalidades de etapas futuras.

Depois de implementar:

1. execute os testes;
2. execute lint;
3. execute type checking se configurado;
4. corrija problemas;
5. revise os arquivos modificados;
6. informe o que foi feito;
7. informe comandos executados;
8. informe limitações;
9. sugira o próximo passo somente no final.

NÃO faça commit automaticamente, a menos que o prompt da etapa solicite explicitamente.

---

# 23. FUTURO — NÃO IMPLEMENTAR AGORA

O projeto poderá posteriormente adicionar:

- Safety Engine
- fire detection
- hole/stairs detection
- construction detection
- depth estimation
- segmentation
- OCR
- Qdrant
- RAG
- RabbitMQ
- processamento assíncrono
- microsserviços
- ESP camera
- Bluetooth
- wearable
- cloud deployment
- observability
- model fine-tuning

A arquitetura atual deve permitir essas extensões sem reescrever o domínio principal.

---

# 24. OBJETIVO FINAL

O resultado esperado da primeira versão é:

Android
    ->
captura foto
    ->
FastAPI
    ->
salva imagem
    ->
YOLO COCO
    ->
OpenCV
    ->
Scene JSON
    ->
PostgreSQL
    ->
usuário pergunta
    ->
Qwen3.5 4B via Ollama
    ->
resposta
    ->
PostgreSQL
    ->
Android TTS

Cada nova fotografia inicia uma nova conversa.

A conversa continua enquanto nenhuma nova fotografia for capturada.

# INFRAESTRUTURA DISTRIBUÍDA

IMPORTANTE:

O PC que executa o backend e o PC que possui a GPU são máquinas diferentes.

O backend NÃO deve assumir que os modelos estão na mesma máquina.

Arquitetura atual:

Android
    |
    | HTTPS
    v
PC DEVELOPMENT
    |
    | Private VPN
    v
PC GPU
    |
    +--> NVIDIA GPU
    |
    +--> Ollama
    |
    +--> Qwen3.5 4B

O endereço do PC GPU deve ser configurável através de:

OLLAMA_BASE_URL

Nunca utilizar IP público hardcoded.

Nunca expor diretamente Ollama à Internet.

A comunicação entre as máquinas deve utilizar uma rede privada/VPN.

SSH é utilizado para administração da máquina GPU, não como mecanismo de comunicação da aplicação.

A aplicação deve utilizar HTTP para comunicação com Ollama.

SSH NÃO deve ser utilizado pelo FastAPI para executar comandos durante o funcionamento normal da aplicação.