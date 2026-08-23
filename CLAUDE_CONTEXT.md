# PROJETO — ASSISTENTE VISUAL PARA PESSOAS CEGAS

## 1. OBJETIVO

Estamos desenvolvendo um projeto de pós-graduação em IA Generativa e LLMs.

O objetivo é construir um protótipo funcional de um assistente visual para pessoas cegas ou com baixa visão.

O sistema recebe uma fotografia capturada por um aplicativo Android e:

1. armazena a imagem no computador que possui a GPU;
2. persiste Scene + Conversation no PostgreSQL (sem nenhum pré-processamento de visão
   computacional sobre a imagem);
3. permite ao usuário fazer perguntas sobre a cena;
4. envia para uma VLM (Ollama/Qwen, com fallback automático para Gemini Flash-Lite):
   - imagem;
   - histórico da conversa;
   - pergunta atual;
   - as tools de reconhecimento de pessoas (`register_person`, `identify_persons`);
5. a VLM decide sozinha se responde com texto direto ou chama uma tool — se chamar
   `register_person`/`identify_persons`, o backend roda reconhecimento facial (InsightFace) sob
   demanda sobre a imagem já enviada;
6. recebe uma resposta textual;
7. salva a pergunta e a resposta;
8. o aplicativo Android lê a resposta utilizando Text-to-Speech.

IMPORTANTE (mudança de arquitetura, PROMPT "Unificar Comportamento Gemini/Ollama"): a primeira
versão deste documento descrevia um pipeline YOLO (COCO) + OpenCV + Scene JSON estruturado
enviado à VLM. Esse pipeline foi removido por completo — a VLM analisa a imagem diretamente, sem
detecção de objetos pré-processada. Reconhecimento de pessoas (cadastro/identificação) é a única
capacidade estruturada que existe hoje, via tool calling e InsightFace.

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

- InsightFace (modelo `buffalo_l`, CPU/ONNX) — detecção facial + embedding para reconhecimento
  de pessoas
- NumPy
- Pillow quando necessário

Generative AI:

- Ollama + Qwen3.5 4B (tier 1, local) — VLM multimodal compatível com Ollama, com suporte a tool
  calling
- Gemini Flash-Lite (tier 2, API) — fallback automático quando o Ollama falha ou está
  desabilitado; também suporta tool calling
- Ambos os tiers recebem exatamente a mesma coisa (imagem + histórico + pergunta + tools:
  `register_person`, `identify_persons`) e decidem sozinhos como responder

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
    +--> Conversation History
    +--> Image
    +--> Tools (register_person, identify_persons)
    |
    v
Ollama / Qwen3.5 4B  --(fallback)-->  Gemini Flash-Lite
    |
    v
texto direto  OU  tool call --> Person Recognition Service --> InsightFace
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

Não colocar reconhecimento facial diretamente nos endpoints.

Não colocar chamadas ao Ollama/Gemini diretamente nos controllers.

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
ImageStorage
    ->
LocalImageStorage

ConversationService
    ->
VisionLanguageModel
    ->
FallbackVisionLanguageModel (Ollama -> Gemini)

ConversationService
    ->
PersonRecognitionService
    ->
FaceEncoder
    ->
InsightFaceEncoder

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

scenarios/

evaluation/

data/
    images/

docker/

docs/

README.md

A estrutura pode ser adaptada se existir uma justificativa técnica melhor.

---

# 8. DETECÇÃO FACIAL (RECONHECIMENTO DE PESSOAS)

Não há mais detector de objetos genérico (YOLO/COCO removido — ver seção 1). A única detecção
estruturada que existe é facial, via InsightFace (`FaceEncoder` protocol / `InsightFaceEncoder`),
acionada sob demanda quando a VLM chama `register_person` ou `identify_persons`.

O detector deve retornar uma estrutura interna independente da biblioteca utilizada.

Não permitir que objetos específicos da biblioteca InsightFace vazem para o domínio.

Exemplo conceitual:

FaceEncoding:
    bbox
    embedding

Não usar nenhum filtro de classe de objeto (ex. "person" do YOLO) antes do reconhecimento facial
— na prática, isso gerava falsos negativos em selfies reais. Quem decide se há um rosto na imagem
é exclusivamente o `FaceEncoder`; aceitar um falso-positivo ocasional (ex. registrar um animal) é
a troca aceita em favor de nunca bloquear uma pessoa real.

Cadastro (`register_person`) exige exatamente um rosto na foto. Identificação
(`identify_persons`) compara cada rosto encontrado contra os cadastros do usuário por similaridade
de cosseno (`FaceMatcher`, limiar configurável).

---

# 9. POSIÇÃO

Usada hoje só para relatar a posição horizontal de um rosto identificado (ex. "Maria à
esquerda"), via `PositionAnalyzer` (domain service, sem dependência externa). Calcular posição
aproximada com base no centro da bounding box.

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

Removido (ColorAnalyzer/OpenCV eram parte do pipeline YOLO — ver seção 1). Perguntas sobre cor
("qual a cor da camisa?", "a mochila é azul?") são respondidas diretamente pela VLM analisando a
imagem, sem nenhuma classificação de cor pré-processada.

---

# 11. SCENE JSON

Removido. A VLM não recebe mais nenhum JSON estruturado descrevendo a cena — só a imagem crua,
o histórico da conversa, a pergunta e as tools disponíveis. Ver seção 1 para o motivo da mudança.

---

# 12. BANCO DE DADOS

Utilizar PostgreSQL.

Entidades principais:

User
Device
Conversation
Scene
Message
Person
PersonPhoto

Relacionamentos:

User
    |
    +--> Conversations
    |
    +--> Devices
    |
    +--> Persons

Conversation
    |
    +--> Scene
    |
    +--> Messages

Person
    |
    +--> PersonPhotos

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
CONVERSATION HISTORY
+
CURRENT QUESTION
+
TOOLS (register_person, identify_persons)

---

# 14. PROMPT DA VLM

A VLM deve ser instruída a:

- ajudar uma pessoa cega ou com baixa visão;
- responder de forma curta e natural;
- priorizar a imagem (não há Scene JSON — ver seção 11);
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
3. face matcher
4. repositories
5. services
6. API
7. integração com Ollama
8. Android posteriormente

Não depender de GPU nos testes unitários.

Mockar:

VisionLanguageModel (Ollama/Gemini)
FaceEncoder (InsightFace)
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
GEMINI_API_KEY
GEMINI_MODEL
IMAGE_STORAGE_PATH
FACE_MODEL_NAME
FACE_MATCH_THRESHOLD

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
PostgreSQL (Scene + Conversation)
    ->
usuário pergunta
    ->
Qwen3.5 4B via Ollama (fallback: Gemini Flash-Lite)
    ->
texto direto OU tool call (register_person/identify_persons -> InsightFace)
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