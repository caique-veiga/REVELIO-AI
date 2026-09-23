# REVELIO-AI: Assistente Visual Multimodal para Pessoas Cegas ou com Baixa Visão

## Identificação

**Aluno:** Caique Veiga de Lima Muniz
**Orientador:** Leonardo
**Turma:** Inteligência Artificial Generativa & Large Language Models
**Matrícula:** 252100206

**Código-fonte:** https://github.com/caique-veiga/REVELIO-AI

---

## Resumo

O REVELIO-AI é um protótipo de assistente visual multimodal desenvolvido para auxiliar pessoas cegas ou com baixa visão na compreensão de cenas capturadas por uma câmera. A aplicação utiliza um aplicativo Android para captura de imagens e interação por voz, um backend desenvolvido em FastAPI e modelos de visão e linguagem (Vision-Language Models — VLMs) para interpretar as imagens e responder às perguntas do usuário.

A arquitetura foi projetada para permitir o uso de um modelo local, executado por meio do Ollama, com fallback para a API do Gemini quando necessário. Após o envio de uma fotografia, o backend armazena a imagem, cria uma nova conversa associada à cena e encaminha a imagem, o histórico da conversa e a pergunta do usuário ao modelo multimodal. O sistema também utiliza Tool Calling para disponibilizar funcionalidades estruturadas, como cadastro e identificação de pessoas, apoiadas pelo InsightFace.

O projeto busca explorar uma arquitetura de baixo custo e modular para aplicações assistivas baseadas em Inteligência Artificial Generativa, combinando processamento multimodal, interação por voz, reconhecimento facial e execução local de modelos.

---

## Introdução

Pessoas cegas ou com baixa visão podem encontrar dificuldades para obter informações visuais presentes no ambiente, especialmente em situações nas quais não há outra pessoa disponível para descrever uma cena. Sistemas de visão computacional podem auxiliar nesse processo ao transformar informações presentes em imagens em descrições compreensíveis.

O REVELIO-AI foi desenvolvido com o objetivo de investigar a aplicação de modelos multimodais capazes de receber imagens e linguagem natural simultaneamente. Durante o desenvolvimento, a abordagem inicialmente baseada em detecção de objetos e processamento estruturado da cena foi simplificada após testes demonstrarem que um VLM poderia interpretar diretamente a imagem e responder a perguntas relacionadas ao conteúdo visual.

A solução atual utiliza uma arquitetura modular monolítica, permitindo separar as responsabilidades da API, dos serviços de aplicação, do domínio e das implementações de infraestrutura. Essa organização facilita a substituição do modelo de visão, do armazenamento e de outros componentes sem alterar as regras centrais da aplicação.

---

## Arquitetura e Implementação

O fluxo principal do sistema é composto pelas seguintes etapas:

1. O usuário captura uma fotografia utilizando o aplicativo Android.
2. A imagem é enviada para o backend por meio da API REST.
3. O backend armazena a imagem localmente e cria uma nova cena e uma nova conversa.
4. O identificador da conversa é retornado ao aplicativo.
5. O usuário realiza uma pergunta por voz sobre a imagem.
6. A pergunta é enviada ao backend juntamente com o histórico da conversa.
7. O backend encaminha a imagem, o histórico, a pergunta e as ferramentas disponíveis para o VLM.
8. O modelo pode responder diretamente ou solicitar a execução de uma ferramenta.
9. A resposta é retornada ao aplicativo Android e pode ser reproduzida por síntese de voz.
10. Quando uma nova fotografia é capturada, uma nova conversa é iniciada.

### Componentes principais

**Aplicativo Android**

Responsável pela captura da fotografia, entrada de voz, comunicação HTTP com o backend e reprodução da resposta utilizando Text-to-Speech (TTS). A interface foi mantida propositalmente simples, priorizando a interação por câmera e voz.

**Backend FastAPI**

Implementa a API REST e coordena o processamento das cenas e das perguntas. A aplicação segue uma organização modular:

```text
Controller
    ↓
Application Service
    ↓
Domain Interfaces
    ↓
Infrastructure
```

As interfaces do domínio permitem desacoplar o sistema das implementações específicas utilizadas na infraestrutura.

**Banco de dados**

O PostgreSQL é utilizado para persistir informações relacionadas às cenas e conversas.

**Armazenamento de imagens**

As imagens recebidas são armazenadas localmente pelo backend. Cada imagem é associada a uma nova cena e a uma nova conversa.

**Vision-Language Model**

A interface de VLM permite utilizar diferentes provedores com o mesmo fluxo de aplicação. A configuração atual utiliza:

- Ollama + modelo Qwen como primeira opção;
- Gemini Flash-Lite como fallback;
- imagem, histórico, pergunta e ferramentas enviados de forma multimodal.

**Reconhecimento facial**

O InsightFace é utilizado para geração de embeddings faciais e comparação de similaridade. O reconhecimento é disponibilizado ao VLM por meio de ferramentas, evitando que a lógica de reconhecimento fique diretamente acoplada ao modelo de linguagem.

### Tool Calling

O modelo pode utilizar ferramentas para executar operações estruturadas relacionadas às pessoas presentes na cena.

Entre as ferramentas implementadas estão:

- `register_person`: registra uma pessoa a partir de uma face detectada;
- `identify_persons`: compara faces presentes na imagem com pessoas previamente registradas.

Dessa forma, o modelo pode utilizar a linguagem natural para decidir quando uma operação estruturada deve ser executada.

---

## Resultados

O protótipo implementa o fluxo principal de captura de imagem, criação de cena, criação de conversa e interação multimodal por perguntas.

A API disponibiliza, entre outros, os seguintes endpoints:

### Criar uma cena

```http
POST /api/v1/scenes
```

Recebe uma imagem em formato multipart e cria uma nova cena e uma nova conversa.

Resposta simplificada:

```json
{
  "scene_id": "...",
  "conversation_id": "...",
  "status": "created"
}
```

### Enviar uma pergunta

```http
POST /api/v1/conversations/{conversation_id}/messages
```

Envia uma pergunta relacionada à cena associada à conversa.

Resposta simplificada:

```json
{
  "answer": "...",
  "scene_id": "..."
}
```

Cada nova fotografia inicia uma nova conversa. Dessa maneira, o histórico utilizado pelo VLM permanece associado somente à cena atualmente analisada.

A arquitetura também permite alternar entre processamento local e processamento por API. O uso do Ollama possibilita executar o modelo localmente em uma máquina equipada com GPU, enquanto o Gemini pode ser utilizado como alternativa quando o modelo local não estiver disponível ou não produzir uma resposta adequada.

O sistema possui ainda testes automatizados utilizando `pytest`, análise estática com `mypy` e verificação de qualidade de código com `Ruff`.

---

## Conclusões

O desenvolvimento do REVELIO-AI permitiu investigar a utilização de modelos multimodais como núcleo de uma aplicação assistiva. A arquitetura inicialmente considerada utilizava uma etapa de detecção de objetos e geração de informações estruturadas antes do processamento pelo modelo de linguagem. Durante os experimentos, observou-se que um VLM poderia receber diretamente a imagem e responder a diferentes perguntas sobre a cena, reduzindo a quantidade de processamento intermediário necessário.

A arquitetura atual concentra o processamento visual no VLM e utiliza Tool Calling para funcionalidades que exigem operações estruturadas, como o reconhecimento de pessoas. Essa abordagem mantém o sistema modular e permite substituir o modelo utilizado sem modificar significativamente as demais camadas da aplicação.

Como trabalhos futuros, podem ser investigados recursos adicionais de acessibilidade e percepção visual, como identificação de obstáculos e situações de risco, OCR, estimativa espacial mais detalhada, processamento contínuo por vídeo e melhorias na interação por voz. Também pode ser avaliada a utilização de modelos locais mais eficientes e estratégias adicionais de fallback entre modelos.

---

## Tecnologias utilizadas

- Python 3.12
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- PostgreSQL
- Pillow
- Ollama
- Qwen
- Gemini Flash-Lite
- InsightFace
- ONNX Runtime
- Android
- pytest
- Ruff
- mypy
- uv
- Docker

---

## Instalação e execução

### Pré-requisitos

- Python 3.12
- uv
- Docker e Docker Compose
- Android Studio, caso o aplicativo Android seja executado localmente
- Ollama, caso seja utilizado o modelo local

### Backend

Clone o repositório:

```bash
git clone https://github.com/caique-veiga/REVELIO-AI.git
cd REVELIO-AI
```

Instale as dependências:

```bash
uv sync
```

Configure as variáveis de ambiente no arquivo `.env`.

Exemplo das principais configurações:

```env
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen
GEMINI_API_KEY=
FACE_MATCH_THRESHOLD=
```

Inicie o PostgreSQL:

```bash
docker compose up -d
```

Execute as migrações:

```bash
uv run alembic upgrade head
```

Inicie a API:

```bash
uv run uvicorn app.main:app --reload
```

A documentação interativa da API estará disponível em:

```text
http://localhost:8000/docs
```

### Testes

Execute os testes automatizados:

```bash
uv run pytest
```

Verifique o código com Ruff:

```bash
uv run ruff check .
```

Execute a verificação de tipos:

```bash
uv run mypy .
```

---

## Escopo atual

O protótipo atual concentra-se em:

- captura de uma imagem;
- interpretação multimodal da cena;
- perguntas em linguagem natural;
- manutenção de contexto dentro da conversa;
- interação por voz no aplicativo Android;
- resposta por síntese de voz;
- execução local de VLM;
- fallback para modelo multimodal por API;
- reconhecimento facial por meio de ferramentas;
- persistência de cenas e conversas.

Funcionalidades como autenticação de usuários, feed, dashboard, RAG e arquitetura distribuída em múltiplos microsserviços não fazem parte do escopo atual do protótipo.

---

## Referências

- Google. **Gemini API Documentation**. Documentação oficial da API Gemini.
- Ollama. **Ollama Documentation**. Documentação oficial do projeto.
- InsightFace. **InsightFace Documentation**. Biblioteca para análise e reconhecimento facial.
- FastAPI. **FastAPI Documentation**. Framework para desenvolvimento de APIs em Python.
- PostgreSQL. **PostgreSQL Documentation**. Sistema gerenciador de banco de dados utilizado no projeto.

---

## Repositório

Código-fonte e documentação do projeto:

https://github.com/caique-veiga/REVELIO-AI
