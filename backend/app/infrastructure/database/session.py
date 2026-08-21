from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    """Fornece a Session da requisição — não decide quando commitar.

    O commit é responsabilidade explícita do Application Service, que
    conhece os limites da unidade de trabalho (ver `SceneService.create_scene`
    e `ConversationService.ask`). Commitar aqui, depois do `yield`, rodaria
    no código de limpeza da dependency do FastAPI — que só executa depois
    de a resposta HTTP já ter sido enviada ao cliente, abrindo uma janela
    de corrida onde uma chamada seguinte imediata não vê o dado ainda
    (causa raiz do 404 intermitente investigado na Etapa 13.1).
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
