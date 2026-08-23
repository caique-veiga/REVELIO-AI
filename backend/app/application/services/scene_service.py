import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.entities.scene import Scene
from app.domain.protocols.image_storage import ImageStorage
from app.infrastructure.database.models import Conversation, SceneModel, User
from app.infrastructure.repositories.conversation_repository import (
    SqlAlchemyConversationRepository,
)
from app.infrastructure.repositories.scene_repository import SqlAlchemySceneRepository

logger = logging.getLogger(__name__)


class SceneService:
    """Orquestra a criação de uma cena:

        validar imagem -> salvar imagem -> PostgreSQL (Scene, Conversation)

    Cada chamada a `create_scene` sempre cria uma nova Scene e uma nova
    Conversation — nunca reaproveita uma conversation existente. Não roda
    nenhum pipeline de visão computacional aqui (YOLO/ColorAnalyzer/
    PositionAnalyzer/SceneBuilder removidos — ver PROMPT "Unificar
    Comportamento Gemini/Ollama"): a VLM analisa a imagem diretamente,
    sem Scene JSON pré-processado.
    """

    def __init__(self, session: Session, image_storage: ImageStorage) -> None:
        self._session = session
        self._image_storage = image_storage
        self._scene_repository = SqlAlchemySceneRepository(session)
        self._conversation_repository = SqlAlchemyConversationRepository(session)

    def create_scene(self, filename: str, content: bytes) -> Scene:
        scene_id = uuid.uuid4()

        # A imagem é salva no filesystem antes da transação do banco (não há
        # transação distribuída entre os dois). Se a persistência falhar
        # depois, o rollback do banco por si só não apaga o arquivo — por
        # isso a remoção compensatória explícita no except abaixo.
        stored_image = self._image_storage.save(scene_id, filename, content)

        try:
            scene_row = self._scene_repository.add(
                SceneModel(
                    id=scene_id,
                    image_storage_key=stored_image.storage_key,
                    image_filename=stored_image.filename,
                    image_mime_type=stored_image.mime_type,
                    image_width=stored_image.width,
                    image_height=stored_image.height,
                    image_size_bytes=stored_image.size_bytes,
                    image_hash=stored_image.sha256,
                )
            )

            user = self._get_or_create_default_user()
            conversation_row = self._conversation_repository.add(
                Conversation(user_id=user.id, scene_id=scene_row.id)
            )

            # Commit explícito aqui, antes de retornar ao controller — a
            # unidade de trabalho (Scene + Conversation) só é considerada
            # concluída quando este commit termina, e só então a resposta
            # HTTP de sucesso é construída.
            self._session.commit()
        except Exception:
            self._session.rollback()
            try:
                self._image_storage.delete(stored_image.storage_key)
            except Exception:
                logger.exception(
                    "Falha ao remover imagem órfã %s após rollback", stored_image.storage_key
                )
            raise

        return Scene(image=stored_image, scene_id=scene_row.id, conversation_id=conversation_row.id)

    def _get_or_create_default_user(self) -> User:
        # Ainda não há autenticação/gestão de usuários (fora de escopo desta
        # etapa) — reutiliza um único usuário padrão como dono provisório de
        # todas as conversations, até existir um fluxo real de identidade.
        user = self._session.execute(select(User).limit(1)).scalar_one_or_none()
        if user is None:
            user = User()
            self._session.add(user)
            self._session.flush()
        return user
