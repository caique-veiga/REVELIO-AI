import logging
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.conversation_service import ConversationService
from app.application.services.person_recognition_service import PersonRecognitionService
from app.application.services.prompt_composer import PromptComposer
from app.application.services.scene_service import SceneService
from app.config.settings import get_settings
from app.domain.protocols.face_encoder import FaceEncoder
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.prompt_loader import PromptLoader
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.domain.services.face_matcher import FaceMatcher
from app.domain.services.position_analyzer import PositionAnalyzer
from app.infrastructure.database.session import get_db_session
from app.infrastructure.prompts.file_prompt_loader import FilePromptLoader
from app.infrastructure.storage.local_image_storage import LocalImageStorage
from app.infrastructure.vision.insightface_encoder import InsightFaceEncoder
from app.infrastructure.vlm.fallback_vision_language_model import FallbackVisionLanguageModel
from app.infrastructure.vlm.gemini_vision_language_model import GeminiVisionLanguageModel
from app.infrastructure.vlm.ollama_vision_language_model import OllamaVisionLanguageModel

logger = logging.getLogger(__name__)


@lru_cache
def get_image_storage() -> ImageStorage:
    settings = get_settings()
    return LocalImageStorage(
        root_path=settings.image_storage_path, max_size_bytes=settings.max_image_size_bytes
    )


@lru_cache
def get_position_analyzer() -> PositionAnalyzer:
    return PositionAnalyzer()


@lru_cache
def get_gemini_vision_language_model() -> GeminiVisionLanguageModel:
    settings = get_settings()
    return GeminiVisionLanguageModel(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.gemini_timeout_seconds,
        image_max_dimension=settings.image_max_dimension,
        image_jpeg_quality=settings.image_jpeg_quality,
        image_enable_optimization=settings.image_enable_optimization,
    )


@lru_cache
def get_vision_language_model() -> VisionLanguageModel:
    settings = get_settings()

    if not settings.gemini_enabled:
        # Gemini é o único fallback suportado (ETAPA 13.1) — sem ele não há
        # nenhum provider de VLM restante para atender a aplicação.
        raise RuntimeError(
            "GEMINI_ENABLED=false, mas o Gemini é o único fallback disponível "
            "quando o Ollama está desabilitado ou indisponível."
        )

    gemini = get_gemini_vision_language_model()

    if not settings.ollama_enabled:
        logger.error("OLLAMA_ENABLED=false — usando Gemini Flash-Lite como único provider de VLM")
        return gemini

    ollama = OllamaVisionLanguageModel(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        max_retries=settings.ollama_max_retries,
        num_ctx=settings.ollama_num_ctx,
    )
    return FallbackVisionLanguageModel(primary=ollama, fallback=gemini)


@lru_cache
def get_face_encoder() -> FaceEncoder:
    settings = get_settings()
    return InsightFaceEncoder(
        model_name=settings.face_model_name, detection_size=settings.face_detection_size
    )


@lru_cache
def get_face_matcher() -> FaceMatcher:
    settings = get_settings()
    return FaceMatcher(threshold=settings.face_match_threshold)


def get_person_recognition_service(
    session: Session = Depends(get_db_session),
    face_encoder: FaceEncoder = Depends(get_face_encoder),
    position_analyzer: PositionAnalyzer = Depends(get_position_analyzer),
    face_matcher: FaceMatcher = Depends(get_face_matcher),
) -> PersonRecognitionService:
    return PersonRecognitionService(
        session=session,
        face_encoder=face_encoder,
        position_analyzer=position_analyzer,
        face_matcher=face_matcher,
    )


def get_scene_service(
    session: Session = Depends(get_db_session),
    image_storage: ImageStorage = Depends(get_image_storage),
) -> SceneService:
    return SceneService(session=session, image_storage=image_storage)


@lru_cache
def get_prompt_loader() -> PromptLoader:
    settings = get_settings()
    return FilePromptLoader(prompts_root=settings.prompts_root)


def get_prompt_composer(
    prompt_loader: PromptLoader = Depends(get_prompt_loader),
) -> PromptComposer:
    return PromptComposer(prompt_loader=prompt_loader)


def get_conversation_service(
    session: Session = Depends(get_db_session),
    image_storage: ImageStorage = Depends(get_image_storage),
    vision_language_model: VisionLanguageModel = Depends(get_vision_language_model),
    prompt_composer: PromptComposer = Depends(get_prompt_composer),
    person_recognition_service: PersonRecognitionService = Depends(get_person_recognition_service),
) -> ConversationService:
    return ConversationService(
        session=session,
        image_storage=image_storage,
        vision_language_model=vision_language_model,
        prompt_composer=prompt_composer,
        person_recognition_service=person_recognition_service,
    )
