from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.conversation_service import ConversationService
from app.application.services.prompt_composer import PromptComposer
from app.application.services.scene_service import SceneService
from app.config.settings import get_settings
from app.domain.entities.model_metadata import ModelMetadata
from app.domain.protocols.color_analyzer import ColorAnalyzer
from app.domain.protocols.image_storage import ImageStorage
from app.domain.protocols.object_detector import ObjectDetector
from app.domain.protocols.prompt_loader import PromptLoader
from app.domain.protocols.vision_language_model import VisionLanguageModel
from app.domain.services.position_analyzer import PositionAnalyzer
from app.domain.services.question_classifier import QuestionClassifier
from app.domain.services.scene_builder import SceneBuilder
from app.infrastructure.database.session import get_db_session
from app.infrastructure.prompts.file_prompt_loader import FilePromptLoader
from app.infrastructure.storage.local_image_storage import LocalImageStorage
from app.infrastructure.vision.opencv_color_analyzer import OpenCVColorAnalyzer
from app.infrastructure.vision.yolo_object_detector import YOLOObjectDetector
from app.infrastructure.vlm.ollama_vision_language_model import OllamaVisionLanguageModel

_DETECTOR_TASK = "detect"
_DETECTOR_DATASET = "COCO"


@lru_cache
def get_image_storage() -> ImageStorage:
    settings = get_settings()
    return LocalImageStorage(
        root_path=settings.image_storage_path, max_size_bytes=settings.max_image_size_bytes
    )


@lru_cache
def get_object_detector() -> ObjectDetector:
    settings = get_settings()
    return YOLOObjectDetector(
        model_path=settings.yolo_model, confidence_threshold=settings.yolo_confidence_threshold
    )


@lru_cache
def get_color_analyzer() -> ColorAnalyzer:
    return OpenCVColorAnalyzer()


@lru_cache
def get_position_analyzer() -> PositionAnalyzer:
    return PositionAnalyzer()


@lru_cache
def get_scene_builder() -> SceneBuilder:
    return SceneBuilder()


@lru_cache
def get_model_metadata() -> ModelMetadata:
    settings = get_settings()
    return ModelMetadata(name=settings.yolo_model, task=_DETECTOR_TASK, dataset=_DETECTOR_DATASET)


@lru_cache
def get_vision_language_model() -> VisionLanguageModel:
    settings = get_settings()
    return OllamaVisionLanguageModel(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        max_retries=settings.ollama_max_retries,
        num_ctx=settings.ollama_num_ctx,
    )


def get_scene_service(
    session: Session = Depends(get_db_session),
    image_storage: ImageStorage = Depends(get_image_storage),
    object_detector: ObjectDetector = Depends(get_object_detector),
    position_analyzer: PositionAnalyzer = Depends(get_position_analyzer),
    color_analyzer: ColorAnalyzer = Depends(get_color_analyzer),
    scene_builder: SceneBuilder = Depends(get_scene_builder),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
) -> SceneService:
    return SceneService(
        session=session,
        image_storage=image_storage,
        object_detector=object_detector,
        position_analyzer=position_analyzer,
        color_analyzer=color_analyzer,
        scene_builder=scene_builder,
        model_metadata=model_metadata,
    )


@lru_cache
def get_prompt_loader() -> PromptLoader:
    settings = get_settings()
    return FilePromptLoader(prompts_root=settings.prompts_root)


@lru_cache
def get_question_classifier() -> QuestionClassifier:
    return QuestionClassifier()


def get_prompt_composer(
    prompt_loader: PromptLoader = Depends(get_prompt_loader),
    question_classifier: QuestionClassifier = Depends(get_question_classifier),
) -> PromptComposer:
    return PromptComposer(prompt_loader=prompt_loader, question_classifier=question_classifier)


def get_conversation_service(
    session: Session = Depends(get_db_session),
    image_storage: ImageStorage = Depends(get_image_storage),
    vision_language_model: VisionLanguageModel = Depends(get_vision_language_model),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
    prompt_composer: PromptComposer = Depends(get_prompt_composer),
) -> ConversationService:
    return ConversationService(
        session=session,
        image_storage=image_storage,
        vision_language_model=vision_language_model,
        model_metadata=model_metadata,
        prompt_composer=prompt_composer,
    )
