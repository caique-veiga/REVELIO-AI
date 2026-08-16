import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_scene_service
from app.api.schemas.scene_creation_schema import SceneCreationResponse
from app.application.services.scene_service import SceneService
from app.domain.protocols.image_storage import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scenes", tags=["scenes"])


@router.post("", response_model=SceneCreationResponse, status_code=status.HTTP_201_CREATED)
async def create_scene(
    file: UploadFile = File(...),
    scene_service: SceneService = Depends(get_scene_service),
) -> SceneCreationResponse:
    content = await file.read()
    filename = file.filename or "upload"

    try:
        scene = scene_service.create_scene(filename=filename, content=content)
    except (UnsupportedImageFormatError, InvalidImageError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Falha ao criar cena a partir da imagem enviada")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a imagem.",
        ) from exc

    assert scene.conversation_id is not None

    return SceneCreationResponse(
        scene_id=scene.scene_id,
        conversation_id=scene.conversation_id,
        object_count=len(scene.objects),
        status="created",
    )
