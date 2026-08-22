import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_conversation_service
from app.api.schemas.message_schema import (
    AskQuestionRequest,
    AskQuestionResponse,
    ConversationSchema,
)
from app.application.services.conversation_service import ConversationService
from app.domain.protocols.conversation_repository import ConversationNotFoundError
from app.domain.protocols.vision_language_model import (
    ModelUnavailableError,
    OllamaUnavailableError,
    VisionLanguageModelError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.post("/{conversation_id}/messages", response_model=AskQuestionResponse)
def ask_question(
    conversation_id: uuid.UUID,
    request: AskQuestionRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> AskQuestionResponse:
    try:
        answer = conversation_service.ask(conversation_id, request.content)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (OllamaUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except VisionLanguageModelError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falha ao processar pergunta da conversation %s", conversation_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a pergunta.",
        ) from exc

    return AskQuestionResponse.from_domain(answer)


@router.get("/{conversation_id}", response_model=ConversationSchema)
def get_conversation(
    conversation_id: uuid.UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationSchema:
    try:
        conversation, messages = conversation_service.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ConversationSchema.from_rows(conversation, messages)
