from fastapi import FastAPI

from app.api.controllers.conversation_controller import router as conversation_router
from app.api.controllers.health_controller import router as health_router
from app.api.controllers.scene_controller import router as scene_router
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(health_router)
app.include_router(scene_router)
app.include_router(conversation_router)
