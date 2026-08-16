from fastapi import FastAPI

from app.api.controllers.health_controller import router as health_router
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.include_router(health_router)
