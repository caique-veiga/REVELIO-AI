from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Revelio AI Backend"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://revelio:revelio@localhost:5432/revelio"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"

    image_storage_path: str = "data/images"

    yolo_model: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
