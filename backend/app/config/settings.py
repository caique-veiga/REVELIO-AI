from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Revelio AI Backend"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://revelio:revelio@localhost:5432/revelio"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_timeout_seconds: float = 30.0
    ollama_max_retries: int = 2
    system_prompt_path: str = "prompts/system/default.md"

    image_storage_path: str = "data/images"
    max_image_size_bytes: int = 10_485_760

    yolo_model: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
