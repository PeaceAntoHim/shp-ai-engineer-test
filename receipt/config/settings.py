import os
from functools import lru_cache
from typing import List
import asyncpg
from pydantic import validator
from dotenv import load_dotenv
load_dotenv()

class Settings():
    """Application settings with environment variable support"""

    # Database - separate components
    db_host: str = os.getenv("DB_HOST")
    db_port: int = os.getenv("DB_PORT", 5432)
    db_user: str = os.getenv("DB_USER")
    db_password: str = os.getenv("DB_PASS")
    db_name: str = os.getenv("DB_NAME")

    # Constructed database URL for async PostgreSQL
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # File Upload
    upload_directory: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = ["jpg", "jpeg", "png", "pdf"]

    # CORS
    allowed_origins: List[str] = ["http://localhost:8000"]

    # Application
    debug: bool = False
    log_level: str = "INFO"

    # AI Configuration
    ai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.1

    @validator("allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()