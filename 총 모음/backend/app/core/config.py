"""
Configuration management for the K-Euphemism Detector API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # API Settings
    API_TITLE: str = "K-Euphemism Detector API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database Settings
    DATABASE_URL: Optional[str] = "postgresql://user:password@localhost:5432/euphemism_db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis Settings (defaults for local, override via environment)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # ML Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/koelectra_euphemism")
    MODEL_GCS_BUCKET: str = os.getenv("MODEL_GCS_BUCKET", "k-euphemism-models")
    MODEL_GCS_PATH: str = os.getenv("MODEL_GCS_PATH", "koelectra_multitask_v2")
    KOELECTRA_MODEL: str = "monologg/koelectra-base-v3-discriminator"
    SENTENCE_TRANSFORMER: str = "jhgan/ko-sbert-sts"
    AUTO_DOWNLOAD_MODEL: bool = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

    # Gemini API Settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-2.0-flash"  # Latest available model

    # Performance Settings
    MAX_TEXT_LENGTH: int = 10000  # Increased for production
    BATCH_SIZE: int = 32
    CONFIDENCE_THRESHOLD: float = 0.85

    # Security Settings (production-ready)
    API_KEY_HEADER: str = "X-API-Key"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @property
    def DEBUG(self) -> bool:
        """Debug mode based on environment"""
        return self.ENVIRONMENT.lower() in ["development", "dev"]

    @property
    def CORS_ORIGINS(self) -> list:
        """CORS origins based on environment"""
        if self.DEBUG:
            # Development: Allow localhost for testing
            return [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ]
        else:
            # Production: Strict whitelist
            return [
                "https://k-euphemism-frontend-245053314944.asia-northeast3.run.app",
                "https://k-euphemism-api-245053314944.asia-northeast3.run.app"
            ]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
