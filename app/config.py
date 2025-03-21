import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # API settings
    API_KEY: str = os.getenv("API_KEY", "your-secure-api-key")

    # LLM settings
    DEEPINFRA_API_TOKEN: str = os.getenv("DEEPINFRA_API_TOKEN", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    DEEPINFRA_ENDPOINT_MODELS: str = "https://api.deepinfra.com/v1/openai/models"
    LLM_TEMPERATURE: float = os.getenv("LLM_TEMPERATURE", 0)
    LLM_TOP_P: float = os.getenv("LLM_TOP_P", 0.1)
    LLM_FREQUENCY_PENALTY: float = os.getenv("LLM_FREQUENCY_PENALTY", 0.1)
    LLM_PRESENCE_PENALTY: float = os.getenv("LLM_PRESENCE_PENALTY", 0.1)
    LLM_MAX_TOKENS: int = os.getenv("LLM_MAX_TOKENS", 3000)

    # Milvus settings
    MILVUS_URI: str = os.getenv(
        "MILVUS_URI",
        "https://in03-26f04ddfb7604fe.serverless.gcp-us-west1.cloud.zilliz.com",
    )
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "")

    # System settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS settings
    CORS_ORIGINS: list = ["*"]  # For production, specify your domains

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def setup_env(self):
        """Setup environment variables."""
        if "DEEPINFRA_API_TOKEN" in os.environ:
            del os.environ["DEEPINFRA_API_TOKEN"]
        os.environ["DEEPINFRA_API_TOKEN"] = self.DEEPINFRA_API_TOKEN


settings = Settings()
settings.setup_env()
