from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "ragbench"
    postgres_user: str = "ragbench"
    postgres_password: str = "ragbench"
    redis_url: str = "redis://redis:6379/0"
    qdrant_url: str = "http://qdrant:6333"
    opensearch_url: str = "http://opensearch:9200"
    opensearch_user: str = "admin"
    opensearch_password: str = "admin"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "gpt-oss:20b"
    artifact_dir: str = "/app/artifacts"
    config_dir: str = "/app/configs"
    data_dir: str = "/app/data"

    @property
    def postgres_url(self) -> str:
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

settings = Settings()
