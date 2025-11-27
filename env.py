from typing import Optional

from pydantic_settings import BaseSettings


class LangfuseSettings(BaseSettings):
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"


class OpenAISettings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL_NAME: str = "gpt-4o"

    class Config:
        env_file = ".env"

class LanggraphSettings(BaseSettings):
    LANGSMITH_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"

class Settings(LangfuseSettings, OpenAISettings, LanggraphSettings):
    pass


env = Settings()
