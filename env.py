from typing import Optional

from pydantic_settings import BaseSettings


class LangfuseSettings(BaseSettings):
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"


class Settings(LangfuseSettings):
    pass


env = Settings()
