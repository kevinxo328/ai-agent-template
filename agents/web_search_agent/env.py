from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class TavilySettings(BaseSettings):
    TAVILY_API_KEY: str = ""


class OpenAISettings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL_NAME: str = "gpt-4o"


class LanggraphSettings(BaseSettings):
    LANGSMITH_API_KEY: Optional[str] = None


class Settings(TavilySettings, OpenAISettings, LanggraphSettings):
    pass

    class Config:
        env_file = BASE_DIR / ".env"


env = Settings()
