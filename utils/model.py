from typing import Optional

from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class OpenAIModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: Optional[str] = None


def create_openai_model(config: OpenAIModelConfig) -> OpenAIModel:
    return OpenAIModel(
        config.model_name,
        provider=OpenAIProvider(api_key=config.api_key, base_url=config.base_url),
    )
