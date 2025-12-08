from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..schema import OpenAIModelConfig


def create_openai_model(config: OpenAIModelConfig, **kwargs) -> OpenAIChatModel:
    """
    Create an OpenAIChatModel instance based on the provided configuration.
    Can use any OpenAI-compatible endpoint by specifying the base_url.
    See https://ai.pydantic.dev/models/openai/ for more details.
    """
    return OpenAIChatModel(
        config.model_name,
        provider=OpenAIProvider(
            openai_client=AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                **kwargs,
            )
        ),
    )
