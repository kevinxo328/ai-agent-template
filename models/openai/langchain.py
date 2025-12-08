from langchain.chat_models import BaseChatModel, init_chat_model

from ..schema import OpenAIModelConfig


def create_openai_model(config: OpenAIModelConfig, **kwargs) -> BaseChatModel:
    """
    Create an OpenAI chat model using LangChain's init_chat_model function.
    """
    return init_chat_model(
        model_provider="openai",
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
        **kwargs,
    )
