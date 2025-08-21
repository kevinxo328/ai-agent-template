from datetime import datetime

from pydantic_ai import Agent, Tool

from utils.langfuse import langfuse
from utils.model import OpenAIModelConfig, create_openai_model


def get_current_time() -> str:
    """
    Get the current time with timezone as a string.
    """
    return datetime.now(tz=None).isoformat()


def create_chat_agent(config: OpenAIModelConfig) -> Agent:
    model = create_openai_model(config)
    instructions = """
    You are a ai assistant. Answer the user's questions. 
    You can use the following tools to assist you: `get_current_time` 
    """

    return Agent(
        model=model,
        instructions=instructions,
        tools=[Tool(get_current_time, takes_ctx=False)],
        instrument=langfuse.auth_check(),
        name="Chat Agent",
    )
