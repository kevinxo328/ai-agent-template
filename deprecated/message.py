from dataclasses import dataclass
from typing import Literal, Optional, Union

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


@dataclass
class UserMessage:
    content: str
    type: Literal["user"] = "user"


@dataclass
class StreamMessage:
    type: Literal["text", "thinking", "tool_call_start", "tool_call_end"]
    timestamp: str
    content: Optional[str] = None
    content_delta: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_call_args: Optional[Union[dict, str]] = None
    tool_call_name: Optional[str] = None


def to_model_message(message: Union[StreamMessage, UserMessage]) -> ModelMessage:
    """
    Convert a user or stream message to a model message.
    Only text messages are converted to model responses because tool calls or thinking processes are not needed for the model.
    """
    if isinstance(message, UserMessage):
        return ModelRequest(parts=[UserPromptPart(content=message.content)])
    elif isinstance(message, StreamMessage) and message.type == "text":
        return ModelResponse(parts=[TextPart(content=message.content or "")])
    raise ValueError("Invalid message type")
