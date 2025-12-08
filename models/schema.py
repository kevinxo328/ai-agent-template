from typing import Optional

from pydantic import BaseModel


class OpenAIModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: Optional[str] = None
