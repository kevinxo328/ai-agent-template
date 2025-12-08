from langfuse import Langfuse

from agents.demo.env import env

langfuse = Langfuse(
    secret_key=env.LANGFUSE_SECRET_KEY,
    public_key=env.LANGFUSE_PUBLIC_KEY,
    host=env.LANGFUSE_HOST,
)
