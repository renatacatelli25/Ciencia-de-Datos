import os
from pathlib import Path

from dotenv import load_dotenv

from providers.base import ModelProvider
from providers.mock import MockProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = (
    "Sos un asistente experto en reseñas de productos alimenticios de Amazon. "
    "Respondé en español, de forma clara y honesta."
)


def get_provider() -> ModelProvider:
    backend = os.getenv("CHAT_BACKEND", "mock").lower()

    if backend == "mock":
        return MockProvider()

    if backend in ("rag", "local", "api"):
        from providers.rag_provider import RagProvider

        return RagProvider(mode=backend if backend != "rag" else os.getenv("AGENT_MODE", "local"))

    raise ValueError(
        f"Backend desconocido: '{backend}'. "
        "Valores válidos: mock, rag, local, api"
    )


class ChatService:
    def __init__(self, provider: ModelProvider | None = None):
        self.provider = provider or get_provider()

    def reply(self, user_message: str, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.provider.generate(messages)
