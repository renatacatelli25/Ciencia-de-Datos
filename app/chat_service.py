import os

from providers.base import ModelProvider
from providers.mock import MockProvider

SYSTEM_PROMPT = (
    "Sos un asistente experto en reseñas de productos alimenticios de Amazon. "
    "Respondé en español, de forma clara y honesta."
)


def get_provider() -> ModelProvider:
    backend = os.getenv("CHAT_BACKEND", "mock").lower()

    if backend == "mock":
        return MockProvider()

    # Futuro:
    # if backend == "openai":
    #     from providers.openai_provider import OpenAIProvider
    #     return OpenAIProvider()
    # if backend == "ml":
    #     from providers.ml_provider import MLProvider
    #     return MLProvider()
    # if backend == "rag":
    #     from providers.rag_provider import RagProvider
    #     return RagProvider()

    raise ValueError(
        f"Backend desconocido: '{backend}'. "
        "Valores válidos por ahora: mock"
    )


class ChatService:
    def __init__(self, provider: ModelProvider | None = None):
        self.provider = provider or get_provider()

    def reply(self, user_message: str, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.provider.generate(messages)
