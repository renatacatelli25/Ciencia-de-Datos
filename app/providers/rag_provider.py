import os
import sys
from pathlib import Path
from typing import Literal

from .base import ModelProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RuntimeMode = Literal["api", "local"]


class RagProvider(ModelProvider):
    """Backend RAG del proyecto: usa el agente de productos ya implementado."""

    def __init__(self, mode: str | RuntimeMode = "local"):
        self.mode: RuntimeMode = "api" if str(mode).lower() == "api" else "local"

    def generate(self, messages: list[dict]) -> str:
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if not user_msg.strip():
            return "No recibí una pregunta. Escribí qué producto o característica buscás."

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        try:
            from src.product_agent import run_agent

            result = run_agent(user_msg, mode=self.mode)
            return result["answer"]
        except Exception as exc:
            backend = os.getenv("CHAT_BACKEND", "rag")
            return (
                f"**Error al consultar el agente RAG** (`{backend}` / `{self.mode}`).\n\n"
                f"Detalle: {exc}\n\n"
                "Verificá que existan los índices Chroma y, en modo `api`, la `OPENAI_API_KEY`."
            )
