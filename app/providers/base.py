from abc import ABC, abstractmethod


class ModelProvider(ABC):
    """Contrato que todos los backends de IA deben cumplir."""

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """
        Recibe el historial de mensajes y devuelve la respuesta del asistente.

        Formato de cada mensaje: {"role": "user"|"assistant"|"system", "content": "..."}
        """
