from .base import ModelProvider

KEYWORD_RESPONSES = {
    "score": (
        "Según el EDA preliminar, el score promedio ronda **4.18 estrellas** "
        "y hay un sesgo positivo: la mayoría de reseñas son 4 o 5 estrellas."
    ),
    "producto": (
        "Los productos con más reseñas concentran miles de opiniones. "
        "En el análisis exploratorio se identificaron los top 20 por `ProductId`."
    ),
    "usuario": (
        "La actividad de usuarios es muy desigual: pocos usuarios escriben "
        "muchas reseñas y la mayoría escribe pocas."
    ),
    "helpfulness": (
        "Muchas reseñas no tienen votos de helpfulness (`Denominator = 0`). "
        "Cuando hay votos, el ratio suele ser alto en reseñas positivas."
    ),
    "tiempo": (
        "Las reseñas cubren un rango amplio, desde 1999 hasta 2012, "
        "con picos de actividad en ciertos años."
    ),
    "longitud": (
        "La longitud de las reseñas varía mucho. En el EDA se analizó "
        "`review_length` para ver si textos más largos reciben más votos útiles."
    ),
}


class MockProvider(ModelProvider):
    """Proveedor simulado para desarrollar la UI sin LLM ni modelo entrenado."""

    def generate(self, messages: list[dict]) -> str:
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        normalized = user_msg.lower()

        for keyword, response in KEYWORD_RESPONSES.items():
            if keyword in normalized:
                return (
                    f"**[MOCK]** Respuesta simulada sobre «{keyword}».\n\n"
                    f"{response}\n\n"
                    f"_Tu pregunta fue:_ «{user_msg}»"
                )

        return (
            "**[MOCK]** Backend simulado activo.\n\n"
            "Todavía no hay LLM ni modelo conectado. Probá preguntar sobre: "
            "**score**, **producto**, **usuario**, **helpfulness**, **tiempo** o **longitud**.\n\n"
            f"_Tu pregunta fue:_ «{user_msg}»"
        )
