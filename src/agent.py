from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - fallback when langgraph is unavailable
    END = "__end__"
    StateGraph = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when openai is unavailable
    OpenAI = None

import pandas as pd


class AgentState(TypedDict):
    pregunta: str
    top_k: int
    score_minimo: Optional[float]
    intentos: int
    max_intentos: int
    productos: List[Dict[str, Any]]
    verificacion: Optional[Dict[str, Any]]
    respuesta_final: Optional[Dict[str, Any]]


VERIFICADOR_SYSTEM_PROMPT = """Sos un verificador de calidad de búsqueda.
Se te va a dar la pregunta de un usuario y una lista de productos recuperados
(con su descripción). Tu trabajo es decidir si ALGUNO de los productos
responde razonablemente la pregunta.

Respondé SOLO con un JSON, sin texto adicional, con este formato exacto:
{"match": true/false, "mejor_producto_id": "<id o null>", "razon": "<explicación breve>"}
"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "src" / "data" / "Reviews.csv"
DESCRIPTIONS_PATH = PROJECT_ROOT / "descripciones_checkpoint.csv"


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_reviews_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset de reviews en {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["ProfileName"] = df["ProfileName"].fillna("UNKNOWN_USER")
    df["Summary"] = df["Summary"].fillna("NO SUMMARY")
    if "Text" not in df.columns:
        df["Text"] = ""
    if "helpfulness_ratio" not in df.columns:
        df["helpfulness_ratio"] = (
            df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"].replace(0, pd.NA)
        ).fillna(0)
    if "full_text" not in df.columns:
        df["full_text"] = (
            df["Summary"].astype(str) + ". " + df["Text"].astype(str)
        ).str.strip()
    return df


def _load_product_catalog() -> List[Dict[str, Any]]:
    df = _load_reviews_df()
    if DESCRIPTIONS_PATH.exists():
        try:
            desc_df = pd.read_csv(DESCRIPTIONS_PATH)
            if {"product_id", "descripcion"}.issubset(desc_df.columns):
                catalog = []
                for _, row in desc_df.iterrows():
                    catalog.append(
                        {
                            "product_id": row["product_id"],
                            "descripcion": row.get("descripcion", ""),
                            "nombre": row.get("product_name_inferred") or row.get("product_name") or "Producto",
                        }
                    )
                if catalog:
                    return catalog
        except Exception:
            pass

    groups = []
    for product_id, group in df.groupby("ProductId", sort=False):
        group = group.sort_values("helpfulness_ratio", ascending=False)
        top_reviews = group.head(3)
        review_text = " ".join(top_reviews["full_text"].fillna("").astype(str))

        nombre = "Producto"
        if "product_name_inferred" in df.columns:
            nombre = df.loc[df["ProductId"] == product_id, "product_name_inferred"].dropna().iloc[0] if not df.loc[df["ProductId"] == product_id, "product_name_inferred"].empty else "Producto"
        else:
            tokens = [t for t in _normalize_text(review_text).split() if len(t) > 2]
            for token in tokens:
                if token in {"cookies", "dog", "treats", "oil", "coconut", "honey", "chips", "chocolate", "mayonnaise", "jerky", "dates", "cornbread", "mustard", "tea", "mix", "cheese", "sauce", "cereal", "bar", "coffee", "candy", "bread", "soup", "seasoning", "granola"}:
                    nombre = token.capitalize()
                    break

        descripcion = " ".join(
            [
                f"{nombre} con puntuación promedio {group['Score'].mean():.1f}/5",
                f"y {len(group)} reseñas disponibles.",
            ]
        )
        groups.append({"product_id": product_id, "descripcion": descripcion, "nombre": nombre})

    return groups


_PRODUCT_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


def cargar_catalogo_productos() -> List[Dict[str, Any]]:
    global _PRODUCT_CATALOG_CACHE
    if _PRODUCT_CATALOG_CACHE is None:
        _PRODUCT_CATALOG_CACHE = _load_product_catalog()
    return _PRODUCT_CATALOG_CACHE


def buscar_productos(pregunta: str, top_k: int = 5, score_minimo: Optional[float] = None) -> List[Dict[str, Any]]:
    catalogo = cargar_catalogo_productos()
    if not catalogo:
        return []

    pregunta_norm = _normalize_text(pregunta)
    pregunta_tokens = [t for t in pregunta_norm.split() if len(t) > 2]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in catalogo:
        texto = " ".join(
            [item.get("nombre", ""), item.get("descripcion", ""), item.get("product_id", "")]
        )
        texto_norm = _normalize_text(texto)
        texto_tokens = set(texto_norm.split())
        overlap = sum(1 for t in pregunta_tokens if t in texto_tokens)
        if overlap == 0 and pregunta_tokens:
            overlap = 0.0
        score = overlap + (0.1 if any(t in texto_norm for t in pregunta_tokens) else 0.0)
        if score_minimo is not None and score < score_minimo:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: (-x[0], x[1]["product_id"]))
    return [item for _, item in scored[:max(1, top_k)]]


def nodo_retrieve(state: AgentState) -> AgentState:
    productos = buscar_productos(
        state["pregunta"], top_k=state["top_k"], score_minimo=state["score_minimo"]
    )
    return {**state, "productos": productos}


def nodo_verify(state: AgentState) -> AgentState:
    if not state["productos"]:
        verificacion = {"match": False, "mejor_producto_id": None, "razon": "No se encontraron productos relevantes."}
        return {**state, "verificacion": verificacion}

    if OpenAI is not None:
        try:
            client_openai = OpenAI()
            lista_productos = "\n".join(
                f"- {p['product_id']}: {p['descripcion']}" for p in state["productos"]
            )
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": VERIFICADOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Pregunta: {state['pregunta']}\n\nProductos recuperados:\n{lista_productos}"},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            verificacion = json.loads(response.choices[0].message.content)
            return {**state, "verificacion": verificacion}
        except Exception:
            pass

    pregunta_norm = _normalize_text(state["pregunta"])
    mejores = state["productos"][0]
    texto = _normalize_text(f"{mejores.get('nombre', '')} {mejores.get('descripcion', '')}")
    match = any(token in texto for token in pregunta_norm.split() if len(token) > 2)
    verificacion = {
        "match": bool(match),
        "mejor_producto_id": mejores.get("product_id"),
        "razon": "Coincidencia por solapamiento lexical entre la pregunta y la descripción del producto." if match else "No se encontró un solapamiento claro entre la pregunta y los productos recuperados.",
    }
    return {**state, "verificacion": verificacion}


def nodo_retry(state: AgentState) -> AgentState:
    return {**state, "top_k": state["top_k"] * 2, "intentos": state["intentos"] + 1}


def nodo_give_up(state: AgentState) -> AgentState:
    return {
        **state,
        "respuesta_final": {
            "match": False,
            "mejor_producto_id": None,
            "razon": f"No se encontró un match aceptable tras {state['intentos']} intento(s).",
        },
    }


def decidir_siguiente_paso(state: AgentState) -> str:
    if state["verificacion"]["match"]:
        return "responder"
    if state["intentos"] < state["max_intentos"]:
        return "reintentar"
    return "rendirse"


class SimpleAgentRunner:
    def invoke(self, state: AgentState) -> AgentState:
        current = state
        current = nodo_retrieve(current)
        current = nodo_verify(current)
        while not current["verificacion"].get("match", False) and current["intentos"] < current["max_intentos"]:
            current = nodo_retry(current)
            current = nodo_retrieve(current)
            current = nodo_verify(current)
        if current["verificacion"].get("match", False):
            return current
        return nodo_give_up(current)


if StateGraph is not None:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodo_retrieve)
    graph.add_node("verify", nodo_verify)
    graph.add_node("retry", nodo_retry)
    graph.add_node("give_up", nodo_give_up)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify",
        decidir_siguiente_paso,
        {"responder": END, "reintentar": "retry", "rendirse": "give_up"},
    )
    graph.add_edge("retry", "retrieve")
    graph.add_edge("give_up", END)
    agente_verificador = graph.compile()
else:
    agente_verificador = SimpleAgentRunner()


def buscar_con_verificacion(pregunta: str, top_k: int = 5, score_minimo: Optional[float] = None, max_intentos: int = 2) -> Dict[str, Any]:
    estado_inicial: AgentState = {
        "pregunta": pregunta,
        "top_k": top_k,
        "score_minimo": score_minimo,
        "intentos": 0,
        "max_intentos": max_intentos,
        "productos": [],
        "verificacion": None,
        "respuesta_final": None,
    }
    estado_final = agente_verificador.invoke(estado_inicial)
    verificacion = estado_final.get("respuesta_final") or estado_final["verificacion"]
    return {
        "productos": estado_final["productos"],
        "verificacion": verificacion,
        "intentos_usados": estado_final["intentos"],
    }


def evaluar_agente_en_golden_set(golden_set: Optional[List[Dict[str, Any]]] = None, top_k: int = 5, max_intentos: int = 2) -> pd.DataFrame:
    if golden_set is None:
        golden_set = [
            {"question": "dog treats for coughing", "relevant_product_ids": ["B002QWP89S"]},
            {"question": "coconut oil for popcorn", "relevant_product_ids": ["B001EO5Q64"]},
            {"question": "dark chocolate smooth", "relevant_product_ids": ["B0009Z9FFW"]},
        ]

    resultados_metricas: List[Dict[str, Any]] = []
    for item in golden_set:
        resultado = buscar_con_verificacion(item.get("question", ""), top_k=top_k, max_intentos=max_intentos)
        ids_recuperados = [p["product_id"] for p in resultado["productos"]]
        hubo_producto_correcto = any(pid in item.get("relevant_product_ids", []) for pid in ids_recuperados)
        agente_dijo_match = bool(resultado["verificacion"].get("match", False))

        resultados_metricas.append(
            {
                "pregunta": item.get("question", ""),
                "producto_correcto_en_resultados": hubo_producto_correcto,
                "agente_dice_match": agente_dijo_match,
                "acierto_agente": hubo_producto_correcto == agente_dijo_match,
                "intentos_usados": resultado["intentos_usados"],
            }
        )

    return pd.DataFrame(resultados_metricas)


if __name__ == "__main__":
    ejemplo = buscar_con_verificacion("dog treats for older dogs", top_k=3, max_intentos=2)
    print(json.dumps(ejemplo, indent=2, ensure_ascii=False))