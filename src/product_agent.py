"""Agente RAG de productos: extract_filters → retrieve → rerank → grade → generate."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

import chromadb
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger("agente_productos")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RuntimeMode = Literal["api", "local"]

CHROMA_CONFIG = {
    "api": {
        "path": PROJECT_ROOT / "chroma_db_productos_openai",
        "collection": "product_search_openai",
        "embedding": "openai",
    },
    "local": {
        "path": PROJECT_ROOT / "chroma_db_productos",
        "collection": "product_search",
        "embedding": "sentence-transformers",
    },
}

EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVE_TOP_K = 15
RERANK_TOP_N = 5
LOCAL_GRADE_MIN_SCORE = -2.0

FALLBACK_ANSWER_PREFIX = "No encontré productos relevantes"


class ProductFilters(TypedDict, total=False):
    brand: Optional[str]
    category: Optional[str]
    price_max: Optional[float]
    rating_min: Optional[float]


class ProductCandidate(TypedDict, total=False):
    product_id: str
    product_name: str
    descripcion: str
    avg_score: float
    rerank_score: float


class AgentState(TypedDict):
    question: str
    filters: ProductFilters
    candidates: List[ProductCandidate]
    reranked: List[ProductCandidate]
    graded: List[ProductCandidate]
    answer: str
    logs: List[str]


class ExtractedFilters(BaseModel):
    brand: Optional[str] = Field(None, description="Marca mencionada explícitamente")
    category: Optional[str] = Field(None, description="Categoría o tipo de producto")
    price_max: Optional[float] = Field(None, description="Precio máximo mencionado")
    rating_min: Optional[float] = Field(None, description="Rating mínimo (1-5)")


class GradeResult(BaseModel):
    relevant: bool = Field(description="True si el producto es relevante para la pregunta")


@tool(args_schema=ExtractedFilters)
def extract_product_filters(
    brand: Optional[str] = None,
    category: Optional[str] = None,
    price_max: Optional[float] = None,
    rating_min: Optional[float] = None,
) -> dict:
    """Registra filtros estructurados detectados en la pregunta."""
    return {"brand": brand, "category": category, "price_max": price_max, "rating_min": rating_min}


FILTER_EXTRACTION_PROMPT = """Analizá la pregunta del usuario sobre productos.

Si menciona explícitamente una marca, categoría/tipo de producto, un precio máximo o un rating mínimo, llamá a la tool `extract_product_filters` con esos valores. Dejá en None lo que no se mencione. Si no hay ningún filtro explícito, no llames a la tool."""

GRADER_PROMPT = """Evaluá si la descripción del producto de abajo es relevante para responder la pregunta del usuario.

No hace falta que sea una coincidencia perfecta ni que mencione las palabras exactas, pero sí que esté relacionado con lo que se pregunta (atributo, uso, problema, tipo de producto, etc). Si el producto no tiene nada que ver, marcalo como no relevante."""

AGENT_SYSTEM_PROMPT = """Sos un asistente que recomienda productos a partir de descripciones generadas de reviews reales.

Reglas:
1. Respondé SOLO en base a los productos que te paso en el contexto. Si no hay ninguno, decilo explícitamente y no inventes productos ni datos.
2. Para cada producto que menciones: nombralo y justificá en una frase por qué encaja con la pregunta, citando algo concreto de su descripción.
3. Ordená de más a menos relevante.
4. Si se detectaron filtros (marca, categoría, rating mínimo) y no hay resultados que los cumplan, decilo explícitamente en vez de mostrar productos que no cumplen.
5. Nunca inventes precio: el dataset no tiene esa información. Si preguntan por precio, aclará que no está disponible.
6. Tono neutral, en español, entre 2 y 5 oraciones.
"""


_agent: Any = None
_runtime_mode: RuntimeMode = "api"
_llm: Optional[ChatOpenAI] = None
_openai_client: Optional[OpenAI] = None
_product_index: Any = None
_cross_encoder: Optional[CrossEncoder] = None
_local_embedding_model: Optional[SentenceTransformer] = None
_grader_llm: Any = None


def get_runtime_mode() -> RuntimeMode:
    return _runtime_mode


def configure_runtime(mode: RuntimeMode = "api") -> None:
    """Cambia entre modo API (OpenAI) y modo local (sin costo de API)."""
    global _runtime_mode, _agent, _product_index, _local_embedding_model
    if mode not in ("api", "local"):
        raise ValueError("mode debe ser 'api' o 'local'")
    if mode == _runtime_mode:
        return
    _runtime_mode = mode
    _agent = None
    _product_index = None
    _local_embedding_model = None
    logger.info("Modo de ejecución del agente: %s", mode)


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return _llm


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_product_index():
    global _product_index
    if _product_index is None:
        cfg = CHROMA_CONFIG[_runtime_mode]
        chroma_client = chromadb.PersistentClient(path=str(cfg["path"]))
        _product_index = chroma_client.get_collection(cfg["collection"])
    return _product_index


def _get_local_embedding_model() -> SentenceTransformer:
    global _local_embedding_model
    if _local_embedding_model is None:
        _local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_embedding_model


def _embed_question(question: str) -> List[float]:
    if _runtime_mode == "local":
        model = _get_local_embedding_model()
        return model.encode([question], show_progress_bar=False).tolist()[0]

    openai_client = _get_openai_client()
    return openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[question]
    ).data[0].embedding


def _extract_filters_local(question: str) -> ProductFilters:
    filtros: ProductFilters = {}
    q = question.lower()

    rating_match = re.search(
        r"(?:rating|puntuaci[oó]n|score)\s*(?:above|over|>|arriba de|de al menos|min(?:imo)?)?\s*(\d+(?:\.\d+)?)",
        q,
    )
    if rating_match:
        filtros["rating_min"] = float(rating_match.group(1))

    price_match = re.search(r"(?:precio|price)\s*(?:max|m[aá]ximo|menor a|<|under)?\s*\$?\s*(\d+(?:\.\d+)?)", q)
    if price_match:
        filtros["price_max"] = float(price_match.group(1))

    return filtros


def _generate_local_answer(question: str, graded: List[ProductCandidate], filters: ProductFilters) -> str:
    if not graded:
        answer = f"{FALLBACK_ANSWER_PREFIX} para esa consulta"
        answer += f" con los filtros detectados ({filters})." if filters else "."
        return answer

    top = graded[0]
    name = top.get("product_name") or "Producto"
    desc = (top.get("descripcion") or "").strip()
    snippet = desc.split(". ")[0].strip()
    if snippet and not snippet.endswith("."):
        snippet += "."

    return (
        f"La opción más relevante es **{name}**. "
        f"{snippet} "
        f"Se seleccionó por relevancia semántica local (CrossEncoder + índice MiniLM)."
    )


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return _cross_encoder


def _get_grader_llm():
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = _get_llm().with_structured_output(GradeResult)
    return _grader_llm


def log_step(state: AgentState, node: str, message: str) -> List[str]:
    entry = f"[{node}] {message}"
    logger.info(entry)
    return state.get("logs", []) + [entry]


def build_chroma_filters(filters: ProductFilters):
    where = None
    if filters.get("rating_min") is not None:
        where = {"avg_score": {"$gte": float(filters["rating_min"])}}

    contains_clauses = []
    if filters.get("brand"):
        contains_clauses.append({"$contains": filters["brand"]})
    if filters.get("category"):
        contains_clauses.append({"$contains": filters["category"]})

    where_document = None
    if len(contains_clauses) == 1:
        where_document = contains_clauses[0]
    elif len(contains_clauses) > 1:
        where_document = {"$and": contains_clauses}

    return where, where_document


def extract_filters_node(state: AgentState) -> AgentState:
    if _runtime_mode == "local":
        filtros = _extract_filters_local(state["question"])
        logs = log_step(
            state,
            "extract_filters",
            f"[local] Filtros heurísticos: {filtros or 'ninguno'}",
        )
        return {**state, "filters": filtros, "logs": logs}

    llm_with_tools = _get_llm().bind_tools([extract_product_filters])
    respuesta = llm_with_tools.invoke([
        {"role": "system", "content": FILTER_EXTRACTION_PROMPT},
        {"role": "user", "content": state["question"]},
    ])

    filtros: ProductFilters = {}
    if respuesta.tool_calls:
        args = respuesta.tool_calls[0]["args"]
        filtros = {k: v for k, v in args.items() if v not in (None, "", [])}

    logs = log_step(state, "extract_filters", f"Filtros detectados: {filtros or 'ninguno'}")
    return {**state, "filters": filtros, "logs": logs}


def retrieve_node(state: AgentState) -> AgentState:
    filters = state.get("filters", {})
    where, where_document = build_chroma_filters(filters)
    logs = list(state.get("logs", []))
    product_index = _get_product_index()

    if filters.get("price_max") is not None:
        logs = log_step(
            {**state, "logs": logs},
            "retrieve",
            f"Se pidió price_max={filters['price_max']} pero el dataset no tiene precios: se ignora ese filtro.",
        )

    query_embedding = _embed_question(state["question"])

    def _query(with_filters: bool):
        kwargs = {"query_embeddings": [query_embedding], "n_results": RETRIEVE_TOP_K}
        if with_filters and where:
            kwargs["where"] = where
        if with_filters and where_document:
            kwargs["where_document"] = where_document
        return product_index.query(**kwargs)

    resultados = _query(with_filters=True)
    candidatos = [
        {
            "product_id": pid,
            "product_name": meta.get("product_name", ""),
            "descripcion": doc,
            "avg_score": meta.get("avg_score", 0.0),
        }
        for pid, doc, meta in zip(
            resultados["ids"][0], resultados["documents"][0], resultados["metadatas"][0]
        )
    ]

    if not candidatos and (where or where_document):
        logs = log_step(
            {**state, "logs": logs},
            "retrieve",
            "0 resultados con filtros aplicados -> reintentando búsqueda semántica sin filtros.",
        )
        resultados = _query(with_filters=False)
        candidatos = [
            {
                "product_id": pid,
                "product_name": meta.get("product_name", ""),
                "descripcion": doc,
                "avg_score": meta.get("avg_score", 0.0),
            }
            for pid, doc, meta in zip(
                resultados["ids"][0], resultados["documents"][0], resultados["metadatas"][0]
            )
        ]

    logs = log_step(
        {**state, "logs": logs},
        "retrieve",
        f"{len(candidatos)} candidatos (where={where}, where_document={where_document})",
    )
    return {**state, "candidates": candidatos, "logs": logs}


def rerank_node(state: AgentState) -> AgentState:
    candidatos = state.get("candidates", [])
    if not candidatos:
        logs = log_step(state, "rerank", "No hay candidatos para reordenar.")
        return {**state, "reranked": [], "logs": logs}

    cross_encoder = _get_cross_encoder()
    pares = [(state["question"], c["descripcion"]) for c in candidatos]
    scores = cross_encoder.predict(pares)

    for c, score in zip(candidatos, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidatos, key=lambda c: c["rerank_score"], reverse=True)[:RERANK_TOP_N]
    resumen = ", ".join(f"{c['product_name']} ({c['rerank_score']:.2f})" for c in reranked)
    logs = log_step(state, "rerank", f"Top {RERANK_TOP_N} tras CrossEncoder: {resumen}")
    return {**state, "reranked": reranked, "logs": logs}


def grade_node(state: AgentState) -> AgentState:
    candidatos = state.get("reranked", [])

    if _runtime_mode == "local":
        graded = [c for c in candidatos if c.get("rerank_score", -999) >= LOCAL_GRADE_MIN_SCORE]
        if not graded and candidatos:
            graded = [candidatos[0]]
        logs = log_step(
            state,
            "grade",
            f"[local] {len(graded)} de {len(candidatos)} candidatos pasaron por score del CrossEncoder.",
        )
        return {**state, "graded": graded, "logs": logs}

    graded = []
    grader_llm = _get_grader_llm()

    for c in candidatos:
        resultado = grader_llm.invoke([
            {"role": "system", "content": GRADER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Pregunta: {state['question']}\n\n"
                    f"Producto: {c['product_name']}\nDescripción: {c['descripcion']}"
                ),
            },
        ])
        if resultado.relevant:
            graded.append(c)

    logs = log_step(state, "grade", f"{len(graded)} de {len(candidatos)} candidatos marcados como relevantes.")
    return {**state, "graded": graded, "logs": logs}


def generate_node(state: AgentState) -> AgentState:
    graded = state.get("graded", [])
    filters = state.get("filters", {})

    if not graded:
        answer = f"{FALLBACK_ANSWER_PREFIX} para esa consulta"
        answer += f" con los filtros detectados ({filters})." if filters else "."
        logs = log_step(state, "generate", "Sin candidatos relevantes -> respuesta de fallback.")
        return {**state, "answer": answer, "logs": logs}

    if _runtime_mode == "local":
        answer = _generate_local_answer(state["question"], graded, filters)
        logs = log_step(state, "generate", "[local] Respuesta generada por plantilla.")
        return {**state, "answer": answer, "logs": logs}

    contexto = "\n".join(
        f"- {p['product_name']} (id {p['product_id']}, rating prom. {p.get('avg_score', 'N/D')}): {p['descripcion']}"
        for p in graded
    )
    filtros_txt = f"\nFiltros detectados en la pregunta: {filters}" if filters else ""

    respuesta = _get_llm().invoke([
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta: {state['question']}{filtros_txt}\n\nProductos disponibles:\n{contexto}"},
    ])

    logs = log_step(state, "generate", "Respuesta generada.")
    return {**state, "answer": respuesta.content, "logs": logs}


def _build_agent():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("extract_filters", extract_filters_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("grade", grade_node)
    graph_builder.add_node("generate", generate_node)

    graph_builder.add_edge(START, "extract_filters")
    graph_builder.add_edge("extract_filters", "retrieve")
    graph_builder.add_edge("retrieve", "rerank")
    graph_builder.add_edge("rerank", "grade")
    graph_builder.add_edge("grade", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()


def get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def run_agent(question: str, mode: RuntimeMode | None = None) -> AgentState:
    if mode is not None:
        configure_runtime(mode)
    estado_inicial: AgentState = {
        "question": question,
        "filters": {},
        "candidates": [],
        "reranked": [],
        "graded": [],
        "answer": "",
        "logs": [],
    }
    return get_agent().invoke(estado_inicial)


def responder_pregunta(pregunta: str, verbose: bool = False) -> str:
    resultado = run_agent(pregunta)
    if verbose:
        print("\n".join(resultado["logs"]))
        print()
    return resultado["answer"]
