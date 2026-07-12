"""Target function para evaluación RAG local."""

from __future__ import annotations

from typing import Literal

from src.product_agent import configure_runtime, run_agent

RuntimeMode = Literal["api", "local"]


def _candidate_to_document(candidate: dict) -> dict:
    return {
        "page_content": candidate.get("descripcion", ""),
        "metadata": {
            "product_id": candidate.get("product_id", ""),
            "product_name": candidate.get("product_name", ""),
            "avg_score": candidate.get("avg_score"),
        },
        "product_id": candidate.get("product_id", ""),
        "product_name": candidate.get("product_name", ""),
        "descripcion": candidate.get("descripcion", ""),
    }


def agent_target(inputs: dict, mode: RuntimeMode = "api") -> dict:
    """Ejecuta el agente y devuelve el contrato esperado por los evaluadores."""
    configure_runtime(mode)
    state = run_agent(inputs["question"])
    graded = state.get("graded", [])
    candidates = state.get("candidates", [])
    documents = [_candidate_to_document(c) for c in graded]
    retrieval_documents = [_candidate_to_document(c) for c in candidates]

    return {
        "answer": state.get("answer", ""),
        "documents": documents,
        "retrieval_documents": retrieval_documents,
        "retrieved_ids": [c["product_id"] for c in candidates if c.get("product_id")],
        "reranked_ids": [c["product_id"] for c in state.get("reranked", []) if c.get("product_id")],
        "graded_ids": [c["product_id"] for c in graded if c.get("product_id")],
        "logs": state.get("logs", []),
        "runtime_mode": mode,
    }
