"""Heurísticas locales para evaluación RAG (sin API)."""

from __future__ import annotations

import re
from typing import Any, Dict, Set


def _tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _is_fallback(answer: str) -> bool:
    return str(answer or "").startswith("No encontré productos relevantes")


def evaluate_correctness_local(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> Dict[str, Any]:
    answer = outputs.get("answer", "")
    reference = reference_outputs.get("answer", "")
    relevant_ids = reference_outputs.get("relevant_product_ids", [])

    if _is_fallback(answer):
        return {"score": False, "explanation": "La respuesta es un fallback sin producto."}

    if any(pid in answer for pid in relevant_ids):
        return {"score": True, "explanation": "La respuesta menciona un product_id gold."}

    graded_ids = set(outputs.get("graded_ids", []))
    if relevant_ids and graded_ids.intersection(relevant_ids):
        return {"score": True, "explanation": "El producto gold llegó al contexto final."}

    overlap = len(_tokens(answer) & _tokens(reference)) / max(len(_tokens(reference)), 1)
    score = overlap >= 0.12
    return {
        "score": score,
        "explanation": f"Overlap léxico con referencia: {overlap:.2f}.",
    }


def evaluate_relevance_local(inputs: dict, outputs: dict) -> Dict[str, Any]:
    answer = outputs.get("answer", "")
    if _is_fallback(answer):
        return {"score": False, "explanation": "No respondió con un producto relevante."}

    question_tokens = _tokens(inputs.get("question", ""))
    answer_tokens = _tokens(answer)
    overlap = len(question_tokens & answer_tokens)
    score = overlap >= 1 or len(answer_tokens) >= 8
    return {
        "score": score,
        "explanation": f"Solapamiento pregunta-respuesta: {overlap} tokens.",
    }


def evaluate_groundedness_local(inputs: dict, outputs: dict) -> Dict[str, Any]:
    answer = outputs.get("answer", "")
    if _is_fallback(answer):
        return {"score": True, "explanation": "Fallback sin afirmaciones adicionales."}

    docs = outputs.get("documents") or []
    if not docs:
        return {"score": False, "explanation": "Hay respuesta pero sin documentos de contexto."}

    context_tokens: Set[str] = set()
    for doc in docs:
        context_tokens |= _tokens(doc.get("page_content") or doc.get("descripcion") or "")

    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return {"score": False, "explanation": "Respuesta vacía."}

    grounded_ratio = len(answer_tokens & context_tokens) / len(answer_tokens)
    score = grounded_ratio >= 0.25
    return {
        "score": score,
        "explanation": f"{grounded_ratio:.0%} de tokens de la respuesta aparecen en el contexto.",
    }


def evaluate_retrieval_relevance_local(inputs: dict, outputs: dict) -> Dict[str, Any]:
    docs = outputs.get("retrieval_documents") or outputs.get("documents") or []
    if not docs:
        return {"score": False, "explanation": "No se recuperaron documentos."}

    question_tokens = _tokens(inputs.get("question", ""))
    if not question_tokens:
        return {"score": False, "explanation": "Pregunta vacía."}

    best_overlap = 0
    for doc in docs:
        doc_tokens = _tokens(doc.get("page_content") or doc.get("descripcion") or "")
        best_overlap = max(best_overlap, len(question_tokens & doc_tokens))

    score = best_overlap >= 1
    return {
        "score": score,
        "explanation": f"Máximo solapamiento pregunta-documento: {best_overlap} tokens.",
    }
