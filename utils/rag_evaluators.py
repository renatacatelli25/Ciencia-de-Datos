"""Evaluadores RAG locales (framework LangSmith, sin dependencia de LangSmith)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from typing_extensions import Annotated

from utils.chunk_functions import recall_at_k, reciprocal_rank
from utils.local_evaluators import (
    evaluate_correctness_local,
    evaluate_groundedness_local,
    evaluate_relevance_local,
    evaluate_retrieval_relevance_local,
)

EvalMode = Literal["api", "local"]


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is grounded in the facts"]


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if retrieved documents are relevant to the question"]


CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

Grade based ONLY on factual accuracy relative to the ground truth answer.
It is OK if the student answer contains more information, as long as it is factually accurate.
If the student answer mentions the same product or key facts as the ground truth, count it as correct even if wording differs."""

RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER.

Ensure the student answer is relevant to the question and helps answer it.
If the student says no relevant products were found, mark as not relevant unless the question truly cannot be answered."""

GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER.

Ensure the student answer is grounded in the facts and does not hallucinate information outside them."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS (retrieved documents).

Mark relevant if the facts contain ANY keywords or semantic meaning related to the question.
It is OK if some unrelated information is present as long as part of the facts relates to the question."""


def _documents_to_string(documents: List[Dict[str, Any]]) -> str:
    parts = []
    for doc in documents:
        content = doc.get("page_content") or doc.get("descripcion") or ""
        name = doc.get("metadata", {}).get("product_name") or doc.get("product_name") or ""
        pid = doc.get("metadata", {}).get("product_id") or doc.get("product_id") or ""
        header = " - ".join(x for x in [name, pid] if x)
        parts.append(f"{header}\n{content}".strip())
    return "\n\n".join(parts)


def _judge(model: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model, temperature=0)


def evaluate_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    grader = _judge().with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
    prompt = (
        f"QUESTION: {inputs['question']}\n"
        f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    grade = grader.invoke([
        {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ])
    return {"score": bool(grade["correct"]), "explanation": grade["explanation"]}


def evaluate_relevance(inputs: dict, outputs: dict) -> Dict[str, Any]:
    grader = _judge().with_structured_output(RelevanceGrade, method="json_schema", strict=True)
    prompt = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grader.invoke([
        {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ])
    return {"score": bool(grade["relevant"]), "explanation": grade["explanation"]}


def evaluate_groundedness(inputs: dict, outputs: dict) -> Dict[str, Any]:
    grader = _judge().with_structured_output(GroundedGrade, method="json_schema", strict=True)
    doc_string = _documents_to_string(outputs.get("documents", []))
    prompt = f"FACTS:\n{doc_string}\n\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grader.invoke([
        {"role": "system", "content": GROUNDED_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ])
    return {"score": bool(grade["grounded"]), "explanation": grade["explanation"]}


def evaluate_retrieval_relevance(inputs: dict, outputs: dict) -> Dict[str, Any]:
    grader = _judge().with_structured_output(
        RetrievalRelevanceGrade, method="json_schema", strict=True
    )
    docs = outputs.get("retrieval_documents") or outputs.get("documents", [])
    doc_string = _documents_to_string(docs)
    prompt = f"QUESTION: {inputs['question']}\n\nFACTS:\n{doc_string or '(no documents retrieved)'}"
    grade = grader.invoke([
        {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ])
    return {"score": bool(grade["relevant"]), "explanation": grade["explanation"]}


def evaluate_deterministic_retrieval(
    outputs: dict,
    reference_outputs: dict,
    top_k: int = 5,
) -> Dict[str, Any]:
    relevant_ids = reference_outputs.get("relevant_product_ids", [])
    retrieved_ids = outputs.get("retrieved_ids") or [
        d.get("metadata", {}).get("product_id") or d.get("product_id")
        for d in outputs.get("documents", [])
    ]
    retrieved_ids = [pid for pid in retrieved_ids if pid][:top_k]
    return {
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "product_hit_at_k": recall_at_k(retrieved_ids, relevant_ids),
    }


def evaluate_product_hit_final(outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    relevant_ids = set(reference_outputs.get("relevant_product_ids", []))
    graded_ids = {
        d.get("metadata", {}).get("product_id") or d.get("product_id")
        for d in outputs.get("documents", [])
    }
    graded_ids.discard(None)
    reranked_ids = set(outputs.get("reranked_ids", []))
    hit_graded = bool(relevant_ids.intersection(graded_ids))
    hit_reranked = bool(relevant_ids.intersection(reranked_ids))
    mentioned = any(pid in (outputs.get("answer") or "") for pid in relevant_ids)
    return {
        "product_hit_final": int(mentioned or hit_graded),
        "product_hit_reranked": int(hit_reranked),
        "fallback": int(str(outputs.get("answer", "")).startswith("No encontré productos relevantes")),
    }


def evaluate_example(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    top_k: int = 5,
    skip_llm: bool = False,
    eval_mode: EvalMode = "api",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "question": inputs["question"],
        **evaluate_deterministic_retrieval(outputs, reference_outputs, top_k=top_k),
        **evaluate_product_hit_final(outputs, reference_outputs),
    }

    if skip_llm:
        return row

    if eval_mode == "local":
        correctness = evaluate_correctness_local(inputs, outputs, reference_outputs)
        relevance = evaluate_relevance_local(inputs, outputs)
        groundedness = evaluate_groundedness_local(inputs, outputs)
        retrieval_relevance = evaluate_retrieval_relevance_local(inputs, outputs)
    else:
        correctness = evaluate_correctness(inputs, outputs, reference_outputs)
        relevance = evaluate_relevance(inputs, outputs)
        groundedness = evaluate_groundedness(inputs, outputs)
        retrieval_relevance = evaluate_retrieval_relevance(inputs, outputs)

    row.update(
        {
            "correctness": int(correctness["score"]),
            "relevance": int(relevance["score"]),
            "groundedness": int(groundedness["score"]),
            "retrieval_relevance": int(retrieval_relevance["score"]),
            "correctness_explanation": correctness["explanation"],
            "relevance_explanation": relevance["explanation"],
            "groundedness_explanation": groundedness["explanation"],
            "retrieval_relevance_explanation": retrieval_relevance["explanation"],
            "eval_mode": eval_mode,
        }
    )
    return row


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    numeric_keys = [
        "correctness",
        "relevance",
        "groundedness",
        "retrieval_relevance",
        "recall_at_k",
        "mrr",
        "product_hit_at_k",
        "product_hit_final",
        "product_hit_reranked",
        "fallback",
    ]
    summary: Dict[str, float] = {}
    for key in numeric_keys:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = sum(values) / len(values)
    summary["n_examples"] = float(len(rows))
    return summary
