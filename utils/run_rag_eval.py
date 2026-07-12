"""Corre la evaluación RAG local sobre el golden set."""

from __future__ import annotations

import sys
from pathlib import Path

# Importante: configurar el path del proyecto ANTES de importar utils/src.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

from src.rag_target import agent_target

try:
    from utils.rag_evaluators import evaluate_example, summarize_results
except ModuleNotFoundError:
    # Fallback si Python no resuelve el paquete utils (Windows / cwd distinto)
    from rag_evaluators import evaluate_example, summarize_results


def load_golden_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    golden_set_path: Path,
    output_dir: Path,
    limit: int | None = None,
    skip_llm: bool = False,
    top_k: int = 5,
    mode: str = "api",
) -> pd.DataFrame:
    examples = load_golden_set(golden_set_path)
    if limit is not None:
        examples = examples[:limit]

    if mode not in ("api", "local"):
        raise ValueError("mode debe ser 'api' o 'local'")

    print(f"Modo de ejecución: {mode} ({'sin API' if mode == 'local' else 'OpenAI'})")

    rows = []
    for idx, example in enumerate(examples, start=1):
        question = example["inputs"]["question"]
        print(f"[{idx}/{len(examples)}] Evaluando: {question[:70]}...")
        outputs = agent_target(example["inputs"], mode=mode)
        row = evaluate_example(
            inputs=example["inputs"],
            outputs=outputs,
            reference_outputs=example["outputs"],
            top_k=top_k,
            skip_llm=skip_llm,
            eval_mode=mode,
        )
        row["answer"] = outputs["answer"]
        row["graded_ids"] = ",".join(outputs.get("graded_ids", []))
        row["runtime_mode"] = mode
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = summarize_results(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"agent_eval_{stamp}.csv"
    summary_path = output_dir / f"agent_eval_summary_{stamp}.json"

    df.to_csv(detail_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Resumen ===")
    for key, value in summary.items():
        if key == "n_examples":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.3f}")

    print(f"\nDetalle guardado en: {detail_path}")
    print(f"Resumen guardado en: {summary_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluación RAG local del agente de productos")
    parser.add_argument(
        "--golden-set",
        default=str(PROJECT_ROOT / "utils" / "golden_set_rag.json"),
        help="Ruta al dataset de evaluación",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directorio donde guardar resultados",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluar solo las primeras N preguntas")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Solo métricas determinísticas (recall@k, mrr, product_hit, fallback)",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api=OpenAI (agente + evaluadores LLM) | local=sin API (MiniLM + heurísticas)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k para recall/MRR")
    args = parser.parse_args()

    run_evaluation(
        golden_set_path=Path(args.golden_set),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        skip_llm=args.skip_llm,
        top_k=args.top_k,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
