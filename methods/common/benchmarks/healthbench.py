#!/usr/bin/env python3
"""Local HealthBench runner with direct, BM25, and MedCPT baselines."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.BM25.retriever import BM25Retriever, DEFAULT_BM25_CORPORA
from methods.MedCPT.retriever import DEFAULT_MEDCPT_CORPORA, MedCPTRetriever
from methods.common.io import make_run_dir, write_json
from methods.common.llm import call_chat_model
from methods.common.prompt import format_retrieved_docs
from methods.common.retrieval import DEFAULT_DB_DIR, parse_corpus_list, summarize_retrieved_docs


DEFAULT_SUBSET_PATHS = {
    "healthbench": PROJECT_ROOT / "data" / "HealthBench" / "healthbench.jsonl",
    "hard": PROJECT_ROOT / "data" / "HealthBench" / "hard.jsonl",
    "consensus": PROJECT_ROOT / "data" / "HealthBench" / "consensus.jsonl",
}


@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: List[str]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RubricItem":
        return cls(
            criterion=str(payload.get("criterion", "")).strip(),
            points=float(payload.get("points", 0.0)),
            tags=[str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```json\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    for candidate in (stripped, stripped[stripped.find("{") : stripped.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def _load_jsonl(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def _resolve_subset_name(data_path: Path, subset: str) -> str:
    if subset:
        return str(subset).strip().lower()
    return data_path.stem.lower()


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in messages or []:
        role = str(row.get("role", "user")).strip().lower() or "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(row.get("content", "")).strip()
        if content:
            rows.append({"role": role, "content": content})
    return rows


def _latest_user_text(messages: Sequence[Dict[str, str]]) -> str:
    for row in reversed(messages):
        if row.get("role") == "user" and row.get("content"):
            return str(row["content"])
    return "\n\n".join(str(row.get("content", "")) for row in messages if row.get("content"))


def _with_retrieved_docs(messages: Sequence[Dict[str, str]], snippets: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    docs = format_retrieved_docs(snippets)
    augmented = list(messages)
    insert_at = 0
    while insert_at < len(augmented) and augmented[insert_at].get("role") == "system":
        insert_at += 1
    augmented.insert(
        insert_at,
        {
            "role": "system",
            "content": (
                "Retrieved medical references:\n"
                f"{docs}\n\n"
                "Use these references when they are helpful and answer the user directly."
            ),
        },
    )
    return augmented


def _build_retriever(args: argparse.Namespace, corpora: Sequence[str]) -> Tuple[Any, str]:
    if args.method == "direct":
        return None, "none"
    if args.method == "bm25":
        return BM25Retriever(db_dir=args.db_dir, corpora=corpora), "BM25"
    if args.method == "medcpt":
        return MedCPTRetriever(db_dir=args.db_dir, corpora=corpora, device=args.device), "MedCPT"
    raise ValueError(f"Unsupported method: {args.method}")


def _judge_rubrics(
    *,
    prompt_messages: Sequence[Dict[str, str]],
    response_text: str,
    rubrics: Sequence[RubricItem],
    judge_model: str,
    judge_timeout_s: int,
    judge_azure_suffix: str,
) -> List[Dict[str, Any]]:
    rubric_lines = [
        f"{idx}. [{rubric.points}] {rubric.criterion}"
        for idx, rubric in enumerate(rubrics)
    ]
    conversation_rows = list(prompt_messages) + [{"role": "assistant", "content": response_text}]
    conversation = "\n\n".join(f"{row['role']}: {row['content']}" for row in conversation_rows)
    user = (
        "Score the last assistant response against all rubric items.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Rubrics:\n"
        f"{chr(10).join(rubric_lines)}\n\n"
        "Rules:\n"
        "- For each rubric item, set criteria_met to true only if the response clearly satisfies it.\n"
        '- If a rubric says "for example", "including", or "such as", the response does not need every listed example.\n'
        "- Be strict about missing medical details.\n\n"
        'Return JSON only with this shape: {"results":[{"index":0,"criteria_met":true,"explanation":"..."}, ...]}'
    )
    messages = [
        {"role": "system", "content": "You are a strict medical benchmark judge."},
        {"role": "user", "content": user},
    ]

    last_response = ""
    for _ in range(3):
        last_response = call_chat_model(
            messages,
            provider="azure",
            model=judge_model,
            timeout_s=judge_timeout_s,
            azure_suffix=judge_azure_suffix,
        )
        payload = _parse_json_object(last_response)
        rows = payload.get("results")
        if not isinstance(rows, list) or len(rows) != len(rubrics):
            continue

        normalized: List[Dict[str, Any]] = []
        valid = True
        for idx, row in enumerate(rows):
            value = row.get("criteria_met")
            if value is not True and value is not False:
                valid = False
                break
            normalized.append(
                {
                    "index": idx,
                    "criteria_met": bool(value),
                    "explanation": str(row.get("explanation", "")).strip(),
                    "raw_response": last_response,
                }
            )
        if valid:
            return normalized
    raise RuntimeError(f"judge did not return valid JSON: {last_response}")


def _calculate_score(rubrics: Sequence[RubricItem], grades: Sequence[Dict[str, Any]]) -> float | None:
    total_possible = sum(item.points for item in rubrics if item.points > 0)
    if total_possible <= 0:
        return None
    achieved = 0.0
    for item, grade in zip(rubrics, grades, strict=True):
        if bool(grade.get("criteria_met")):
            achieved += item.points
    return achieved / total_possible


def _score_sample(
    *,
    example_tags: Sequence[str],
    rubrics: Sequence[RubricItem],
    grades: Sequence[Dict[str, Any]],
) -> tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    overall_score = _calculate_score(rubrics, grades)
    if overall_score is None:
        raise RuntimeError("No positive-point HealthBench rubric items")

    metrics: Dict[str, float] = {"overall_score": float(overall_score)}
    for tag in example_tags:
        metrics[str(tag)] = float(overall_score)

    rubric_tag_pairs: Dict[str, List[tuple[RubricItem, Dict[str, Any]]]] = {}
    for rubric_item, grade in zip(rubrics, grades, strict=True):
        for tag in rubric_item.tags:
            rubric_tag_pairs.setdefault(tag, []).append((rubric_item, grade))

    for tag, pairs in rubric_tag_pairs.items():
        tag_rubrics = [item for item, _ in pairs]
        tag_grades = [grade for _, grade in pairs]
        score = _calculate_score(tag_rubrics, tag_grades)
        if score is not None:
            metrics[tag] = float(score)

    rubric_rows: List[Dict[str, Any]] = []
    for rubric_item, grade in zip(rubrics, grades, strict=True):
        rubric_rows.append(
            {
                **rubric_item.to_dict(),
                "criteria_met": bool(grade.get("criteria_met")),
                "explanation": str(grade.get("explanation", "")).strip(),
            }
        )
    return float(overall_score), metrics, rubric_rows


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _aggregate_metric_means(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metric_values: Dict[str, List[float]] = {}
    for row in rows:
        metrics = row.get("metrics") or {}
        for name, value in metrics.items():
            try:
                metric_values.setdefault(str(name), []).append(float(value))
            except Exception:
                continue
    return {name: _mean(values) for name, values in metric_values.items()}


def run(args: argparse.Namespace) -> None:
    data_path = _resolve_path(args.data_path)
    subset = _resolve_subset_name(data_path, args.subset)
    output_root = _resolve_path(args.output_root)
    default_corpora = DEFAULT_BM25_CORPORA if args.method == "bm25" else DEFAULT_MEDCPT_CORPORA
    corpora = parse_corpus_list(args.corpus, default=default_corpora)
    rows = _load_jsonl(data_path, limit=args.limit)
    retriever, retriever_name = _build_retriever(args, corpora)

    run_dir = (
        _resolve_path(args.run_dir)
        if args.run_dir
        else make_run_dir(
            output_root,
            method=f"healthbench_{args.method}",
            dataset="healthbench",
            split=subset,
            model=args.model,
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.json"
    if not args.run_dir or not results_path.exists():
        results_path.write_text("", encoding="utf-8")

    config = vars(args).copy()
    config.update(
        {
            "data_path": str(data_path),
            "subset": subset,
            "output_root": str(output_root),
            "run_dir": str(run_dir),
            "db_dir": str(_resolve_path(args.db_dir)),
            "corpus": ",".join(corpora),
        }
    )
    write_json(run_dir / "config.json", config)

    results = _load_existing_results(results_path) if args.run_dir else []
    completed_ids = {str(item.get("id", "")).strip() for item in results if str(item.get("id", "")).strip()}
    if completed_ids:
        print(f"[resume] loaded {len(completed_ids)} existing HealthBench results from {results_path}", flush=True)

    failed_samples = sum(1 for item in results if item.get("score") is None)
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("prompt_id", f"row_{idx - 1}"))
        if row_id in completed_ids:
            continue

        prompt_messages = _normalize_messages(row.get("prompt") or [])
        example_tags = [str(tag).strip() for tag in row.get("example_tags", []) if str(tag).strip()]
        rubrics = [RubricItem.from_dict(item) for item in row.get("rubrics", [])]

        model_messages = prompt_messages
        raw_response = ""
        answer_text = ""
        score = None
        metrics: Dict[str, float] = {}
        rubric_rows: List[Dict[str, Any]] = []
        snippets: List[Dict[str, Any]] = []
        error = ""

        try:
            if retriever is not None:
                query = _latest_user_text(prompt_messages)
                snippets, _ = retriever.retrieve(query, k=args.top_k)
                model_messages = _with_retrieved_docs(prompt_messages, snippets)
            raw_response = call_chat_model(
                model_messages,
                provider=args.provider,
                model=args.model,
                timeout_s=args.timeout_s,
                azure_suffix=args.azure_suffix,
            )
            answer_text = _normalize_text(raw_response)
            if not answer_text:
                raise RuntimeError("model returned empty answer")
            grades = _judge_rubrics(
                prompt_messages=prompt_messages,
                response_text=answer_text,
                rubrics=rubrics,
                judge_model=args.judge_model,
                judge_timeout_s=args.judge_timeout_s,
                judge_azure_suffix=args.judge_azure_suffix,
            )
            score, metrics, rubric_rows = _score_sample(
                example_tags=example_tags,
                rubrics=rubrics,
                grades=grades,
            )
        except Exception as exc:
            failed_samples += 1
            error = str(exc)

        result = {
            "id": row_id,
            "score": score,
            "metrics": metrics,
            "rubric_items": rubric_rows,
            "answer_text": answer_text,
            "raw_response": raw_response,
            "trace": {
                "method": args.method,
                "model": args.model,
                "judge_model": args.judge_model,
                "retriever": retriever_name,
                "corpus": ",".join(corpora) if retriever is not None else "none",
                "top_k": int(args.top_k) if retriever is not None else 0,
            },
        }
        if snippets:
            result["trace"]["retrieved_docs"] = summarize_retrieved_docs(snippets)
        if error:
            result["trace"]["error"] = error
        results.append(result)
        completed_ids.add(row_id)

        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        status = "ERROR" if score is None else f"score={score:.4f}"
        print(f"[{idx}/{len(rows)}] id={result['id']} {status}", flush=True)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    scored = [float(row["score"]) for row in results if row.get("score") is not None]
    metric_means = _aggregate_metric_means(results)
    summary = {
        "method": args.method,
        "dataset": "healthbench",
        "split": subset,
        "num_samples": len(results),
        "failed_samples": failed_samples,
        "score": _mean(scored),
        "overall_score": metric_means.get("overall_score", 0.0),
        "model": args.model,
        "judge_model": args.judge_model,
        "retriever": retriever_name,
        "run_dir": str(run_dir),
    }
    if retriever is not None:
        summary["corpus"] = ",".join(corpora)
        summary["top_k"] = int(args.top_k)

    write_json(summary_path, summary)
    write_json(metrics_path, metric_means)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["direct", "bm25", "medcpt"], default="direct")
    parser.add_argument("--data-path", default=str(DEFAULT_SUBSET_PATHS["hard"]))
    parser.add_argument("--subset", default="")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument(
        "--provider",
        choices=["auto", "azure", "openai", "qwen"],
        default="azure",
    )
    parser.add_argument("--model", default="o3-mini")
    parser.add_argument("--azure-suffix", default="_2")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument(
        "--judge-azure-suffix",
        default="",
        help="Empty string forces the primary Azure route from Keys/env.sh.",
    )
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--judge-timeout-s", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--run-dir", default="", help="Resume/write results in a fixed run directory.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR))
    parser.add_argument(
        "--corpus",
        default=",".join(parse_corpus_list(None, default=DEFAULT_BM25_CORPORA)),
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    try:
        run(build_arg_parser().parse_args())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
