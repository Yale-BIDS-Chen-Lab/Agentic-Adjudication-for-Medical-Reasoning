#!/usr/bin/env python3
"""Local MedRBench baselines with optional retrieval and judge-based evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summary import build_summary
from methods.BM25.retriever import BM25Retriever, DEFAULT_BM25_CORPORA
from methods.MedCPT.retriever import DEFAULT_MEDCPT_CORPORA, MedCPTRetriever
from methods.common.io import make_run_dir, write_json
from methods.common.llm import call_chat_model
from methods.common.prompt import format_retrieved_docs
from methods.common.retrieval import DEFAULT_DB_DIR, parse_corpus_list, summarize_retrieved_docs


DEFAULT_DIAG_PATH = PROJECT_ROOT / "data" / "MedRBench" / "diagnosis_957_cases_with_rare_disease_491.json"
DEFAULT_TREAT_PATH = PROJECT_ROOT / "data" / "MedRBench" / "treatment_496_cases_with_rare_disease_165.json"


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    pattern = (
        r"(?:^|\n)(?:#+\s*)?"
        r"(?:final answer|answer|diagnosis|final diagnosis|treatment plan|final treatment plan)"
        r"\s*:\s*(.+?)(?:\n(?:#+\s*)?[A-Za-z][^\n]{0,40}:|\Z)"
    )
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _normalize_text(match.group(1))
    return _normalize_text(raw)


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


def _load_case_map(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object keyed by case id: {path}")
    return payload


def _select_cases(case_map: Dict[str, Dict[str, Any]], limit: int) -> List[Tuple[str, Dict[str, Any]]]:
    return list(case_map.items())[: max(0, int(limit))]


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


def _case_parts(task: str, case_row: Dict[str, Any]) -> Tuple[str, str, str]:
    generate_case = case_row.get("generate_case") or {}
    case_summary = str(generate_case.get("case_summary", "")).strip()
    if task == "diagnose":
        return case_summary, "", _normalize_text(str(generate_case.get("diagnosis_results", "")).strip())
    return (
        case_summary,
        str(generate_case.get("treatment_planning_analysis", "")).strip(),
        _normalize_text(str(generate_case.get("treatment_plan_results", "")).strip()),
    )


def _retrieval_query(task: str, case_row: Dict[str, Any]) -> str:
    case_summary, analysis, _ = _case_parts(task, case_row)
    text = case_summary if task == "diagnose" else f"{case_summary}\n\n{analysis}"
    return _normalize_text(text)


def _base_messages(task: str, case_id: str, case_row: Dict[str, Any]) -> List[Dict[str, str]]:
    case_summary, analysis, _ = _case_parts(task, case_row)
    if task == "diagnose":
        user = (
            "Read the following medical case and provide the single most likely final diagnosis.\n"
            "Return only the diagnosis phrase, without bullets or explanation.\n\n"
            f"Case ID: {case_id}\n\n"
            f"Case summary:\n{case_summary}"
        )
    else:
        user = (
            "Read the following medical case and provide the final treatment plan.\n"
            "Return only the treatment plan, without bullets or explanation.\n\n"
            f"Case ID: {case_id}\n\n"
            f"Case summary:\n{case_summary}\n\n"
            f"Existing treatment analysis:\n{analysis}"
        )
    return [
        {
            "role": "system",
            "content": "You are a careful clinical reasoning assistant. Give one concise final answer.",
        },
        {"role": "user", "content": user},
    ]


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
                "Use these references when they are helpful and then provide one final answer."
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


def _judge_prediction(
    *,
    task: str,
    case_summary: str,
    gold: str,
    pred: str,
    judge_model: str,
    judge_timeout_s: int,
    judge_azure_suffix: str,
) -> Dict[str, Any]:
    if task == "diagnose":
        task_rules = (
            "Treat synonymous disease names, standard abbreviations, and equivalent clinical naming as correct. "
            "Broader, clearly different, or conflicting diagnoses are incorrect."
        )
        task_name = "final diagnosis"
    else:
        task_rules = (
            "Treat equivalent drug names, formulations, or wording as correct if they recommend the same core treatment. "
            "Materially different therapy, a missing key intervention, or a contradictory plan is incorrect."
        )
        task_name = "final treatment plan"

    user = (
        f"Judge whether the model answer matches the reference {task_name} for the same medical case.\n\n"
        f"Case summary:\n{case_summary}\n\n"
        f"Reference answer:\n{gold}\n\n"
        f"Model answer:\n{pred}\n\n"
        "Rules:\n"
        f"- {task_rules}\n"
        "- If the model answer lists multiple inconsistent final answers, mark it incorrect unless one final answer is clearly committed.\n"
        "- Ignore harmless wording differences.\n\n"
        'Return JSON only: {"correct": true/false, "explanation": "..."}'
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
        value = payload.get("correct")
        if value is True or value is False:
            return {
                "correct": bool(value),
                "explanation": str(payload.get("explanation", "")).strip(),
                "raw_response": last_response,
            }
    raise RuntimeError(f"judge did not return valid JSON: {last_response}")


def _run_task(
    *,
    task: str,
    case_path: Path,
    case_limit: int,
    run_dir: Path,
    args: argparse.Namespace,
    retriever: Any,
    retriever_name: str,
    corpora: Sequence[str],
) -> Dict[str, Any]:
    case_map = _load_case_map(case_path)
    selected_cases = _select_cases(case_map, case_limit)
    results_path = run_dir / f"{task}_results.jsonl"
    summary_path = run_dir / f"{task}_summary.json"
    if not args.run_dir or not results_path.exists():
        results_path.write_text("", encoding="utf-8")

    results = _load_existing_results(results_path) if args.run_dir else []
    completed_ids = {str(item.get("id", "")).strip() for item in results if str(item.get("id", "")).strip()}
    if completed_ids:
        print(f"[resume] loaded {len(completed_ids)} existing {task} results from {results_path}", flush=True)

    for idx, (case_id, case_row) in enumerate(selected_cases, start=1):
        if case_id in completed_ids:
            continue

        case_summary, _, gold = _case_parts(task, case_row)
        raw_response = ""
        pred = ""
        judge: Dict[str, Any] = {}
        snippets: List[Dict[str, Any]] = []
        error = ""

        try:
            model_messages = _base_messages(task, case_id, case_row)
            if retriever is not None:
                query = _retrieval_query(task, case_row)
                snippets, _ = retriever.retrieve(query, k=args.top_k)
                model_messages = _with_retrieved_docs(model_messages, snippets)
            raw_response = call_chat_model(
                model_messages,
                provider=args.provider,
                model=args.model,
                timeout_s=args.timeout_s,
                azure_suffix=args.azure_suffix,
            )
            pred = _extract_answer(raw_response)
            if not pred:
                raise RuntimeError("model returned empty answer")
            judge = _judge_prediction(
                task=task,
                case_summary=case_summary,
                gold=gold,
                pred=pred,
                judge_model=args.judge_model,
                judge_timeout_s=args.judge_timeout_s,
                judge_azure_suffix=args.judge_azure_suffix,
            )
        except Exception as exc:
            error = str(exc)

        correct = bool(judge.get("correct")) if judge else False
        result = {
            "id": case_id,
            "pred": pred or None,
            "gold": gold,
            "correct": correct,
            "raw_response": raw_response,
            "trace": {
                "method": args.method,
                "task": task,
                "model": args.model,
                "judge_model": args.judge_model,
                "retriever": retriever_name,
                "corpus": ",".join(corpora) if retriever is not None else "none",
                "top_k": int(args.top_k) if retriever is not None else 0,
            },
        }
        if snippets:
            result["trace"]["retrieved_docs"] = summarize_retrieved_docs(snippets)
        if judge:
            result["trace"]["judge_explanation"] = judge.get("explanation", "")
            result["trace"]["judge_raw_response"] = judge.get("raw_response", "")
        if error:
            result["trace"]["error"] = error
        results.append(result)
        completed_ids.add(case_id)

        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{task} {idx}/{len(selected_cases)}] id={case_id} correct={correct}", flush=True)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    extra = {
        "task": task,
        "judge_model": args.judge_model,
        "source_path": str(case_path),
    }
    if retriever is not None:
        extra["corpus"] = ",".join(corpora)
        extra["top_k"] = int(args.top_k)

    summary = build_summary(
        results,
        method=args.method,
        dataset=f"medrbench_{task}",
        split="test",
        model=args.model,
        run_dir=run_dir,
        retriever=retriever_name,
        extra=extra,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def _build_overall_summary(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    task_summaries: Dict[str, Dict[str, Any]],
    retriever_name: str,
    corpora: Sequence[str],
) -> Dict[str, Any]:
    num_samples = sum(int(summary.get("num_samples", 0)) for summary in task_summaries.values())
    correct = sum(int(summary.get("correct", 0)) for summary in task_summaries.values())
    failed = sum(int(summary.get("failed_samples", 0)) for summary in task_summaries.values())
    payload = {
        "method": args.method,
        "dataset": "medrbench",
        "split": args.task,
        "num_samples": num_samples,
        "correct": correct,
        "accuracy": (correct / num_samples) if num_samples else 0.0,
        "failed_samples": failed,
        "model": args.model,
        "judge_model": args.judge_model,
        "retriever": retriever_name,
        "run_dir": str(run_dir),
        "tasks": task_summaries,
    }
    if retriever_name != "none":
        payload["corpus"] = ",".join(corpora)
        payload["top_k"] = int(args.top_k)
    return payload


def run(args: argparse.Namespace) -> None:
    output_root = _resolve_path(args.output_root)
    default_corpora = DEFAULT_BM25_CORPORA if args.method == "bm25" else DEFAULT_MEDCPT_CORPORA
    corpora = parse_corpus_list(args.corpus, default=default_corpora)
    retriever, retriever_name = _build_retriever(args, corpora)

    run_dir = (
        _resolve_path(args.run_dir)
        if args.run_dir
        else make_run_dir(
            output_root,
            method=f"medrbench_{args.method}",
            dataset="medrbench",
            split=args.task,
            model=args.model,
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["diag_path"] = str(_resolve_path(args.diag_path))
    config["treat_path"] = str(_resolve_path(args.treat_path))
    config["output_root"] = str(output_root)
    config["run_dir"] = str(run_dir)
    config["db_dir"] = str(_resolve_path(args.db_dir))
    config["corpus"] = ",".join(corpora)
    write_json(run_dir / "config.json", config)

    task_summaries: Dict[str, Dict[str, Any]] = {}
    if args.task in {"diagnose", "both"}:
        task_summaries["diagnose"] = _run_task(
            task="diagnose",
            case_path=_resolve_path(args.diag_path),
            case_limit=args.diag_cases,
            run_dir=run_dir,
            args=args,
            retriever=retriever,
            retriever_name=retriever_name,
            corpora=corpora,
        )
    if args.task in {"treatment", "both"}:
        task_summaries["treatment"] = _run_task(
            task="treatment",
            case_path=_resolve_path(args.treat_path),
            case_limit=args.treat_cases,
            run_dir=run_dir,
            args=args,
            retriever=retriever,
            retriever_name=retriever_name,
            corpora=corpora,
        )

    overall_summary = _build_overall_summary(
        run_dir=run_dir,
        args=args,
        task_summaries=task_summaries,
        retriever_name=retriever_name,
        corpora=corpora,
    )
    write_json(run_dir / "summary.json", overall_summary)
    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["direct", "bm25", "medcpt"], default="direct")
    parser.add_argument("--task", choices=["diagnose", "treatment", "both"], default="both")
    parser.add_argument("--diag-path", default=str(DEFAULT_DIAG_PATH))
    parser.add_argument("--treat-path", default=str(DEFAULT_TREAT_PATH))
    parser.add_argument("--diag-cases", type=int, default=5)
    parser.add_argument("--treat-cases", type=int, default=5)
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
