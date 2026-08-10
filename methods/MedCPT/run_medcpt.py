#!/usr/bin/env python3
"""Minimal MedCPT QA baseline over the shared MedRAG corpus layout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summary import build_summary
from methods.MedCPT.retriever import DEFAULT_MEDCPT_CORPORA, MedCPTRetriever
from methods.common.data import (
    canonical_split_from_path,
    gold_display,
    infer_dataset_name,
    load_jsonl,
    normalize_sample,
    sample_is_correct,
)
from methods.common.io import make_run_dir, write_json, write_jsonl
from methods.common.llm import call_chat_model
from methods.common.parser import parse_answer
from methods.common.prompt import build_retrieval_prompt
from methods.common.retrieval import DEFAULT_DB_DIR, parse_corpus_list, summarize_retrieved_docs


def run(args: argparse.Namespace) -> None:
    """Run MedCPT retrieval + one answer pass on one JSONL split."""
    data_path = Path(args.data_path).resolve()
    dataset = infer_dataset_name(data_path, args.dataset)
    split = args.split or canonical_split_from_path(data_path)
    output_root = Path(args.output_root).resolve()
    corpora = parse_corpus_list(args.corpus)

    raw_rows = load_jsonl(data_path, limit=args.limit)
    samples = [
        normalize_sample(row, dataset=dataset, split=split, row_index=i)
        for i, row in enumerate(raw_rows)
    ]
    retriever = MedCPTRetriever(db_dir=args.db_dir, corpora=corpora, device=args.device)

    run_dir = make_run_dir(output_root, method="medcpt", dataset=dataset, split=split, model=args.model)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    config = vars(args).copy()
    config.update(
        {
            "data_path": str(data_path),
            "dataset": dataset,
            "split": split,
            "output_root": str(output_root),
            "db_dir": str(Path(args.db_dir).resolve()),
            "corpus": ",".join(corpora),
        }
    )
    write_json(config_path, config)

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        raw_response = ""
        error: Optional[str] = None
        pred: Optional[str] = None
        snippets: List[Dict[str, Any]] = []
        gold = gold_display(sample)

        try:
            snippets, _ = retriever.retrieve(sample["question"], k=args.top_k)
            raw_response = call_chat_model(
                build_retrieval_prompt(sample, snippets),
                provider=args.provider,
                model=args.model,
                timeout_s=args.timeout_s,
                azure_suffix=args.azure_suffix,
            )
            pred = parse_answer(raw_response, sample["options"])
            if pred is None:
                raise RuntimeError("could not parse answer option")
        except Exception as exc:
            error = str(exc)

        correct = sample_is_correct(sample, pred)
        result = {
            "id": sample["id"],
            "pred": pred,
            "gold": gold,
            "correct": correct,
            "raw_response": raw_response,
            "trace": {
                "method": "medcpt",
                "model": args.model,
                "retriever": "MedCPT",
                "corpus": ",".join(corpora),
                "top_k": int(args.top_k),
                "retrieved_docs": summarize_retrieved_docs(snippets),
            },
        }
        if len(sample.get("gold_options") or []) > 1:
            result["gold_options"] = sample["gold_options"]
        if error:
            result["trace"]["error"] = error
        results.append(result)

        write_jsonl(results_path, results)
        print(
            f"[{idx}/{len(samples)}] id={sample['id']} pred={pred} "
            f"gold={gold} correct={correct} retrieved={len(snippets)}",
            flush=True,
        )
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    summary = build_summary(
        results,
        method="medcpt",
        dataset=dataset,
        split=split,
        model=args.model,
        run_dir=run_dir,
        retriever="MedCPT",
        extra={
            "corpus": ",".join(corpora),
            "top_k": int(args.top_k),
        },
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MedCPT medical QA baseline.")
    parser.add_argument("--data-path", required=True, help="Path to a MedAgentsBench JSONL file.")
    parser.add_argument("--dataset", default="", help="Dataset name. Defaults to inferred name from the data path.")
    parser.add_argument("--split", default="", help="Canonical split name. Defaults from file name.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"), help="Directory for run outputs.")
    parser.add_argument(
        "--provider",
        choices=["auto", "azure", "openai", "qwen"],
        default="auto",
        help="Inference backend. With auto, Qwen model names use Qwen; others use Azure.",
    )
    parser.add_argument("--model", default="o3-mini", help="Model/deployment name.")
    parser.add_argument(
        "--azure-suffix",
        default=None,
        help="Azure env suffix such as _1 or _2. Auto-picks _2 for o3-mini if available.",
    )
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Root directory of corpus indexes and chunks.")
    parser.add_argument(
        "--corpus",
        default=",".join(parse_corpus_list(None, default=DEFAULT_MEDCPT_CORPORA)),
        help="Comma-separated corpus list. Default: pubmed,textbooks,statpearls,wikipedia",
    )
    parser.add_argument("--device", default="cpu", help="Dense query encoder device. Default: cpu")
    parser.add_argument("--top-k", type=int, default=16, help="Number of retrieved chunks to pass to the model.")
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Optional delay between API calls.")
    return parser


if __name__ == "__main__":
    try:
        run(build_arg_parser().parse_args())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
