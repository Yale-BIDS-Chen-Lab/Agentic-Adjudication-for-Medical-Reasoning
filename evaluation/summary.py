"""Summary builders for one completed method run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import count_correct, count_failed


def build_summary(
    results: List[Dict[str, Any]],
    *,
    method: str,
    dataset: str,
    split: str,
    model: str,
    run_dir: Path,
    retriever: str = "none",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical summary object described in EXPERIMENT_CONTRACT.md."""
    num_samples = len(results)
    correct = count_correct(results)
    summary: Dict[str, Any] = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "num_samples": num_samples,
        "correct": correct,
        "accuracy": (correct / num_samples) if num_samples else 0.0,
        "failed_samples": count_failed(results),
        "model": model,
        "retriever": retriever,
        "run_dir": str(run_dir),
    }
    if extra:
        summary.update(extra)
    return summary
