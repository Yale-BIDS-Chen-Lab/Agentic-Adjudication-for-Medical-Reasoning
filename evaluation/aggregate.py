"""Aggregate helpers for multiple summary.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_summary(path: Path) -> Dict[str, Any]:
    """Load one summary.json file."""
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_summaries(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate compatible summaries by simple sample-weighted accuracy."""
    rows: List[Dict[str, Any]] = list(summaries)
    total_samples = sum(int(row.get("num_samples", 0)) for row in rows)
    total_correct = sum(int(row.get("correct", 0)) for row in rows)
    total_failed = sum(int(row.get("failed_samples", 0)) for row in rows)
    return {
        "num_runs": len(rows),
        "num_samples": total_samples,
        "correct": total_correct,
        "accuracy": (total_correct / total_samples) if total_samples else 0.0,
        "failed_samples": total_failed,
        "runs": rows,
    }
