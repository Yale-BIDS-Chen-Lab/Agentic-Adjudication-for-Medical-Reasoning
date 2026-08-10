"""Metrics over canonical result rows."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def is_correct(pred: Any, gold: Any) -> bool:
    """Canonical MCQ correctness with support for multi-answer gold strings/lists."""
    if pred is None:
        return False

    normalized_pred = str(pred).strip().upper()
    if isinstance(gold, (list, tuple, set)):
        accepted = {str(item).strip().upper() for item in gold if str(item).strip()}
        return normalized_pred in accepted

    normalized_gold = str(gold).strip().upper()
    if "," in normalized_gold:
        accepted = {part.strip() for part in normalized_gold.split(",") if part.strip()}
        return normalized_pred in accepted
    return normalized_pred == normalized_gold


def count_correct(results: Iterable[Dict[str, Any]]) -> int:
    """Count rows already marked correct or infer correctness from pred/gold."""
    total = 0
    for row in results:
        if "correct" in row:
            total += int(bool(row["correct"]))
        else:
            total += int(is_correct(row.get("pred"), row.get("gold_options", row.get("gold"))))
    return total


def count_failed(results: Iterable[Dict[str, Any]]) -> int:
    """Count failed samples.

    A row is failed if the method recorded an error or did not produce a parsed
    prediction. Failed rows still remain in results.jsonl and are scored wrong.
    """
    failed = 0
    for row in results:
        trace = row.get("trace") or {}
        failed += int(bool(trace.get("error")) or row.get("pred") is None)
    return failed
