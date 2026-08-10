"""Filesystem helpers for experiment outputs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def make_run_dir(output_root: Path, *, method: str, dataset: str, split: str, model: str) -> Path:
    """Create a timestamped run directory under the requested output root."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    run_dir = output_root / f"{timestamp}_{method}_{dataset}_{split}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write canonical result rows as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write one JSON object with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
