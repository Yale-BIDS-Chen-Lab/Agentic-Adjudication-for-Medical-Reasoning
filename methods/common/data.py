"""Dataset loading and canonical sample conversion.

Raw files copied from MedAgentsBench do not all expose identical metadata, so
methods should not consume those rows directly. This module converts each raw
row into the sample schema fixed in EXPERIMENT_CONTRACT.md:

{
  "dataset": "...",
  "split": "...",
  "id": "...",
  "question": "...",
  "options": {"A": "..."},
  "gold": "A",
  "task_type": "mcq"
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def infer_dataset_name(path: Path, explicit: str = "") -> str:
    """Infer a stable dataset name from a data path unless explicitly provided."""
    if str(explicit).strip():
        return str(explicit).strip().lower()

    parent = path.parent.name.strip().lower()
    grandparent = path.parent.parent.name.strip().lower() if path.parent.parent != path.parent else ""
    if grandparent == "nejm":
        return f"nejm_{parent}"
    return parent


def canonical_split_from_path(path: Path) -> str:
    """Map benchmark file names to the split names used in our summaries."""
    name = path.name
    if name == "test.jsonl":
        return "test_full"
    if name == "test_hard.jsonl":
        return "test_hard"
    if name == "sampled_50.jsonl":
        return "sampled_50"
    if name == "sampled_50_hard.jsonl":
        return "sampled_50_hard"
    return path.stem


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a JSONL file, optionally stopping after ``limit`` non-empty rows."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_EMBEDDED_OPTION_RE = re.compile(r"[\r\n]+\s*([A-Z])\.\s*")


def _extract_gold_labels(raw_answer: Any) -> List[str]:
    """Return one or more normalized gold option labels from a raw answer field."""
    parts: List[str] = []
    seen = set()

    if isinstance(raw_answer, (list, tuple)):
        candidates = raw_answer
    else:
        text = str(raw_answer or "").strip().upper()
        if not text:
            return []
        candidates = re.split(r"[,;/|]", text)

    for candidate in candidates:
        label = str(candidate or "").strip().upper()
        if label and label not in seen:
            seen.add(label)
            parts.append(label)
    return parts


def _split_embedded_option_labels(label: str, value: Any) -> Dict[str, str]:
    """Split malformed option strings that accidentally contain later option labels.

    Example source row:
        {"A": "foo\\n B. bar", "C": "...", "D": "..."}
    becomes:
        {"A": "foo", "B": "bar", "C": "...", "D": "..."}
    """
    text = str(value or "").strip()
    if not text:
        return {}

    leading_label = f"{label}. "
    if text.upper().startswith(leading_label):
        text = text[len(leading_label) :].strip()

    matches = list(_EMBEDDED_OPTION_RE.finditer(text))
    if not matches:
        return {label: text}

    segments: Dict[str, str] = {}
    prefix = text[: matches[0].start()].strip()
    if prefix:
        segments[label] = prefix

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment_text = text[start:end].strip()
        segment_label = match.group(1).strip().upper()
        if segment_label and segment_text:
            segments[segment_label] = segment_text

    return segments or {label: text}


def _normalize_options(raw_options: Any, *, row_index: int) -> Dict[str, str]:
    """Normalize raw option maps and repair simple embedded-label formatting issues."""
    if not isinstance(raw_options, dict) or not raw_options:
        raise ValueError("sample has no valid options")

    options: Dict[str, str] = {}
    for raw_label, raw_value in raw_options.items():
        label = str(raw_label).strip().upper()
        if not label:
            continue
        for split_label, split_value in _split_embedded_option_labels(label, raw_value).items():
            if split_label and split_value and split_label not in options:
                options[split_label] = split_value

    if not options:
        raise ValueError(f"row {row_index} has no non-empty options")
    return options


def _repair_nejm_row(raw: Dict[str, Any], *, dataset: str) -> Dict[str, Any]:
    """Repair a small set of known malformed NEJM rows from the source dump."""
    if not str(dataset).startswith("nejm_"):
        return raw

    sample = dict(raw)
    options = dict(sample.get("options") or {})
    realidx = str(sample.get("realidx", "")).strip()

    if dataset == "nejm_general_surgery" and realidx in {"35", "136"}:
        value = str(options.get("B", "")).strip()
        marker = "\nThird."
        if value and marker in value and "C" not in options:
            before, after = value.split(marker, 1)
            options["B"] = before.strip()
            options["C"] = after.strip()
            sample["options"] = options
        return sample

    if dataset == "nejm_internal_medicine" and realidx == "114":
        question = str(sample.get("question", "")).strip()
        marker = "\nIV DDAVP .a."
        if marker in question and "A" not in options:
            sample["question"] = question.split(marker, 1)[0].strip()
            options["A"] = "IV DDAVP."
            sample["options"] = options
        return sample

    if dataset == "nejm_obgyn" and realidx == "74":
        value = str(options.get("B", "")).strip()
        marker = "\nG."
        if value and marker in value and "C" not in options:
            before, after = value.split(marker, 1)
            options["B"] = before.strip()
            options["C"] = after.strip()
            sample["options"] = options
        return sample

    if dataset == "nejm_pediatrics" and realidx == "68":
        value = str(options.get("B", "")).strip()
        marker = ". C."
        if value and marker in value and "C" not in options:
            before, after = value.split(marker, 1)
            options["B"] = (before.strip() + ".").strip()
            options["C"] = after.strip()
            sample["options"] = options
        return sample

    return sample


def gold_display(sample: Dict[str, Any]) -> str:
    """Render one or more accepted gold labels for logs and result rows."""
    labels = sample.get("gold_options") or [sample.get("gold")]
    return ",".join(str(label).strip().upper() for label in labels if str(label).strip())


def sample_is_correct(sample: Dict[str, Any], pred: Optional[str]) -> bool:
    """Return True when ``pred`` matches any accepted gold label for the sample."""
    if pred is None:
        return False
    normalized_pred = str(pred).strip().upper()
    gold_labels = {
        str(label).strip().upper()
        for label in (sample.get("gold_options") or [sample.get("gold")])
        if str(label).strip()
    }
    return normalized_pred in gold_labels


def normalize_sample(raw: Dict[str, Any], *, dataset: str, split: str, row_index: int) -> Dict[str, Any]:
    """Convert one raw MedAgentsBench row into the canonical MCQ sample schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"row {row_index} is not a JSON object")

    raw = _repair_nejm_row(raw, dataset=dataset)

    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError(f"row {row_index} has no question")

    options = _normalize_options(raw.get("options") or {}, row_index=row_index)

    gold_labels = _extract_gold_labels(raw.get("answer_idx", ""))
    if not gold_labels:
        raise ValueError(f"row {row_index} has no answer_idx")

    missing_gold = [label for label in gold_labels if label not in options]
    if missing_gold:
        raise ValueError(
            f"row {row_index} gold answer(s) {','.join(missing_gold)!r} "
            f"are not in options {sorted(options)}"
        )

    # realidx is stable in the source benchmark; row_index is the fallback.
    sample_id = raw.get("realidx", row_index)
    return {
        "dataset": str(dataset).strip().lower(),
        "split": str(split).strip(),
        "id": str(sample_id),
        "question": question,
        "options": options,
        "gold": gold_labels[0],
        "gold_options": gold_labels,
        "task_type": "mcq",
    }
