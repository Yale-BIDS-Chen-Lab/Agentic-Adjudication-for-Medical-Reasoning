"""Output parsers for canonical result rows."""

from __future__ import annotations

import re
from typing import Dict, Optional


def parse_answer(raw_response: str, options: Dict[str, str]) -> Optional[str]:
    """Parse one MCQ option letter from model text.

    The parser is deliberately shared across methods so Direct, BM25, MedCPT,
    and 2-agent results are scored with the same answer extraction rules.
    """
    allowed = {str(k).strip().upper() for k in options}
    text = str(raw_response or "").strip()
    if not text:
        return None

    # Expected case: the prompt asks for a bare option letter.
    first_token = re.match(r"^\s*\(?([A-Z])\)?(?:[.\):\-\s]|$)", text, re.IGNORECASE)
    if first_token:
        value = first_token.group(1).upper()
        if value in allowed:
            return value

    # Robustness for JSON-like or verbose answers.
    patterns = [
        r'(?im)^\s*answer(?:_choice)?\s*(?::|=|-|\s)+\(?([A-Z])\)?\s*$',
        r'(?im)^\s*final answer\s*(?::|=|-|\s)+\(?([A-Z])\)?\s*$',
        r'"answer(?:_choice)?"\s*:\s*"([A-Z])"',
        r"\banswer(?:_choice)?\s*[:\-]\s*\(?([A-Z])\)?",
        r"\bfinal answer\s*[:\-]\s*\(?([A-Z])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).upper()
            if value in allowed:
                return value

    # PubMedQA often has yes/no/maybe option text; map that text back to a letter.
    lower = text.lower()
    for key, option_text in options.items():
        normalized_option = option_text.strip().lower()
        if normalized_option and re.search(rf"\b{re.escape(normalized_option)}\b", lower):
            return key.upper()

    return None
