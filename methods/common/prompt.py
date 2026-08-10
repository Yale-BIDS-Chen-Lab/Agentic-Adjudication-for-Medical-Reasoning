"""Prompt builders shared by benchmark methods.

All prompt text should live here rather than inside method runners. That keeps
method code focused on orchestration and makes later prompt comparisons explicit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def format_options(options: Dict[str, str]) -> str:
    """Render MCQ options in stable letter order."""
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))


def format_retrieved_docs(snippets: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved snippets in a stable human-readable layout."""
    rows: List[str] = []
    for idx, item in enumerate(snippets, start=1):
        title = str(item.get("title", "")).strip() or "(untitled)"
        corpus = str(item.get("corpus", "")).strip()
        content = str(item.get("content", "")).strip() or str(item.get("contents", "")).strip()
        header = f"Document [{idx}]"
        if corpus:
            header += f" [{corpus}]"
        rows.append(f"{header}\nTitle: {title}\nContent: {content}".strip())
    return "\n\n".join(rows)


def build_direct_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the no-retrieval Direct baseline prompt.

    The output is a chat-completions message list because Direct, BM25, MedCPT,
    and 2-agent methods can all reuse the same LLM interface.
    """
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    user = (
        "Answer the following medical multiple-choice question.\n"
        f"Return only one option letter from: {labels}.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        "Final answer:"
    )
    return [
        {
            "role": "system",
            "content": "You are a careful medical QA assistant. Choose exactly one option.",
        },
        {"role": "user", "content": user},
    ]


def build_retrieval_prompt(sample: Dict[str, Any], snippets: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build the shared retrieval prompt used by BM25 and MedCPT."""
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    docs = format_retrieved_docs(snippets)
    user = (
        "Answer the following medical multiple-choice question using the retrieved documents as evidence.\n"
        f"Return only one option letter from: {labels}.\n\n"
        f"Retrieved documents:\n{docs}\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        "Final answer:"
    )
    return [
        {
            "role": "system",
            "content": "You are a careful medical QA assistant. Use the documents when helpful and choose exactly one option.",
        },
        {"role": "user", "content": user},
    ]
