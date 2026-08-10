"""LLM provider wrapper used by all methods.

This module hides provider setup and keeps the method runners
from mixing model-routing details with experiment logic.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


QWEN_MODEL_ALIASES = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen/qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen/qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


def _env_value(name: str) -> str:
    """Read an environment variable without leaking secrets in logs."""
    return os.getenv(name, "").strip()


def _normalize_qwen_model(model: str) -> str:
    value = str(model or _env_value("QWEN_MODEL") or "Qwen/Qwen3.5-9B").strip()
    return QWEN_MODEL_ALIASES.get(value.lower(), value)


def _looks_like_qwen_model(model: str) -> bool:
    return "qwen" in str(model or "").strip().lower()


def _resolve_provider(provider: str, model: str) -> str:
    """Choose the concrete backend while keeping method runners model-centric."""
    key = str(provider or "auto").strip().lower()
    if key in {"", "auto"}:
        return "qwen" if _looks_like_qwen_model(model) else "azure"
    if key in {"azure", "azure_openai", "openai_azure"}:
        return "azure"
    if key in {"openai"}:
        return "openai"
    if key in {"qwen", "qwen_openai", "qwen_openai_compatible"}:
        return "qwen"
    raise ValueError(f"Unsupported provider: {provider}")


def _strip_thinking_blocks(text: str) -> str:
    """Remove Qwen thinking blocks when the serving backend returns them inline."""
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _pick_azure_suffix(model: str, explicit_suffix: Optional[str]) -> str:
    """Choose the Azure env suffix for a deployment.

    In the current cluster env, o3-mini has commonly lived behind *_2. The user
    can still override this with --azure-suffix when a run needs a fixed route.
    """
    if explicit_suffix is not None:
        return explicit_suffix
    model_key = str(model).strip().lower()
    if model_key in {"o3-mini", "o3"} and _env_value("AZURE_ENDPOINT_2") and _env_value("AZURE_API_KEY_2"):
        return "_2"
    if _env_value("AZURE_ENDPOINT_1") and _env_value("AZURE_API_KEY_1"):
        return "_1"
    return ""


def _azure_client(model: str, azure_suffix: Optional[str]):
    try:
        from openai import AzureOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'openai'. Activate an environment with "
            "openai installed before running this method."
        ) from exc

    suffix = _pick_azure_suffix(model, azure_suffix)
    endpoint = _env_value(f"AZURE_ENDPOINT{suffix}")
    api_key = _env_value(f"AZURE_API_KEY{suffix}")
    api_version = _env_value(f"AZURE_API_VERSION{suffix}") or _env_value("AZURE_API_VERSION")
    if not endpoint or not api_key or not api_version:
        raise RuntimeError(
            "Missing Azure environment variables. Source .env or export the required "
            "variables, or pass --provider openai for non-Azure OpenAI."
        )
    return AzureOpenAI(azure_endpoint=endpoint.rstrip("/"), api_key=api_key, api_version=api_version)


def _qwen_client(timeout_s: int):
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'openai'. Activate an environment with "
            "openai installed before running Qwen-backed inference."
        ) from exc

    base_url = _env_value("QWEN_OPENAI_BASE_URL") or "http://localhost:8000/v1"
    api_key = _env_value("QWEN_OPENAI_API_KEY") or "EMPTY"
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout_s)


def call_chat_model(
    messages: List[Dict[str, str]],
    *,
    provider: str,
    model: str,
    timeout_s: int,
    azure_suffix: Optional[str],
) -> str:
    """Call one chat model and return stripped text content."""
    resolved_provider = _resolve_provider(provider, model)
    if resolved_provider == "azure":
        client = _azure_client(model, azure_suffix)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout_s,
        )
    elif resolved_provider == "openai":
        from openai import OpenAI

        client = OpenAI(timeout=timeout_s)
        response = client.chat.completions.create(model=model, messages=messages)
    elif resolved_provider == "qwen":
        client = _qwen_client(timeout_s)
        max_tokens = int(_env_value("QWEN_MAX_TOKENS") or "1024")
        response = client.chat.completions.create(
            model=_normalize_qwen_model(model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not response.choices:
        raise RuntimeError("model returned no choices")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("model returned empty content")
    return _strip_thinking_blocks(str(content))
