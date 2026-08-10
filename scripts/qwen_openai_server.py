#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, List

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()
MODEL_NAME = ""
TOKENIZER = None
MODEL = None
DEVICE = "cpu"
ENABLE_THINKING = True


def _prepare_messages(messages: List[Dict[str, Any]], response_format: Dict[str, Any] | None) -> List[Dict[str, str]]:
    system_contents: List[str] = []
    normalized: List[Dict[str, str]] = []
    for msg in messages or []:
        role = str(msg.get("role", "user")).strip().lower() or "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_contents.append(content)
        else:
            normalized.append({"role": role, "content": content})
    if response_format and response_format.get("type") == "json_schema":
        schema = (((response_format.get("json_schema") or {}).get("schema")) or {})
        system_contents.insert(
            0,
            "Return JSON only. Do not include markdown fences or explanations.\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}",
        )
    if not system_contents:
        return normalized
    return [{"role": "system", "content": "\n\n".join(system_contents)}, *normalized]


def _render_prompt(messages: List[Dict[str, str]]) -> str:
    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate_text(
    messages: List[Dict[str, Any]],
    response_format: Dict[str, Any] | None,
    temperature: float | None,
    max_tokens: int | None,
) -> Dict[str, Any]:
    prompt_messages = _prepare_messages(messages, response_format)
    prompt = _render_prompt(prompt_messages)
    inputs = TOKENIZER(prompt, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    do_sample = (temperature or 0.0) > 0
    gen_kwargs = {
        "max_new_tokens": int(max_tokens or 512),
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature or 0.7)
        gen_kwargs["top_p"] = 0.95
    with torch.no_grad():
        output = MODEL.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output[0][prompt_len:]
    text = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_tokens": int(prompt_len),
        "completion_tokens": int(new_tokens.shape[-1]),
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> JSONResponse:
    body = await req.json()
    result = _generate_text(
        messages=body.get("messages") or [],
        response_format=body.get("response_format"),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
    )
    now = int(time.time())
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": now,
            "model": str(body.get("model") or MODEL_NAME),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }
    )


def main() -> None:
    global MODEL_NAME, TOKENIZER, MODEL, DEVICE, ENABLE_THINKING

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    MODEL_NAME = args.model
    ENABLE_THINKING = not args.disable_thinking
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    MODEL.eval()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

