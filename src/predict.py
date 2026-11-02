"""Predictors: one `predict(image_path) -> dict` per system (shared contract with Person B).

Person A owns this. Each predictor returns raw model output; the evaluator canonicalizes.
Endpoints come from env (.env): MODAL_VLLM_URL + MODAL_VLLM_API_KEY for the Modal vLLM server.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

load_dotenv()

PROMPT = "Extract the chemical structure from this image and output ONLY the canonical SMILES string."
MAX_TOKENS = 108  # 1.5x the longest label (72 Qwen3.5 tokens, data/dataset_report.json); smoke runs used 150


@cache
def _modal_client() -> OpenAI:
    return OpenAI(base_url=os.environ["MODAL_VLLM_URL"].rstrip("/") + "/v1", api_key=os.environ["MODAL_VLLM_API_KEY"])


def _image_message(image_path: str) -> list[dict]:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]


def _chat(client: OpenAI, model: str, image_path: str, escalated: bool = False,
          thinking: bool = False, max_tokens: int = MAX_TOKENS) -> dict:
    messages = _image_message(image_path)
    # Modal servers return 503 while a container is cold-starting; wait rather than fail.
    for _ in range(90):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                # Qwen3.5 thinks by default; we want a bare SMILES (thinking=True only for the strawman check).
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            )
            latency = time.perf_counter() - t0
            break
        except APIStatusError as e:
            if e.status_code != 503:
                raise
            time.sleep(10)
    else:
        raise TimeoutError(f"{model} endpoint stayed unavailable")
    text = resp.choices[0].message.content or ""
    if "</think>" in text:  # thinking mode: keep only the final answer after the reasoning block
        text = text.split("</think>")[-1]
    return {
        "smiles": text,
        "latency_s": latency,
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "attempts": 1,
        "escalated": escalated,
    }


def predict_untuned(image_path: str) -> dict:
    return _chat(_modal_client(), "base", image_path)


def predict_tuned(image_path: str, lora: str = "smoke") -> dict:
    return _chat(_modal_client(), lora, image_path)


def predict_untuned_thinking(image_path: str) -> dict:
    """Strawman check: untuned model with thinking on and a 1,024-token budget (reasoning counted as output)."""
    return _chat(_modal_client(), "base", image_path, thinking=True, max_tokens=THINKING_MAX_TOKENS)


THINKING_MAX_TOKENS = 1024
SYSTEMS = {"untuned": predict_untuned, "tuned": predict_tuned, "untuned_thinking": predict_untuned_thinking}
SYSTEM_CAPS = {"untuned_thinking": THINKING_MAX_TOKENS}  # loop-rate cap per system; default MAX_TOKENS
