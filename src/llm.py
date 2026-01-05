# src/llm.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# Carga .env automáticamente si existe (útil en notebooks)
load_dotenv(override=False)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_client() -> OpenAI:
    api_key = _require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def get_default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def get_default_params() -> Dict[str, Any]:
    """
    Default generation params for OpenAI Responses API.
    You can override per call.
    """
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_output_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

    # top_p is optional; if not set, omit it
    top_p_raw = os.getenv("OPENAI_TOP_P")

    params: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if top_p_raw is not None and top_p_raw != "":
        params["top_p"] = float(top_p_raw)

    return params


def chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str | Dict[str, Any]] = None,
    **overrides: Any,
):
    """
    Thin wrapper over OpenAI Responses API (via client.responses.create),
    using the "input" format with role-based messages.

    - messages: list of { "role": "system"|"user"|"assistant", "content": "..." }
    - tools: list of tool schemas (we'll pass them through)
    - tool_choice: e.g. "auto" or {"type":"function","function":{"name":"get_weather"}}
    - overrides: temperature, max_tokens, top_p, etc.
    """
    client = get_client()
    model = model or get_default_model()
    params = get_default_params()
    params.update(overrides)

    payload: Dict[str, Any] = {
        "model": model,
        "input": messages,
        **params,
    }

    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return client.responses.create(**payload)


# ---------------------------
# Helpers to consume responses
# ---------------------------

def extract_text(response: Any) -> str:
    """
    Try to extract the assistant text from an OpenAI response.
    Works with the Responses API object.
    """
    # The Responses API commonly provides: response.output_text
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: walk output items
    try:
        parts: List[str] = []
        for item in response.output:
            # item could have type="message" with content list
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        parts.append(getattr(c, "text", ""))
        joined = "".join(parts).strip()
        return joined
    except Exception:
        return ""


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls (function calls) if any.
    Returns list of dicts with keys like: name, arguments, call_id
    """
    calls: List[Dict[str, Any]] = []
    try:
        for item in response.output:
            if getattr(item, "type", None) == "tool_call":
                tool_name = getattr(item, "name", None) or getattr(getattr(item, "function", None), "name", None)
                args = getattr(item, "arguments", None)
                call_id = getattr(item, "id", None)
                calls.append(
                    {"name": tool_name, "arguments": args, "call_id": call_id, "raw": item}
                )
    except Exception:
        pass
    return calls

'''
COMO USARLO EN EL NOTEBOOK

from src.llm import chat, extract_text

messages = [
    {"role": "system", "content": "Eres un asistente turístico."},
    {"role": "user", "content": "Dame un plan de 1 día en la ciudad."}
]

resp = chat(messages)
print(extract_text(resp))
'''