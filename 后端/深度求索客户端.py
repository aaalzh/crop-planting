from __future__ import annotations

import json
import os
import re
import ssl
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _looks_like_env_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""))


def _resolve_api_key(llm_cfg: Dict[str, Any]) -> str:
    direct_key = str(llm_cfg.get("api_key", "")).strip()
    if direct_key:
        return direct_key

    api_env = str(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")).strip() or "DEEPSEEK_API_KEY"
    if api_env and not _looks_like_env_name(api_env):
        return api_env
    return str(os.getenv(api_env, "")).strip()


def _provider_name(llm_cfg: Dict[str, Any]) -> str:
    return str(llm_cfg.get("provider", "llm")).strip() or "llm"


def _extract_http_error_message(exc: HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        return raw.strip()

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message", "")).strip()
        code = str(error.get("code", "")).strip()
        if code and message:
            return f"{code}: {message}"
        if message:
            return message
    return raw.strip()


def llm_client_ready(config: Dict[str, Any]) -> bool:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    if not bool(llm_cfg.get("enabled", False)):
        return False
    return bool(_resolve_api_key(llm_cfg))


def request_llm_chat(
    *,
    config: Dict[str, Any],
    user_message: str,
    logger: Any | None = None,
) -> Dict[str, Any]:
    llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
    if not bool(llm_cfg.get("enabled", False)):
        raise RuntimeError("llm_disabled")

    api_key = _resolve_api_key(llm_cfg)
    if not api_key:
        raise RuntimeError("missing_api_key")

    provider = _provider_name(llm_cfg)
    endpoint = str(llm_cfg.get("endpoint", "https://api.deepseek.com/chat/completions")).strip()
    model = str(llm_cfg.get("model", "deepseek-chat")).strip() or "deepseek-chat"
    timeout_seconds = _safe_int(llm_cfg.get("timeout_seconds", 60), 60) or 60
    max_tokens = _safe_int(llm_cfg.get("max_tokens"))
    temperature = _safe_float(llm_cfg.get("temperature"))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": str(user_message or "")},
        ],
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None and model != "deepseek-reasoner":
        payload["temperature"] = temperature

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds, context=ssl.create_default_context()) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = _extract_http_error_message(exc)
        if logger is not None and hasattr(logger, "warning"):
            logger.warning("%s http error %s: %s", provider, exc.code, detail)
        raise RuntimeError(f"llm_http_error:{exc.code}:{detail}") from exc
    except URLError as exc:
        if logger is not None and hasattr(logger, "warning"):
            logger.warning("%s network error: %s", provider, exc)
        raise RuntimeError("llm_network_error") from exc

    try:
        data = json.loads(raw)
    except Exception as exc:
        if logger is not None and hasattr(logger, "warning"):
            logger.warning("%s invalid json: %s", provider, exc)
        raise RuntimeError("llm_invalid_json") from exc

    choices = data.get("choices", [])
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    text = str(message.get("content", "")).strip()
    if not text:
        raise RuntimeError("llm_empty_answer")

    return {
        "provider": provider,
        "model": model,
        "text": text,
        "raw": data,
    }


def deepseek_client_ready(config: Dict[str, Any]) -> bool:
    return llm_client_ready(config)


def request_deepseek_chat(
    *,
    config: Dict[str, Any],
    user_message: str,
    logger: Any | None = None,
) -> Dict[str, Any]:
    return request_llm_chat(config=config, user_message=user_message, logger=logger)
