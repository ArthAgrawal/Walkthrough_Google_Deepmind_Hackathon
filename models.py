"""Wrappers for every model call (Hydrogen/Neon/Carbon text models + Decisions API).

All calls go through `_call_with_retry`, which retries transient errors (408/429/5xx and
network errors) after 1 s, 2 s, 4 s, and logs each call to logs/model_calls.jsonl with
inline media bytes stripped. SDK-level retries are disabled so they don't stack with ours.

Decisions API shape (verified against aistudio.google.com/docs/decisions-api, the codelab,
and google-genai 3.6.1 source):
    client.decisions.create(model=..., input=str | [blocks], questions={qid: {...}})
    blocks: {"type": "text", "text": ...}
            {"type": "image"|"audio"|"video"|"document", "data": <base64 str> | "uri": ...,
             "mime_type": ..., "resolution": "low"|"medium"|"high"|"ultra_high"}  (resolution: image/video)
    questions: {"type": "noul",   "instructions": str, "criteria": {"true_description", "false_description"}}
               {"type": "choice", "instructions": str, "criteria": {option_key: description}}   # 2-255
               {"type": "score",  "instructions": str, "criteria": [level0, level1, ...]}       # 2-255
    answers:   noul -> .noul (P(yes)), .confidence ; choice -> .choice, .probabilities, .confidence
               score -> .score, .legend, .probabilities, .confidence
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

import config

log = logging.getLogger("models")

_client: genai.Client | None = None
_client_lock = threading.Lock()
_log_lock = threading.Lock()

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".m4a": "audio/m4a", ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/mov",
    ".pdf": "application/pdf",
}


# ---------------------------------------------------------------- client

def get_client() -> genai.Client:
    """Return a shared genai.Client (v1beta, SDK retries off), loading GEMINI_API_KEY from .env."""
    global _client
    with _client_lock:
        if _client is None:
            load_dotenv(config.ROOT / ".env")
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
            _client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    api_version=config.API_VERSION,
                    retry_options=types.HttpRetryOptions(attempts=1),  # we retry ourselves
                ),
            )
    return _client


# ---------------------------------------------------------------- logging

def _sanitize(obj: Any) -> Any:
    """Return a JSON-safe copy of obj with inline media bytes / base64 data replaced by a size note."""
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes stripped>"
    if hasattr(obj, "model_dump"):
        try:
            return _sanitize(obj.model_dump(exclude_none=True))
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 256:
                out[k] = f"<base64 {len(v)} chars stripped>"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _write_log(record: dict) -> None:
    """Append one JSON record to the model-call log file (thread-safe)."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_sanitize(record), ensure_ascii=False, default=str)
    with _log_lock, config.MODEL_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- retry

def _status_of(exc: Exception) -> int | None:
    """Extract an HTTP status code from either SDK error family (genai APIError or Decisions GenAiError)."""
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _is_retryable(exc: Exception) -> bool:
    """True for transient failures: 408/429/5xx, or network errors with no status.

    Client-side timeouts are NOT retried: a slow model won't get faster, and callers fall back.
    """
    if getattr(exc, "status", None) == "DEADLINE_EXCEEDED":   # our own deadline (server-side 504)
        return False
    status = _status_of(exc)
    if status is not None:
        return status in config.RETRY_STATUS_CODES
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return False
    return any(s in name for s in ("connect", "network", "transport", "remoteprotocol"))


def _call_with_retry(fn: Callable[[], Any], record: dict) -> Any:
    """Run fn(); on transient errors retry after each delay in config.RETRY_DELAYS_S. Logs the outcome."""
    delays = list(config.RETRY_DELAYS_S)
    attempt = 0
    t0 = time.perf_counter()
    while True:
        attempt += 1
        try:
            result = fn()
            record.update(attempts=attempt, ms=round((time.perf_counter() - t0) * 1000), ok=True)
            return result
        except Exception as exc:
            status = _status_of(exc)
            if _is_retryable(exc) and delays:
                wait = delays.pop(0)
                hint = " (429 often = spend cap hit; check aistudio.google.com/spend)" if status == 429 else ""
                log.warning("%s attempt %d failed (%s %s)%s; retrying in %ss",
                            record.get("model"), attempt, status, type(exc).__name__, hint, wait)
                time.sleep(wait)
                continue
            record.update(attempts=attempt, ms=round((time.perf_counter() - t0) * 1000), ok=False,
                          error=f"{status} {type(exc).__name__}: {str(exc)[:2000]}")
            _write_log(record)
            raise


# ---------------------------------------------------------------- text models

def _with_thinking(config_: types.GenerateContentConfig | dict | None, thinking: str | None,
                   timeout_s: float | None) -> types.GenerateContentConfig:
    """Return a GenerateContentConfig with thinking_level and request timeout set (caller's values win)."""
    if config_ is None:
        cfg = types.GenerateContentConfig()
    elif isinstance(config_, dict):
        cfg = types.GenerateContentConfig.model_validate(config_)
    else:
        cfg = config_.model_copy(deep=True)
    if thinking and cfg.thinking_config is None:
        cfg.thinking_config = types.ThinkingConfig(thinking_level=thinking)
    if timeout_s and cfg.http_options is None:
        cfg.http_options = types.HttpOptions(timeout=int(timeout_s * 1000))
    return cfg


def generate(model: str, contents: Any, config_: types.GenerateContentConfig | dict | None = None,
             step: str = "", thinking: str | None = config.THINKING_DEFAULT,
             timeout_s: float | None = config.CALL_TIMEOUT_S) -> types.GenerateContentResponse:
    """Call a text/multimodal Gemini model (HYDROGEN/NEON/CARBON) with retry + logging.

    `thinking` sets ThinkingConfig.thinking_level (MINIMAL/LOW/MEDIUM/HIGH/XHIGH); defaults to
    config.THINKING_DEFAULT. Pass thinking=None to use the model's own default.
    A thinking_config already inside `config_` wins over `thinking`.
    `timeout_s` bounds one request; timeouts raise without retry.
    """
    client = get_client()
    config_ = _with_thinking(config_, thinking, timeout_s)
    record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": "generate", "step": step,
              "model": model, "prompt": contents, "config": config_}
    resp = _call_with_retry(
        lambda: client.models.generate_content(model=model, contents=contents, config=config_), record)
    try:
        text = resp.text
    except Exception:
        text = None
    record.update(output=text, usage=getattr(resp, "usage_metadata", None))
    _write_log(record)
    log.info("%s [%s] %d ms", model, step or "-", record["ms"])
    return resp


def hydrogen(contents: Any, config_=None, step: str = "",
             thinking: str | None = config.THINKING_DEFAULT) -> types.GenerateContentResponse:
    """Fast model: transcription, quick scans."""
    return generate(config.HYDROGEN, contents, config_, step, thinking)


def neon(contents: Any, config_=None, step: str = "",
         thinking: str | None = config.THINKING_DEFAULT) -> types.GenerateContentResponse:
    """Understanding + writing model: household facts, thermostat reading, complaint text."""
    return generate(config.NEON, contents, config_, step, thinking)


def carbon(contents: Any, config_=None, step: str = "",
           thinking: str | None = config.THINKING_CARBON,
           timeout_s: float | None = config.CARBON_TIMEOUT_S) -> types.GenerateContentResponse:
    """Hard-cases model: re-checks uncertain findings (higher thinking, bounded by a timeout)."""
    return generate(config.CARBON, contents, config_, step, thinking, timeout_s)


def parse_json(resp: types.GenerateContentResponse) -> Any:
    """Return the parsed JSON body of a response made with response_mime_type='application/json'."""
    if getattr(resp, "parsed", None) is not None:
        parsed = resp.parsed
        return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


# ---------------------------------------------------------------- Decisions API

def decide(input_: str | dict | list, questions: dict, step: str = "") -> dict[str, Any]:
    """Ask all `questions` against one shared `input_` in ONE Decisions API call.

    Returns {question_id: answer} where each answer is the SDK pydantic object
    (NoulAnswer / ChoiceAnswer / ScoreAnswer). Raises if the response was content-blocked.
    """
    client = get_client()
    record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": "decide", "step": step,
              "model": config.DECISION, "prompt": input_, "config": {"questions": questions}}
    resp = _call_with_retry(
        lambda: client.decisions.create(model=config.DECISION, input=input_, questions=questions), record)
    fb = getattr(resp, "prompt_feedback", None)
    if fb is not None and getattr(fb, "block_reason", None):
        record.update(ok=False, error=f"blocked: {fb.block_reason} {fb.block_reason_message}")
        _write_log(record)
        raise RuntimeError(f"Decisions input blocked: {fb.block_reason} {fb.block_reason_message}")
    answers = resp.answers or {}
    record.update(output={k: a for k, a in answers.items()}, usage=getattr(resp, "usage", None))
    _write_log(record)
    log.info("%s [%s] %d questions, %d ms", config.DECISION, step or "-", len(questions), record["ms"])
    return answers


def media_block(path: str | Path, kind: str | None = None, resolution: str | None = "low") -> dict:
    """Build a Decisions API inline content block (base64 `data`) from a local file."""
    p = Path(path)
    mime = _MIME_BY_SUFFIX.get(p.suffix.lower())
    if mime is None:
        raise ValueError(f"Unknown media type for {p.name}")
    kind = kind or {"image": "image", "audio": "audio", "video": "video"}.get(mime.split("/")[0], "document")
    block = {"type": kind, "data": base64.b64encode(p.read_bytes()).decode("ascii"), "mime_type": mime}
    if resolution and kind in ("image", "video"):
        block["resolution"] = resolution
    return block
