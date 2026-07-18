"""OpenAI Chat Completions client for LLM-as-judge.

Auth (first match wins):
  1. OPENAI_API_KEY env
  2. OPENAI_API_KEY in repo `.env`
  3. OPENAI_API_KEY in risePlatform `.env` (shared local key)

Default model: gpt-5-mini (cheap GPT-5 tier).
Uses urllib only — no openai pip package required.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_API_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

REPO_ROOT = Path(__file__).resolve().parents[1]
RISE_ENV = Path.home() / "Documents" / "risePlatform" / ".env"

# Fatal: stop the whole judge run (do not continue to next item).
_STOP_PHRASES = (
    "insufficient_quota",
    "insufficient funds",
    "exceeded your current quota",
    "billing",
    "payment required",
    "credit",
    "no money",
    "rate_limit",
    "rate limit",
    "too many requests",
    "tokens per min",
    "requests per min",
)


@dataclass(frozen=True)
class ApiResult:
    text: str
    model: str
    error: Optional[str] = None
    http_status: Optional[int] = None
    stop_run: bool = False
    finish_reason: Optional[str] = None


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            out[k] = v
    return out


def resolve_api_key() -> Optional[str]:
    for name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    for path in (REPO_ROOT / ".env", RISE_ENV):
        val = _parse_dotenv(path).get("OPENAI_API_KEY", "").strip()
        if val:
            return val
    return None


def is_stop_error(msg: str, *, http_status: int | None = None) -> bool:
    """True when we should halt the whole run (quota / billing / rate limit)."""
    if http_status in {402, 429}:
        return True
    low = (msg or "").lower()
    return any(p in low for p in _STOP_PHRASES)


def run_openai_chat(
    prompt: str,
    *,
    system: str,
    model: str | None = None,
    max_tokens: int = 4096,
    timeout_s: int = 180,
) -> ApiResult:
    key = resolve_api_key()
    model_id = model or DEFAULT_API_MODEL
    if not key:
        return ApiResult(
            text="",
            model=model_id,
            error=(
                "No OPENAI_API_KEY found. Export it, or put OPENAI_API_KEY=... in "
                f"{REPO_ROOT / '.env'}"
            ),
            stop_run=True,
        )

    is_gpt5 = bool(re.match(r"^gpt-5", model_id) or model_id.startswith("o"))
    body: dict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        # Force a JSON object so content is less often empty/prose.
        "response_format": {"type": "json_object"},
    }
    if is_gpt5:
        # Reasoning models can spend the whole budget internally → empty content.
        body["max_completion_tokens"] = max_tokens
        body["reasoning_effort"] = "low"
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = 0

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        err = f"HTTP {e.code}: {detail[:800]}"
        # Older models may reject response_format / reasoning_effort — retry plain once.
        if e.code == 400 and (
            "response_format" in detail
            or "reasoning_effort" in detail
            or "Unsupported" in detail
        ):
            body.pop("response_format", None)
            body.pop("reasoning_effort", None)
            req2 = urllib.request.Request(
                API_URL,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req2, timeout=timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e2:
                detail2 = e2.read().decode("utf-8", errors="replace")
                err2 = f"HTTP {e2.code}: {detail2[:800]}"
                return ApiResult(
                    text="",
                    model=model_id,
                    error=err2,
                    http_status=e2.code,
                    stop_run=is_stop_error(err2, http_status=e2.code),
                )
            except Exception as e2:
                err2 = str(e2)
                return ApiResult(
                    text="",
                    model=model_id,
                    error=err2,
                    stop_run=is_stop_error(err2),
                )
        else:
            return ApiResult(
                text="",
                model=model_id,
                error=err,
                http_status=e.code,
                stop_run=is_stop_error(err, http_status=e.code),
            )
    except Exception as e:
        err = str(e)
        return ApiResult(
            text="",
            model=model_id,
            error=err,
            stop_run=is_stop_error(err),
        )

    choice0 = (payload.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    finish = choice0.get("finish_reason")
    text = (msg.get("content") or "").strip()
    # Some reasoning models put empty content with refusal / other fields.
    if not text and msg.get("refusal"):
        return ApiResult(
            text="",
            model=payload.get("model", model_id),
            error=f"refusal: {msg.get('refusal')}",
            finish_reason=finish,
        )
    if not text:
        return ApiResult(
            text="",
            model=payload.get("model", model_id),
            error=f"empty content (finish_reason={finish})",
            finish_reason=finish,
        )
    return ApiResult(
        text=text,
        model=payload.get("model", model_id),
        finish_reason=finish,
    )
