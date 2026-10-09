"""Provider-agnostic LLM client for RxBridge.

Only used for the multilingual EXPLANATION stage. Extraction is a model we train
ourselves and is out of scope here.

Providers are tried in priority order:

  1. featherless  (hackathon sponsor)  — OpenAI-compatible
  2. cerebras                          — OpenAI-compatible
  3. gemini                            — Google Generative Language REST

Credentials are read, in order, from the process environment, ``$RXBRIDGE_ENV_FILE``,
``./.env`` and ``/Users/faye/.env``. Secret *values* are never logged or returned.

Public API
----------
complete(messages, model=None, temperature=0.2, max_tokens=256) -> str
explain_schedule(schedule, target_language) -> str
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

__all__ = [
    "complete",
    "explain_schedule",
    "probe_provider",
    "LLMError",
    "NoProviderAvailable",
    "ProviderError",
    "PROVIDERS",
]

# --- tuning -----------------------------------------------------------------
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 1  # exactly one retry (two attempts total) per provider

_UA = "RxBridge/0.1 (+https://github.com/rxbridge)"


# --- errors -----------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for all LLM client failures."""


class NoProviderAvailable(LLMError):
    """No configured provider could produce a completion."""


class ProviderError(LLMError):
    """A single provider request failed."""


# --- provider table ---------------------------------------------------------
@dataclass(frozen=True)
class Provider:
    name: str
    kind: str  # "openai" | "gemini"
    base_url: str
    api_key_env: str
    default_model: str
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="featherless",
        kind="openai",
        base_url="https://api.featherless.ai/v1",
        api_key_env="FEATHERLESS_API_KEY",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        extra_headers={
            "HTTP-Referer": "https://github.com/rxbridge",
            "X-Title": "RxBridge",
        },
    ),
    Provider(
        name="cerebras",
        kind="openai",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY_1",
        default_model="qwen-3.8-27b",
    ),
    Provider(
        name="gemini",
        kind="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    ),
)

_BY_NAME = {p.name: p for p in PROVIDERS}
_selected_provider: str | None = None  # memoised once a provider succeeds


# --- secrets ----------------------------------------------------------------
_ENV_FILES = (
    os.environ.get("RXBRIDGE_ENV_FILE"),
    os.path.join(os.getcwd(), ".env"),
    "/Users/faye/.env",
)
_env_cache: dict[str, str] | None = None


def _load_env_files() -> dict[str, str]:
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    values: dict[str, str] = {}
    for path in _ENV_FILES:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    values.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        except OSError:
            continue
    _env_cache = values
    return values


def _get_secret(name: str) -> str | None:
    """Return the secret for ``name`` or ``None``. Value is never logged."""
    val = os.environ.get(name)
    if val:
        return val
    return _load_env_files().get(name) or None


def _redact(text: str) -> str:
    """Best-effort scrub of anything that looks like a credential."""
    if not text:
        return ""
    out = text
    for name in ("FEATHERLESS_API_KEY", "CEREBRAS_API_KEY_1", "GEMINI_API_KEY"):
        secret = _get_secret(name)
        if secret:
            out = out.replace(secret, "***REDACTED***")
    # gemini keys travel in the query string
    if "key=" in out:
        import re

        out = re.sub(r"key=[^&\s\"']+", "key=***REDACTED***", out)
    return out


def _safe_body(resp: httpx.Response, limit: int = 300) -> str:
    try:
        return _redact(resp.text)[:limit]
    except Exception:  # noqa: BLE001 - never let diagnostics raise
        return "<unreadable body>"


# --- transport --------------------------------------------------------------
def _normalize_messages(messages: Sequence[dict[str, Any]] | str) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user")).lower()
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": str(m.get("content", ""))})
    return out


def _model_for(provider: Provider, model: str | None) -> str:
    """Resolve the model id. Supports ``"provider:model"`` to target a provider."""
    if not model:
        return provider.default_model
    if ":" in model:
        _, _, tail = model.partition(":")
        return tail or provider.default_model
    return model


def _call_openai(
    provider: Provider,
    key: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    url = provider.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": _UA,
    }
    headers.update(provider.extra_headers)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        resp = client.post(url, json=payload, headers=headers)
    if resp.status_code >= 400:
        raise ProviderError(
            f"{provider.name} HTTP {resp.status_code}: {_safe_body(resp)}"
        )
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{provider.name}: malformed response ({exc})") from exc
    return (content or "").strip()


def _call_gemini(
    provider: Provider,
    key: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m["role"] != "system"
    ]
    generation: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    # Gemini 2.5/3.x are thinking models; disable thoughts so small token
    # budgets produce visible text instead of being consumed by reasoning.
    if model.startswith(("gemini-2.5", "gemini-3")):
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{provider.base_url.rstrip('/')}/models/{model}:generateContent"
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        resp = client.post(
            url,
            params={"key": key},
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": _UA},
        )
    if resp.status_code >= 400:
        raise ProviderError(
            f"{provider.name} HTTP {resp.status_code}: {_safe_body(resp)}"
        )
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise ProviderError(f"{provider.name}: no candidates ({_safe_body(resp)})")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    return "".join(str(p.get("text", "")) for p in parts).strip()


def _dispatch(
    provider: Provider,
    key: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if provider.kind == "gemini":
        return _call_gemini(provider, key, messages, model, temperature, max_tokens)
    return _call_openai(provider, key, messages, model, temperature, max_tokens)


def _ordered_providers() -> list[Provider]:
    if _selected_provider and _selected_provider in _BY_NAME:
        first = _BY_NAME[_selected_provider]
        return [first] + [p for p in PROVIDERS if p.name != first.name]
    return list(PROVIDERS)


# --- public API -------------------------------------------------------------
def complete(
    messages: Sequence[dict[str, Any]] | str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> str:
    """Return a completion string, auto-selecting the first working provider.

    Tries providers in priority order (a previously successful provider is
    tried first). Each provider gets exactly one retry. Raises
    :class:`NoProviderAvailable` if every provider fails.
    """
    global _selected_provider
    msgs = _normalize_messages(messages)
    last_error: Exception | None = None

    for provider in _ordered_providers():
        key = _get_secret(provider.api_key_env)
        if not key:
            last_error = NoProviderAvailable(
                f"{provider.name}: {provider.api_key_env} not configured"
            )
            continue
        chosen = _model_for(provider, model)
        for _attempt in range(MAX_RETRIES + 1):
            try:
                text = _dispatch(
                    provider, key, msgs, chosen, temperature, max_tokens
                )
            except ProviderError as exc:
                last_error = exc
                continue
            except httpx.HTTPError as exc:
                last_error = ProviderError(
                    f"{provider.name}: transport error {type(exc).__name__}"
                )
                continue
            if text:
                _selected_provider = provider.name
                return text
            last_error = ProviderError(f"{provider.name}: empty completion")

    raise NoProviderAvailable(
        f"all providers failed; last error: {last_error}"
    ) from last_error


def probe_provider(provider: Provider) -> dict[str, Any]:
    """Issue exactly ONE minimal request. No retries. Returns a result dict."""
    result: dict[str, Any] = {
        "name": provider.name,
        "model": provider.default_model,
        "status": "FAIL",
        "latency": 0.0,
        "chars": 0,
        "sample": "",
        "error": "",
    }
    key = _get_secret(provider.api_key_env)
    if not key:
        result["status"] = (
            "UNAVAILABLE_NEEDS_REGISTRATION"
            if provider.name == "featherless"
            else "FAIL"
        )
        result["error"] = f"{provider.api_key_env} not configured"
        return result

    messages = [{"role": "user", "content": "Reply with the single word OK."}]
    start = time.perf_counter()
    try:
        text = _dispatch(provider, key, messages, provider.default_model, 0.0, 16)
    except Exception as exc:  # noqa: BLE001 - report, never raise from a probe
        result["latency"] = round(time.perf_counter() - start, 3)
        result["error"] = _redact(str(exc))[:200]
        return result
    result["latency"] = round(time.perf_counter() - start, 3)
    result["chars"] = len(text)
    result["sample"] = text[:80]
    result["status"] = "OK" if text.strip() else "FAIL"
    if not text.strip():
        result["error"] = "empty completion"
    return result


# --- explanation stage ------------------------------------------------------
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "ht": "Haitian Creole",
    "ar": "Arabic",
    "zh": "Chinese",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "ko": "Korean",
    "ru": "Russian",
    "tl": "Tagalog",
    "so": "Somali",
    "am": "Amharic",
    "pl": "Polish",
    "de": "German",
    "it": "Italian",
    "ne": "Nepali",
}

# Languages we actively trust the model to render. Anything else (or an unknown
# code) falls back to English while keeping drug names unchanged.
_SUPPORTED_LANGS = set(_LANG_NAMES) - {"en"}


def _resolve_language(target_language: str | None) -> tuple[str, bool]:
    """Return (language_name, fell_back_to_english)."""
    code = (target_language or "").strip().lower()
    if code in _SUPPORTED_LANGS:
        return _LANG_NAMES[code], False
    if code == "en":
        return "English", False
    return "English", True


def _build_prompt(
    schedule: dict[str, Any], language: str, fell_back: bool
) -> list[dict[str, str]]:
    system = (
        "You are helping a patient with low literacy or a language barrier. "
        "Given this prescription schedule as JSON, write 2-3 short sentences in "
        f"{language} that a patient can hear read aloud and understand. No medical "
        "advice, no new instructions, no disclaimers. Use very simple words. "
        "Include the drug name, how much, and when. Keep every drug name exactly "
        "as written; do not translate or re-spell it."
    )
    if fell_back:
        system += (
            " The requested language is not supported; write the sentences in "
            "English, but keep every drug name exactly as written in the JSON."
        )
    user = json.dumps(schedule, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _template_explain(schedule: dict[str, Any], language: str) -> str:
    """Deterministic, LLM-free English fallback (keeps drug names verbatim)."""
    parts: list[str] = []
    for drug in schedule.get("drugs", []) or []:
        if not isinstance(drug, dict):
            continue
        name = drug.get("drug") or drug.get("name") or "your medicine"
        dose = str(drug.get("dose", "") or "")
        unit = str(drug.get("dose_unit", "") or "")
        amount = f"{dose} {unit}".strip()
        times = drug.get("times_of_day") or []
        when = (
            ", ".join(str(t) for t in times)
            if times
            else str(drug.get("frequency", "") or "")
        )
        sentence = f"Take {name}"
        if amount:
            sentence += f" {amount}"
        if when:
            sentence += f" {when}"
        parts.append(sentence + ".")
    if not parts:
        return "No prescription schedule is available."
    return " ".join(parts)


def explain_schedule(schedule: dict[str, Any], target_language: str) -> str:
    """Build the RxBridge prompt and return the patient-facing sentence.

    Falls back to English (drug names unchanged) for unsupported languages, and
    to a deterministic template if every LLM provider is unavailable.
    """
    schedule = schedule or {}
    code = target_language or str(schedule.get("target_language") or "en")
    language, fell_back = _resolve_language(code)
    messages = _build_prompt(schedule, language, fell_back)
    try:
        text = complete(messages, temperature=0.1, max_tokens=256)
    except LLMError:
        return _template_explain(schedule, language)
    return text or _template_explain(schedule, language)
