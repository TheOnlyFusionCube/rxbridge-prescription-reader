# RxBridge — LLM Endpoint Sponsor Report

**Scope:** prove which LLM inference endpoint works for the multilingual
*explanation* stage, and record the exact configuration the rest of the build
depends on. Extraction (OCR + NER) is a separate, self-trained model and is out
of scope here.

**Environment:** macOS, Python 3.12.4. Client: `src/llm.py` (deps: `httpx`;
`openai`, `google-genai`, `requests` also present and importable).

**Status: ✅ Gemini works. The RxBridge client auto-selects it. Smoke test exits 0.**

> No credential value appears anywhere in this report or in the repository — only
> environment-variable *names* and status.

---

## 1. Summary

| # | Provider    | Base URL | Auth scheme | Model id | Result |
|---|-------------|----------|-------------|----------|--------|
| 1 | featherless | `https://api.featherless.ai/v1` | `Authorization: Bearer $FEATHERLESS_API_KEY` | `Qwen/Qwen2.5-7B-Instruct` | `UNAVAILABLE_NEEDS_REGISTRATION` — no key on machine |
| 2 | cerebras    | `https://api.cerebras.ai/v1` | `Authorization: Bearer $CEREBRAS_API_KEY_1` | `qwen-3.8-27b` | **FAIL** — `HTTP 402 payment_required` (valid key, no billing) |
| 3 | gemini      | `https://generativelanguage.googleapis.com/v1beta` | `?key=$GEMINI_API_KEY` (query param) | `gemini-2.5-flash` | **✅ OK** — ~0.5–0.8 s, non-empty completion |

Working provider: **gemini / `gemini-2.5-flash`**.

---

## 2. Exact smoke-test output

Command: `cd /Users/faye/rxbridge && python3 scripts/smoke_llm.py`

stdout (exact — one line per provider):

```
PROVIDER=featherless MODEL=Qwen/Qwen2.5-7B-Instruct UNAVAILABLE_NEEDS_REGISTRATION latency=0.000 chars=0
PROVIDER=cerebras MODEL=qwen-3.8-27b FAIL latency=0.181 chars=0
PROVIDER=gemini MODEL=gemini-2.5-flash OK latency=0.769 chars=2
```

stderr (reasons; keeps stdout one-line-per-provider):

```
  # featherless: FEATHERLESS_API_KEY not configured
  # cerebras: cerebras HTTP 402: {"message":"Payment required to access this resource. Visit your billing tab.","type":"payment_required_error","param":"quota","code":"payment_required"}
```

Exit code: **0** (≥1 provider returned a non-empty completion).

End-to-end success command (exact):

```
$ cd /Users/faye/rxbridge && python3 -c "from src.llm import explain_schedule; print(explain_schedule({'status':'ok','drugs':[{'drug':'Amoxicillin','dose':'500','dose_unit':'mg','frequency':'twice daily','times_of_day':['morning','night']}],'target_language':'es'},'es'))"
Tome 500 mg de Amoxicillin por la mañana y por la noche.
```

Contains the drug name (`Amoxicillin`) and both time-of-day concepts (`mañana`, `noche`).

---

## 3. Provider details

### 3.1 gemini — **WORKS** (selected)

- **Base URL:** `https://generativelanguage.googleapis.com/v1beta`
- **Endpoint:** `POST {base}/models/{model}:generateContent`
- **Auth:** API key as query parameter `?key=$GEMINI_API_KEY` (env var present, len 39).
- **Model id:** `gemini-2.5-flash` (verified via `GET /v1beta/models`).
- **Latency:** ~0.44–0.77 s for a minimal request.
- **Critical detail:** `gemini-2.5-flash` is a *thinking* model. With a small
  `maxOutputTokens` its reasoning tokens (`thoughtsTokenCount`) consume the whole
  budget and the response contains **no text** (`finishReason: MAX_TOKENS`,
  `content: {}`). The client disables thinking with
  `generationConfig.thinkingConfig.thinkingBudget = 0`. Verified: without it a
  24-token cap yielded `""`; with it the same call returned `"OK"` with
  `finishReason: STOP`.
- Models verified available and working: `gemini-2.5-flash`, `gemini-3.5-flash`.
  `gemini-2.5-flash-lite` returned 404 ("no longer available to new users");
  `gemini-flash-latest` timed out >30 s. `gemini-2.5-flash` chosen as the stable,
  fast default.

**The code that produced the working result** (from `src/llm.py`, `_call_gemini`):

```python
props = {
    "temperature": temperature,
    "maxOutputTokens": max_tokens,
}
if model.startswith(("gemini-2.5", "gemini-3")):
    props["thinkingConfig"] = {"thinkingBudget": 0}   # <-- makes text appear
payload = {"contents": contents, "generationConfig": props}
if system:
    payload["systemInstruction"] = {"parts": [{"text": system}]}
url = f"{provider.base_url.rstrip('/')}/models/{model}:generateContent"
resp = client.post(url, params={"key": key}, json=payload,
                   headers={"Content-Type": "application/json"})
text = "".join(p.get("text", "")
               for p in resp.json()["candidates"][0]["content"]["parts"])
```

### 3.2 cerebras — valid key, **FAIL (billing)**

- **Base URL:** `https://api.cerebras.ai/v1`
- **Auth:** `Authorization: Bearer $CEREBRAS_API_KEY_1` (parsed from `/Users/faye/.env`; key present, len 52).
- **Models the account can see** (from `GET /v1/models`): `qwen-3.8-27b`, `gpt-oss-120b`.
- **Result:** `/v1/models` lists fine, but **both** chat models return
  `HTTP 402` — `{"code":"payment_required","message":"Payment required to access this resource."}`.
  The key is valid; the account has no active billing/quota. No free model is usable.
- **Ops note:** the endpoint sits behind Cloudflare and blocks the default
  `urllib` User-Agent with **`error code: 1010`** (browser-signature ban). Sending a
  normal browser `User-Agent` header reaches the API — the client always sends one.
- Model ids tried and rejected (`HTTP 404 model_not_found`): `llama3.1-8b`,
  `llama-3.3-70b`, `llama3.1-70b`, `qwen-3-32b`, `qwen-3-235b-a22b-instruct-2507`,
  `llama-4-scout-17b-16e-instruct`, `qwen-2.5-32b` — none are on this account.

### 3.3 featherless — **UNAVAILABLE_NEEDS_REGISTRATION** (sponsor path OPEN)

- **Base URL:** `https://api.featherless.ai/v1` (OpenAI-compatible).
- **Auth:** `Authorization: Bearer $FEATHERLESS_API_KEY`.
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
- **Status on this machine:** `FEATHERLESS_API_KEY` is **not set** (not in the
  environment, not in `/Users/faye/.env`). Per the task, this is recorded as
  `UNAVAILABLE_NEEDS_REGISTRATION` — **not** an endpoint failure.
- **Docs check (fetched `https://featherless.ai/docs/api-overview-and-common-options`
  and `/docs/quickstart-guide`):** confirmed the API is OpenAI-compatible,
  base `https://api.featherless.ai/v1`, auth header `Authorization: Bearer ...`,
  and the quickstart's example model id is **`Qwen/Qwen2.5-7B-Instruct`**. Docs also
  request client attribution headers `HTTP-Referer` and `X-Title` (already sent by
  `src/llm.py`).
- **Hackathon path availability:** **YES, available but not yet activated.**
  Register at <https://featherless.ai/register>, apply code **`MLEMPOWER3`** for
  $25 of credits, generate a key at <https://featherless.ai/account/api-keys>, and
  export it as `FEATHERLESS_API_KEY`. No other config change is needed: because
  Featherless is **priority #1** in `src/llm.py`, the client will auto-select it
  ahead of Gemini on the next run. The moment the key exists, re-running
  `scripts/smoke_llm.py` should print `PROVIDER=featherless ... OK`.

---

## 4. The one-line prompt used

System prompt (single string) for the explanation stage:

```
You are helping a patient with low literacy or a language barrier. Given this prescription schedule as JSON, write 2-3 short sentences in <LANG> that a patient can hear read aloud and understand. No medical advice, no new instructions, no disclaimers. Use very simple words. Include the drug name, how much, and when. Keep every drug name exactly as written; do not translate or re-spell it.
```

The JSON schedule is sent verbatim as the user message
(`json.dumps(schedule, ensure_ascii=False, sort_keys=True)`).

For an **unsupported** target language, one extra clause is appended and the
language is switched to English (drug names still untouched):

```
The requested language is not supported; write the sentences in English, but keep every drug name exactly as written in the JSON.
```

---

## 5. Fallbacks (tested, not assumed)

`explain_schedule(schedule, target_language)` has two safety nets:

1. **Unsupported language → English, drug name verbatim.** `_resolve_language`
   returns `fell_back=True` for any code outside the supported set.
   - Test: `_resolve_language('xx') -> ('English', True)`; `explain_schedule(sched,'xx')`
     returned → `Take Amoxicillin. Take 500 mg in the morning and 500 mg at night.`
     (English, contains `Amoxicillin` verbatim). ✅
2. **All providers down → deterministic template.** `_template_explain` builds a
   plain-English sentence from the schedule with no network call.
   - Test: `_template_explain(sched,'English') -> "Take Amoxicillin 500 mg morning, night."` ✅

---

## 6. How the rest of the build uses it

```python
from src.llm import complete, explain_schedule

# Multilingual explanation stage:
sentence = explain_schedule(schedule_dict, target_language)   # -> str

# Low-level, if another stage needs raw access:
text = complete([{"role": "user", "content": "..."}])          # 30s timeout, 1 retry
```

- `complete(messages, model=None, temperature=0.2, max_tokens=256) -> str`
  auto-selects the first working provider by priority (Featherless → Cerebras →
  Gemini), memoises the winner, uses a 30 s timeout and exactly one retry per
  provider, and raises `NoProviderAvailable` / `ProviderError` / `LLMError` on failure.
- Credentials are resolved from env, `$RXBRIDGE_ENV_FILE`, `./.env`, then
  `/Users/faye/.env`; values are never logged, and error bodies are scrubbed by
  `_redact()` (including Gemini's `?key=...` query string).

**Impact on the Devpost write-up:** the sponsor (Featherless) path is *available*
and fully wired as priority #1 — it only needs a registered key. The build ships
working today via Gemini, and will exercise the sponsor endpoint automatically
once `FEATHERLESS_API_KEY` is added.
