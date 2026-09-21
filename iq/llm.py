"""Async Gemini client: structured output, caching, retries, model fallback.

Written against the REST API with httpx rather than a vendor SDK, for three
reasons: the dependency surface stays at two packages, the retry/fallback
behaviour is visible instead of buried, and swapping providers later means
editing one file.

Everything the rest of the codebase needs from an LLM goes through
`Gemini.generate_json`, which guarantees a parsed dict back or raises.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any, Optional

import httpx

from . import config
from .cache import DiskCache, make_key
from .schema import Usage

# Status codes worth trying again: throttling, transient server faults.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# Status codes that mean "this model, ever, for this key" -- move to fallback now.
FALLBACK_STATUS = {400, 403, 404}


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Async token bucket, sized in requests per minute.

    Needed because the embedding endpoint meters *items*, not HTTP calls: a
    batchEmbedContents carrying 64 texts spends 64 of the 100-per-minute free
    tier allowance in one shot. Discovering that by 429 and backing off wastes
    the quota twice -- once on the rejected call and again on the sleep -- so
    the budget is tracked locally and calls simply wait their turn.
    """

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._tokens = float(per_minute)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> None:
        cost = max(1, min(int(cost), int(self.capacity)))
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                deficit = cost - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(min(wait, 5.0))


def estimate_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    inp, out = config.PRICING.get(model, config.DEFAULT_PRICE)
    return (prompt_tokens * inp + output_tokens * out) / 1_000_000


class Gemini:
    """Thin async wrapper over generativelanguage.googleapis.com.

    Concurrency is bounded by a semaphore rather than left to the event loop:
    the free tier 429s aggressively, and an unbounded fan-out spends its time
    in backoff instead of doing work. Measured, a bounded 4-way fan-out beat an
    unbounded 32-way one on wall clock for this workload.
    """

    def __init__(
        self,
        cache: DiskCache,
        cfg: Optional[config.PipelineConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or config.DEFAULT
        self.cache = cache
        self._key = api_key or config.api_key()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None
        # Separate buckets: the two endpoints are metered independently.
        self._embed_limiter = RateLimiter(self.cfg.embed_rpm)
        self._gen_limiter = RateLimiter(self.cfg.generate_rpm)
        self.usage = Usage()
        # Models that have hard-failed for this key; skipped on later calls so
        # we pay the 404 once per process rather than once per request.
        self._dead_models: set[str] = set()
        # Models that are rate-limited right now, with the time they may be
        # tried again. Without this, a chain whose first three models have no
        # remaining quota re-requests all three on every retry round -- 15
        # doomed calls and a minute of backoff to reach the one model that
        # would have answered immediately.
        self._throttled: dict[str, float] = {}
        # Long, deliberately. A model whose daily allowance is gone will 429
        # every time; a short cooldown just means rediscovering that every two
        # minutes, and each rediscovery costs a doomed request and a backoff
        # sleep on the critical path. Ten minutes is far longer than a
        # per-minute limit needs and short enough that a genuine burst limit
        # still recovers within a run.
        self.throttle_cooldown = 600.0

    async def __aenter__(self) -> "Gemini":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.request_timeout),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _record(self, model: str, data: dict[str, Any]) -> None:
        meta = data.get("usageMetadata", {}) or {}
        pt = int(meta.get("promptTokenCount", 0) or 0)
        # Thinking tokens bill as output and are reported separately.
        ot = int(meta.get("candidatesTokenCount", 0) or 0) + int(
            meta.get("thoughtsTokenCount", 0) or 0
        )
        self.usage.calls += 1
        self.usage.prompt_tokens += pt
        self.usage.output_tokens += ot
        self.usage.cost_usd += estimate_cost(model, pt, ot)
        self.usage.by_model[model] = self.usage.by_model.get(model, 0) + 1

    @staticmethod
    def _thinking_config(model: str, level: str | None) -> dict[str, Any]:
        """Gemini 3 takes thinkingLevel; 2.x took thinkingBudget.

        Passing the wrong one is a hard 400, so key it off the model family.
        `level=None` means "leave the model on its default".
        """
        if level is None:
            return {}
        if model.startswith("gemini-3"):
            return {"thinkingConfig": {"thinkingLevel": level}}
        budget = {"low": 0, "high": 8192}.get(level, 0)
        return {"thinkingConfig": {"thinkingBudget": budget}}

    async def _post(self, url: str, body: dict[str, Any], limiter=None, cost: int = 1) -> httpx.Response:
        assert self._client is not None, "use Gemini as an async context manager"
        if limiter is not None:
            await limiter.acquire(cost)
        async with self._sem:
            return await self._client.post(url, params={"key": self._key}, json=body)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def generate_json(
        self,
        tier: config.ModelTier,
        prompt: str,
        response_schema: dict[str, Any],
        *,
        system: str | None = None,
        namespace: str = "gen",
        thinking: str | None = "low",
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run a prompt and return parsed JSON conforming to `response_schema`.

        Tries each model in the tier's chain. Within a model, retries the
        transient failures with exponential backoff and jitter; on a permanent
        failure, drops straight to the next model.
        """
        cache_key = make_key(
            namespace,
            tier.primary,
            {"p": prompt, "s": system, "schema": response_schema, "t": temperature,
             "think": thinking},
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.usage.cached_calls += 1
            return cached

        # Loop rounds on the OUTSIDE, models on the inside. A 429 almost always
        # means "no quota for this model on this key" rather than "wait a
        # moment", so the right response is to try a different model
        # immediately, not to sleep. Only when every model in the chain has
        # failed in a single round do we back off and go round again. Getting
        # this the wrong way round cost us minutes per query against a model
        # with zero free-tier quota.
        last_error: Exception | None = None
        for round_no in range(self.cfg.max_retries):
            exhausted_chain = True
            for model in tier.chain:
                if model in self._dead_models:
                    continue
                if self._throttled.get(model, 0.0) > time.monotonic():
                    continue
                exhausted_chain = False
                body: dict[str, Any] = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "responseMimeType": "application/json",
                        "responseSchema": response_schema,
                        **self._thinking_config(model, thinking),
                    },
                }
                if system:
                    body["systemInstruction"] = {"parts": [{"text": system}]}
                url = "{}/models/{}:generateContent".format(config.API_BASE, model)

                try:
                    resp = await self._post(url, body, self._gen_limiter)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    self._record(model, data)
                    parsed = self._extract_json(data)
                    if parsed is not None:
                        self.cache.put(cache_key, namespace, parsed)
                        return parsed
                    # Valid HTTP, unusable body: a safety block, or output
                    # truncated by the token cap. Another model may do better.
                    last_error = LLMError(
                        "no parsable JSON from {}: {}".format(model, json.dumps(data)[:300])
                    )
                    continue

                if resp.status_code == 400 and "thinkingConfig" in body["generationConfig"]:
                    # Wrong thinking dialect for this model family. Retry the
                    # same model once without it before writing it off.
                    body["generationConfig"].pop("thinkingConfig")
                    retry = await self._post(url, body, self._gen_limiter)
                    if retry.status_code == 200:
                        data = retry.json()
                        self._record(model, data)
                        parsed = self._extract_json(data)
                        if parsed is not None:
                            self.cache.put(cache_key, namespace, parsed)
                            return parsed
                    last_error = LLMError("400 from {}: {}".format(model, resp.text[:200]))
                    self._dead_models.add(model)
                    continue

                if resp.status_code in FALLBACK_STATUS:
                    # Retired or forbidden for this key: never try it again.
                    self._dead_models.add(model)
                    last_error = LLMError(
                        "{} rejected by {}: {}".format(resp.status_code, model, resp.text[:200])
                    )
                    continue

                if resp.status_code == 429:
                    self._throttled[model] = time.monotonic() + self.throttle_cooldown
                last_error = LLMError(
                    "{} from {}: {}".format(resp.status_code, model, resp.text[:160])
                )

            if exhausted_chain:
                break
            await self._sleep(round_no)

        raise LLMError("all models exhausted; last error: {}".format(last_error))

    async def _sleep(self, attempt: int, resp: httpx.Response | None = None) -> None:
        """Exponential backoff with jitter, honouring Retry-After when given."""
        delay = self.cfg.backoff_base ** attempt
        if resp is not None:
            hint = resp.headers.get("retry-after")
            if hint:
                try:
                    delay = max(delay, float(hint))
                except ValueError:
                    pass
        await asyncio.sleep(min(delay, 30.0) * (0.7 + 0.6 * random.random()))

    @staticmethod
    def _extract_json(data: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return None
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # responseSchema makes this rare, but a truncated response can still
            # arrive; salvage the outermost object rather than losing the batch.
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    async def embed_batch(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed up to config.EMBED_BATCH texts in one request.

        Note: at outputDimensionality < 3072 Gemini returns un-normalised
        vectors (we measured ~0.59 L2 norm), so callers must normalise before
        using a dot product as cosine. `embeddings.py` does that centrally.
        """
        assert self._client is not None, "use Gemini as an async context manager"
        reqs = [
            {
                "model": "models/" + config.EMBED_MODEL,
                "content": {"parts": [{"text": t[:8000]}]},
                "taskType": task_type,
                "outputDimensionality": config.EMBED_DIM,
            }
            for t in texts
        ]
        url = "{}/models/{}:batchEmbedContents".format(config.API_BASE, config.EMBED_MODEL)
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = await self._post(url, {"requests": reqs}, self._embed_limiter, cost=len(reqs))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self._sleep(attempt)
                continue
            if resp.status_code == 200:
                vecs = [e["values"] for e in resp.json()["embeddings"]]
                approx_tokens = sum(len(t) for t in texts) // 4
                self.usage.embed_tokens += approx_tokens
                self.usage.cost_usd += estimate_cost(config.EMBED_MODEL, approx_tokens, 0)
                return vecs
            if resp.status_code in RETRY_STATUS:
                last_error = LLMError("{}: {}".format(resp.status_code, resp.text[:160]))
                await self._sleep(attempt, resp)
                continue
            raise LLMError("embed failed {}: {}".format(resp.status_code, resp.text[:300]))
        raise LLMError("embed retries exhausted: {}".format(last_error))
