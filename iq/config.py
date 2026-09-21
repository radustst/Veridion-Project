"""Central configuration.

Every knob the pipeline uses lives here so that cost/latency/accuracy
trade-offs can be inspected and changed in one place rather than being
scattered through the code as magic numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Hand-rolled to keep the dependency list at two packages; existing
    environment variables always win so CI can override the file.
    """
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export GEMINI_API_KEY in your shell."
        )
    return key


# --------------------------------------------------------------------------
# Models.
#
# Tiered deliberately: the cheap tier sees every shortlisted company, the
# mid tier sees only the uncertain ones, and the planner runs exactly once
# per query. Fallbacks exist because the public Gemini endpoints rate-limit
# and retire models without notice -- we observed both while building this.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """A preferred model plus ordered fallbacks used on 404/429/503."""

    primary: str
    fallbacks: tuple[str, ...] = ()

    @property
    def chain(self) -> tuple[str, ...]:
        return (self.primary, *self.fallbacks)


# Chosen by probing this key: gemini-3.5-flash and the pro tier return a hard
# 429 (no free-tier quota), and gemini-3.7-flash was flapping 503. The chains
# below are ordered by capability and every entry was verified reachable.
# Every chain ends in a lite model. The larger flash models have a small daily
# free-tier allowance and return a hard 429 once it is gone; a chain that does
# not terminate in something reachable simply fails the stage. When escalation
# falls back this far it is no longer escalating to a *stronger* model, only to
# a different one thinking harder -- the writeup says so rather than pretending
# the tier always bites.
PLANNER = ModelTier("gemini-3.6-flash", ("gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"))
QUALIFIER = ModelTier("gemini-3.5-flash-lite", ("gemini-3.1-flash-lite",))
ESCALATION = ModelTier("gemini-3.6-flash", ("gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"))
JUDGE = ModelTier("gemini-3.8-flash", ("gemini-3.6-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"))

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # 3072 is available; 768 costs 4x less RAM and scores the same here.
EMBED_BATCH = 25

# Approximate published USD/1M-token rates, used for the cost report. These are
# estimates for relative comparison between strategies, not billing figures.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.8-flash": (0.30, 2.50),
    "gemini-embedding-001": (0.15, 0.0),
}
DEFAULT_PRICE = (0.30, 2.50)


# --------------------------------------------------------------------------
# Pipeline behaviour
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # Stage 2: how many candidates survive cheap retrieval and reach an LLM.
    # Keyed by the planner's complexity classification -- this is the direct
    # answer to "simple queries receive the same expensive treatment".
    shortlist_by_complexity: dict[str, int] = field(
        default_factory=lambda: {"STRUCTURED": 30, "SEMANTIC": 55, "REASONING": 80}
    )
    # Retrieval fusion
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_bm25: float = 0.7
    weight_naics: float = 0.6

    # Stage 3: batching + concurrency
    qualify_batch_size: int = 8
    max_concurrency: int = 4

    # Stage 4: escalation band. Only companies the cheap tier is unsure about,
    # or where retrieval and the LLM disagree, pay for a stronger model.
    escalate_enabled: bool = True
    escalate_conf_low: float = 0.35
    escalate_conf_high: float = 0.72
    escalate_max_fraction: float = 0.25  # hard cap on the cost blast radius
    escalate_batch_size: int = 5
    # thinkingLevel for the escalation tier. "high" was measured at 41s and
    # ~6k thinking tokens to adjudicate two companies -- it dominated both the
    # latency and the cost of an otherwise 3-second query, for no measurable
    # accuracy gain on this data. The capability jump between tiers comes from
    # the bigger model, not from letting it ruminate.
    escalate_thinking: str = "low"

    # Stage 5: final ranking
    unknown_field_penalty: float = 0.06  # per unverifiable hard constraint
    retrieval_tiebreak_weight: float = 0.10
    min_score_to_return: float = 0.35
    # If nothing clears the bar, surface this many near misses rather than an
    # empty list. See the zero-result recovery note in ranking.py.
    no_match_fallback: int = 10

    # Rate limits, in requests per minute, kept just under the observed free
    # tier ceilings. embed_content meters each TEXT in a batch as one request
    # (limit 100/min), which is not obvious from the API shape and is the
    # single most surprising constraint in this integration.
    embed_rpm: int = 90
    # Measured, not guessed: the 429 body names the limit, and for
    # gemini-3.5-flash-lite it is 15 requests per minute. The bucket is global
    # while the quota is per-model, so this is set a little above a single
    # model's ceiling -- the fallback chain spreads load across two lite
    # models, and the per-model cooldown absorbs the rest. Set to 110
    # originally, which simply converted the whole allowance into 429s and
    # backoff: the client spent its time being rejected instead of working.
    generate_rpm: int = 22

    # Reliability
    max_retries: int = 5
    backoff_base: float = 1.6
    request_timeout: float = 120.0

    use_cache: bool = True


DEFAULT = PipelineConfig()
