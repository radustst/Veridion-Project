"""Typed objects passed between pipeline stages.

Each stage consumes one of these and produces the next, so any stage can be
tested in isolation by handing it a literal object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Tri(str, Enum):
    """Three-valued constraint outcome.

    UNKNOWN is the whole point: 38% of this dataset has no employee_count and
    27% has no year_founded. Treating missing data as failure silently deletes
    correct answers; treating it as success floods the results. UNKNOWN keeps
    the company alive, flags it, and lets a later stage try to infer the value.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Verdict(str, Enum):
    QUALIFIED = "QUALIFIED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


@dataclass
class Naics:
    code: str
    label: str
    share: Optional[float] = None


@dataclass
class Company:
    idx: int
    operational_name: Optional[str] = None
    website: Optional[str] = None
    year_founded: Optional[int] = None
    country_code: Optional[str] = None
    region_name: Optional[str] = None
    town: Optional[str] = None
    employee_count: Optional[int] = None
    revenue: Optional[float] = None
    primary_naics: Optional[Naics] = None
    secondary_naics: list[Naics] = field(default_factory=list)
    description: str = ""
    business_model: list[str] = field(default_factory=list)
    target_markets: list[str] = field(default_factory=list)
    core_offerings: list[str] = field(default_factory=list)
    is_public: Optional[bool] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.operational_name or self.website or f"company-{self.idx}"

    @property
    def naics_codes(self) -> list[str]:
        codes = [self.primary_naics.code] if self.primary_naics else []
        codes += [n.code for n in self.secondary_naics]
        return [c for c in codes if c]

    def location(self) -> str:
        bits = [b for b in (self.town, self.region_name) if b]
        if self.country_code:
            bits.append(self.country_code.upper())
        return ", ".join(bits) if bits else "unknown location"

    def completeness(self) -> float:
        """Fraction of the discriminative fields that are actually present."""
        present = [
            self.year_founded, self.country_code, self.employee_count, self.revenue,
            self.primary_naics, self.description or None, self.is_public,
            self.business_model or None, self.core_offerings or None,
        ]
        return sum(f is not None for f in present) / len(present)


@dataclass
class NumericRange:
    """An inclusive numeric constraint; either end may be open."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def is_set(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    def check(self, value: Optional[float]) -> Tri:
        if value is None:
            return Tri.UNKNOWN
        if self.minimum is not None and value < self.minimum:
            return Tri.FAIL
        if self.maximum is not None and value > self.maximum:
            return Tri.FAIL
        return Tri.PASS

    def describe(self, unit: str = "") -> str:
        if self.minimum is not None and self.maximum is not None:
            return "{:g}-{:g}{}".format(self.minimum, self.maximum, unit)
        if self.minimum is not None:
            return ">={:g}{}".format(self.minimum, unit)
        if self.maximum is not None:
            return "<={:g}{}".format(self.maximum, unit)
        return "any"


@dataclass
class Criterion:
    """One weighted, independently checkable requirement from the query."""

    name: str
    description: str
    weight: float = 1.0
    must_have: bool = True


@dataclass
class QuerySpec:
    """The planner's structured reading of a natural-language query."""

    query: str
    complexity: str = "SEMANTIC"  # STRUCTURED | SEMANTIC | REASONING
    intent_summary: str = ""

    # --- hard, machine-checkable constraints ---
    countries: list[str] = field(default_factory=list)  # ISO-2, already expanded
    country_phrase: str = ""  # the geography as the user wrote it
    employees: NumericRange = field(default_factory=NumericRange)
    revenue_usd: NumericRange = field(default_factory=NumericRange)
    founded_year: NumericRange = field(default_factory=NumericRange)
    is_public: Optional[bool] = None

    # --- soft, semantic signals ---
    ideal_profile: str = ""       # HyDE: a synthetic ideal company description
    role_statement: str = ""      # what the company must DO, not who it serves
    naics_prefixes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)

    def has_hard_constraints(self) -> bool:
        return bool(
            self.countries
            or self.is_public is not None
            or self.employees.is_set()
            or self.revenue_usd.is_set()
            or self.founded_year.is_set()
        )


@dataclass
class FilterOutcome:
    """Result of applying hard constraints to one company."""

    company: Company
    per_constraint: dict[str, Tri] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(v is Tri.FAIL for v in self.per_constraint.values())

    @property
    def unknowns(self) -> list[str]:
        return [k for k, v in self.per_constraint.items() if v is Tri.UNKNOWN]


@dataclass
class Candidate:
    """A company that survived filtering, with its retrieval evidence."""

    company: Company
    dense_score: float = 0.0
    bm25_score: float = 0.0
    naics_score: float = 0.0
    fused_score: float = 0.0
    unknown_constraints: list[str] = field(default_factory=list)


@dataclass
class Qualification:
    """The LLM's judgement on one candidate."""

    verdict: Verdict = Verdict.REJECTED
    confidence: float = 0.0
    reason: str = ""
    criteria_met: dict[str, bool] = field(default_factory=dict)
    inferred: dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    tier: str = ""


@dataclass
class RankedCompany:
    company: Company
    score: float
    qualification: Qualification
    candidate: Candidate

    def to_dict(self) -> dict[str, Any]:
        naics = None
        if self.company.primary_naics:
            naics = "{} {}".format(
                self.company.primary_naics.code, self.company.primary_naics.label
            )
        return {
            "rank_score": round(self.score, 4),
            "name": self.company.name,
            "website": self.company.website,
            "location": self.company.location(),
            "naics": naics,
            "employee_count": self.company.employee_count,
            "revenue_usd": self.company.revenue,
            "year_founded": self.company.year_founded,
            "is_public": self.company.is_public,
            "verdict": self.qualification.verdict.value,
            "confidence": round(self.qualification.confidence, 3),
            "reason": self.qualification.reason,
            "criteria_met": self.qualification.criteria_met,
            "unverifiable": self.candidate.unknown_constraints,
            "escalated": self.qualification.escalated,
            "decided_by": self.qualification.tier,
            "retrieval_score": round(self.candidate.fused_score, 4),
        }


@dataclass
class Usage:
    """Token/cost accounting, aggregated across a run."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    cached_calls: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.embed_tokens += other.embed_tokens
        self.cached_calls += other.cached_calls
        self.cost_usd += other.cost_usd
        for k, v in other.by_model.items():
            self.by_model[k] = self.by_model.get(k, 0) + v

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "embed_tokens": self.embed_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls_by_model": self.by_model,
        }


@dataclass
class QueryResult:
    query: str
    spec: QuerySpec
    companies: list[RankedCompany]
    usage: Usage
    timings: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    error: str = ""  # set when the query failed outright rather than returning nothing
