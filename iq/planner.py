"""Stage 0 -- turn a natural-language query into a machine-usable QuerySpec.

Exactly one LLM call per query, regardless of database size. That is what
makes the economics work: the expensive reasoning about *what the user means*
happens once, and its output is then applied to every company for free.

The single most important thing this stage produces is `ideal_profile`: a
synthetic description of the company the user is imagining. We embed that
instead of the raw query (the HyDE trick). This is the direct fix for the
failure the brief calls out -- "companies supplying packaging for cosmetics
brands" embeds close to cosmetics brands, but a *synthetic packaging
manufacturer profile* embeds close to packaging manufacturers.
"""
from __future__ import annotations

import datetime
import re
from typing import Any

from . import config, geo
from .llm import Gemini, LLMError
from .schema import Criterion, NumericRange, QuerySpec

PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "complexity": {"type": "string", "enum": ["STRUCTURED", "SEMANTIC", "REASONING"]},
        "intent_summary": {"type": "string"},
        "role_statement": {"type": "string"},
        "ideal_profile": {"type": "string"},
        "geography": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Region words, country names or ISO-2 codes. Empty if unconstrained.",
        },
        "employee_min": {"type": "integer", "nullable": True},
        "employee_max": {"type": "integer", "nullable": True},
        "revenue_min_usd": {"type": "number", "nullable": True},
        "revenue_max_usd": {"type": "number", "nullable": True},
        "founded_after": {"type": "integer", "nullable": True},
        "founded_before": {"type": "integer", "nullable": True},
        "is_public": {"type": "boolean", "nullable": True},
        "naics_prefixes": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "negative_keywords": {"type": "array", "items": {"type": "string"}},
        "disqualifiers": {"type": "array", "items": {"type": "string"}},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "weight": {"type": "number"},
                    "must_have": {"type": "boolean"},
                },
                "required": ["name", "description", "weight", "must_have"],
            },
        },
    },
    "required": [
        "complexity", "intent_summary", "role_statement", "ideal_profile",
        "geography", "naics_prefixes", "keywords", "criteria", "disqualifiers",
    ],
}

SYSTEM = """You translate business search queries into structured search plans \
over a database of company profiles. You are precise, literal about explicit \
constraints, and thoughtful about implicit ones. You never invent constraints \
the user did not express."""

PROMPT = """Today is {today}.

Convert this query into a search plan.

QUERY: {query}

The companies being searched have these fields: operational_name, website,
year_founded, address (country/region/town), employee_count, revenue (USD),
primary_naics + secondary_naics (code and label), description, business_model
(e.g. B2B, SaaS, Manufacturing, Marketplace), target_markets (industries
served), core_offerings (products/services provided), is_public.

Produce:

1. complexity -- how much reasoning does qualifying a company require?
   STRUCTURED: almost entirely checkable from fields (country, headcount,
     revenue, public/private, an obvious industry).
     e.g. "Public software companies with more than 1,000 employees"
   SEMANTIC: needs the description understood, but the target role is direct.
     e.g. "B2B SaaS companies providing HR solutions in Europe"
   REASONING: requires inferring a company's ROLE in an ecosystem, a supply
     chain relationship, or a subjective judgement.
     e.g. "Companies that could supply packaging for a D2C cosmetics brand",
          "Fast-growing fintechs competing with traditional banks"

2. role_statement -- one sentence stating what a matching company must
   ACTUALLY DO or BE. Write it from the perspective of the company that
   QUALIFIES, never the customer it serves.
   Make it exactly as broad as the query, never narrower. Do not add
   qualifiers the user did not write. "Software companies" covers IT services
   firms, custom development shops and systems integrators, not only vendors
   of proprietary software products -- adding a word like "proprietary" or
   "primarily" silently excludes most of the real answers. Narrow only on
   what the query actually says.
   CRITICAL for supply-chain queries. For "companies that could supply
   packaging materials for a D2C cosmetics brand", the role_statement is
   "manufactures or distributes primary and secondary packaging such as
   bottles, jars, tubes, closures, cartons or labels" -- it is NOT "sells
   cosmetics". The cosmetics brand is the CUSTOMER, not the answer.

3. ideal_profile -- 60-110 words written as if it were a real company's
   database record: what it makes or does, its industry, its offerings, its
   business model, who it sells to. This text is embedded and matched against
   real company records, so write it in the register of a company description,
   not as a restatement of the query. Do not name the country. Do not use the
   words "looking for" or "companies that".

4. geography -- region words ("Europe", "Scandinavia", "DACH"), country names
   or ISO-2 codes, exactly as constrained by the query. Empty array if the
   query names no location. Never guess a location from the industry.

5. Numeric constraints, only where the query states them. Convert units:
   "$50 million" -> 50000000. "more than 1,000 employees" -> employee_min 1001.
   "fewer than 200 employees" -> employee_max 199. "founded after 2018" ->
   founded_after 2019. Use null for anything unstated. Do NOT invent a
   headcount cap for the word "startup" or a revenue floor for "large" unless
   the query gives a number -- express those as soft criteria instead.

6. naics_prefixes -- 2 to 6 digit NAICS prefixes a matching company would
   plausibly carry. Prefer 3-4 digit prefixes for recall. Include every
   plausible branch: a packaging supplier could be 3221 (paper), 3261
   (plastics), 3272 (glass), 3315/3329 (metal), 4241 (wholesale).

7. keywords -- 8-15 terms likely to appear in a matching company's
   description or core_offerings. Use industry vocabulary, not query words.

8. negative_keywords -- terms that signal a near-miss: topically adjacent
   companies that must NOT qualify.

9. disqualifiers -- 2-4 short rules naming the specific near-miss categories a
   careless matcher would wrongly accept. Be concrete.
   e.g. "A software vendor selling logistics management tools is not a
   logistics company -- it is a software company."

10. criteria -- 2-5 independently checkable requirements, each with a weight
    (0-1, summing to roughly 1) and must_have. Mark must_have=true only for
    requirements the query states or plainly implies; a company failing a
    must_have criterion cannot qualify. Keep criteria about the company's
    nature and role; constraints already captured as numeric filters above do
    not need to be repeated here."""


def _range(lo: Any, hi: Any) -> NumericRange:
    def num(v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f

    return NumericRange(minimum=num(lo), maximum=num(hi))


def build_spec(query: str, data: dict[str, Any]) -> QuerySpec:
    """Assemble a QuerySpec from raw planner output, applying safety nets."""
    countries = geo.resolve([str(g) for g in data.get("geography") or []])
    phrase = ", ".join(str(g) for g in (data.get("geography") or []))

    # Safety net: the planner occasionally drops an obvious geography. If it
    # returned none but the query plainly names a place, apply it anyway.
    if not countries:
        scanned, scanned_phrase = geo.extract_from_text(query)
        if scanned:
            countries, phrase = scanned, scanned_phrase

    criteria = []
    for c in data.get("criteria") or []:
        try:
            criteria.append(
                Criterion(
                    name=str(c.get("name", "")).strip() or "criterion",
                    description=str(c.get("description", "")).strip(),
                    weight=float(c.get("weight", 1.0) or 0.0),
                    must_have=bool(c.get("must_have", False)),
                )
            )
        except (TypeError, ValueError):
            continue

    prefixes = [
        str(p).strip() for p in (data.get("naics_prefixes") or [])
        if str(p).strip().isdigit()
    ]

    complexity = str(data.get("complexity", "SEMANTIC")).upper()
    if complexity not in {"STRUCTURED", "SEMANTIC", "REASONING"}:
        complexity = "SEMANTIC"

    pub = data.get("is_public")
    return QuerySpec(
        query=query,
        complexity=complexity,
        intent_summary=str(data.get("intent_summary", "")).strip(),
        countries=sorted(countries),
        country_phrase=phrase,
        employees=_range(data.get("employee_min"), data.get("employee_max")),
        revenue_usd=_range(data.get("revenue_min_usd"), data.get("revenue_max_usd")),
        founded_year=_range(data.get("founded_after"), data.get("founded_before")),
        is_public=pub if isinstance(pub, bool) else None,
        ideal_profile=str(data.get("ideal_profile", "")).strip(),
        role_statement=str(data.get("role_statement", "")).strip(),
        naics_prefixes=prefixes,
        keywords=[str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()],
        negative_keywords=[
            str(k).strip() for k in (data.get("negative_keywords") or []) if str(k).strip()
        ],
        criteria=criteria,
        disqualifiers=[
            str(d).strip() for d in (data.get("disqualifiers") or []) if str(d).strip()
        ],
    )


_STOPWORDS = frozenset(
    """a an and are as at be by for in is of on or that the to with companies
    company business firms firm that which who providing provides provide
    more than over under fewer less least most using used use""".split()
)


def heuristic_spec(query: str) -> QuerySpec:
    """A deterministic, LLM-free fallback plan.

    The planner is a single point of failure: one 429 on the plan call used to
    raise, and because the queries ran under a single `asyncio.gather`, one
    failed plan destroyed all twelve results. That is a bad trade -- the rest
    of the pipeline works perfectly well from a crude plan, just less
    precisely.

    So when planning fails we degrade instead of collapsing: geography is
    still resolved from the static table, numeric constraints are read off the
    query with regexes, and the query text itself becomes the retrieval probe.
    Precision drops; the system still answers. The result is marked
    SEMANTIC and carries no rubric, so downstream stages know they are working
    without one.
    """
    countries, phrase = geo.extract_from_text(query)
    low = query.lower()

    def find(*patterns: str) -> float | None:
        for pattern in patterns:
            m = re.search(pattern, low)
            if m:
                raw = m.group(1).replace(",", "")
                try:
                    value = float(raw)
                except ValueError:
                    continue
                scale = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                if scale and scale.startswith("b"):
                    value *= 1_000_000_000
                elif scale and scale.startswith("m"):
                    value *= 1_000_000
                elif scale and scale.startswith("k"):
                    value *= 1_000
                return value
        return None

    employees = NumericRange()
    revenue = NumericRange()
    founded = NumericRange()

    emp_min = find(r"(?:more than|over|at least|above|>)\s*([\d,]+)\s*(?:\+)?\s*employees")
    emp_max = find(r"(?:fewer than|less than|under|below|<)\s*([\d,]+)\s*employees")
    if emp_min is not None:
        employees.minimum = emp_min + 1
    if emp_max is not None:
        employees.maximum = emp_max - 1

    rev_min = find(r"revenue[^.]{0,24}?(?:over|above|more than|exceeding|>)\s*\$?\s*([\d.,]+)\s*(billion|million|thousand|bn|m|k)?",
                   r"(?:over|above|more than)\s*\$\s*([\d.,]+)\s*(billion|million|thousand|bn|m|k)?")
    if rev_min is not None:
        revenue.minimum = rev_min

    year_after = find(r"(?:founded|established|started)\s*(?:after|since|in or after)\s*(\d{4})")
    year_before = find(r"(?:founded|established)\s*(?:before|prior to)\s*(\d{4})")
    if year_after is not None:
        founded.minimum = year_after + 1
    if year_before is not None:
        founded.maximum = year_before - 1

    is_public: bool | None = None
    if re.search(r"\bpublic(?:ly)?(?:\straded)?\b", low):
        is_public = True
    elif re.search(r"\bprivate(?:ly held)?\b", low):
        is_public = False

    keywords = [w for w in re.findall(r"[a-z0-9-]{3,}", low) if w not in _STOPWORDS]

    return QuerySpec(
        query=query,
        complexity="SEMANTIC",
        intent_summary="heuristic plan (planner unavailable)",
        countries=sorted(countries),
        country_phrase=phrase,
        employees=employees,
        revenue_usd=revenue,
        founded_year=founded,
        is_public=is_public,
        ideal_profile=query,
        role_statement=query,
        keywords=keywords[:15],
        criteria=[],
        disqualifiers=[],
    )


async def plan_query(client: Gemini, query: str) -> QuerySpec:
    try:
        data = await client.generate_json(
            config.PLANNER,
            PROMPT.format(query=query, today=datetime.date.today().isoformat()),
            PLANNER_SCHEMA,
            system=SYSTEM,
            namespace="planner",
            thinking="high",
        )
    except LLMError:
        return heuristic_spec(query)
    return build_spec(query, data)
