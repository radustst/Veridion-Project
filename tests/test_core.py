"""Tests for the deterministic parts of the pipeline.

Scope is deliberate: everything here runs offline in milliseconds and has a
single correct answer. The LLM stages are covered by the evaluation harness
instead, because asserting on a model's prose in a unit test produces a suite
that fails when a prompt improves.

The cases below are mostly regressions for real bugs found while building:
un-normalised embeddings, Python-repr dicts in a JSON file, and geography
words that an LLM answers differently on different days.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation import metrics
from iq import filters, geo, retrieval
from iq.config import PipelineConfig
from iq.embeddings import l2_normalise
from iq.llm import RateLimiter
from iq.loading import (
    company_card, company_document, dedupe_companies, parse_company,
)
from iq.planner import build_spec, heuristic_spec
from iq.ranking import rank
from iq.schema import (
    Candidate, Company, Criterion, FilterOutcome, Naics, NumericRange,
    Qualification, QuerySpec, Tri, Verdict,
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
RAW_STRINGY = {
    "website": "x.ro",
    "operational_name": "Stringy SRL",
    "year_founded": 2003.0,
    "address": "{'country_code': 'ro', 'region_name': 'Bucharest', 'town': 'Bucharest'}",
    "employee_count": None,
    "revenue": 4.8e7,
    "primary_naics": "{'code': '488510', 'label': 'Freight Transportation Arrangement'}",
    "description": "Freight forwarding.",
    "business_model": ["B2B"],
    "target_markets": ["automotive"],
    "core_offerings": ["freight forwarding"],
    "is_public": False,
    "secondary_naics": None,
}

RAW_DICTY = {
    "operational_name": "Dicty AB",
    "address": {"country_code": "SE", "town": "Malmo"},
    "primary_naics": {"code": "333611", "label": "Turbine Manufacturing"},
    "secondary_naics": [{"code": "221118", "label": "Other Electric Power", "share": 0.3}],
    "employee_count": 150.0,
    "is_public": True,
}


def test_parses_python_repr_strings():
    """The dataset stores some nested fields as Python reprs, not JSON."""
    c = parse_company(RAW_STRINGY, 0)
    assert c.country_code == "ro"
    assert c.town == "Bucharest"
    assert c.primary_naics is not None and c.primary_naics.code == "488510"
    assert c.year_founded == 2003 and isinstance(c.year_founded, int)
    assert c.employee_count is None


def test_parses_real_dicts_and_normalises_country_case():
    c = parse_company(RAW_DICTY, 1)
    assert c.country_code == "se"  # was "SE"
    assert c.employee_count == 150
    assert c.secondary_naics[0].code == "221118"
    assert c.naics_codes == ["333611", "221118"]


def test_missing_fields_never_raise():
    c = parse_company({}, 2)
    assert c.name == "company-2"
    assert c.naics_codes == []
    assert 0.0 <= c.completeness() <= 1.0
    assert company_document(c)      # must not raise on an empty record
    assert "unknown" in company_card(c)


def test_card_prints_unknown_rather_than_omitting():
    """The LLM must be able to tell absent from zero."""
    card = company_card(parse_company(RAW_STRINGY, 0))
    assert "Employees: unknown" in card
    assert "Revenue USD: 48,000,000" in card


def test_dedupe_collapses_repeated_websites():
    """The dataset ships 13 websites twice; CIRKEL Energi's rows are identical."""
    rows = [
        Company(idx=0, operational_name="CIRKEL Energi", website="cirkelenergi.dk", country_code="dk"),
        Company(idx=1, operational_name="CIRKEL Energi", website="cirkelenergi.dk", country_code="dk"),
        Company(idx=2, operational_name="Other", website="other.dk", country_code="dk"),
    ]
    out, dropped = dedupe_companies(rows)
    assert dropped == 1 and len(out) == 2


def test_dedupe_ignores_www_prefix():
    rows = [
        Company(idx=0, operational_name="A", website="www.example.com"),
        Company(idx=1, operational_name="A", website="example.com"),
    ]
    out, dropped = dedupe_companies(rows)
    assert dropped == 1 and len(out) == 1


def test_dedupe_backfills_missing_fields_from_the_dropped_row():
    """Merging must never lose information the duplicate happened to carry."""
    rows = [
        Company(idx=0, operational_name="A", website="a.com", employee_count=50,
                revenue=1.0, year_founded=2001, country_code="se", description="d",
                is_public=False, business_model=["B2B"], core_offerings=["x"]),
        Company(idx=1, operational_name="A", website="a.com", town="Malmo",
                target_markets=["automotive"]),
    ]
    out, dropped = dedupe_companies(rows)
    assert dropped == 1
    assert out[0].employee_count == 50          # kept the richer record
    assert out[0].target_markets == ["automotive"]  # took what only the other had


def test_dedupe_keeps_distinct_companies_sharing_a_name():
    """Same name in different countries is two companies, not one."""
    rows = [
        Company(idx=0, operational_name="ENERCON", country_code="se"),
        Company(idx=1, operational_name="ENERCON", country_code="de"),
    ]
    out, dropped = dedupe_companies(rows)
    assert dropped == 0 and len(out) == 2


def test_dedupe_leaves_unidentifiable_rows_alone():
    rows = [Company(idx=0), Company(idx=1)]
    out, dropped = dedupe_companies(rows)
    assert dropped == 0 and len(out) == 2


# ---------------------------------------------------------------------------
# geography
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["Scandinavia"], {"se", "no", "dk"}),
        (["Nordics"], {"se", "no", "dk", "fi", "is"}),
        (["Germany", "france"], {"de", "fr"}),
        (["DACH"], {"de", "at", "ch"}),
        (["USA"], {"us"}),
        (["uk"], {"gb"}),
        (["global"], set()),
        (["Atlantis"], set()),  # unknown tokens are dropped, not guessed
    ],
)
def test_region_resolution(tokens, expected):
    assert geo.resolve(tokens) == expected


def test_scandinavia_excludes_finland_consistently():
    """A stable definition is the whole reason this is a table, not a prompt."""
    for _ in range(5):
        assert "fi" not in geo.resolve(["Scandinavia"])
    assert "fi" in geo.resolve(["Nordics"])


def test_europe_contains_switzerland_and_uk_but_not_us():
    eu = geo.resolve(["Europe"])
    assert {"ch", "gb", "de", "ro"} <= eu
    assert "us" not in eu


def test_text_scan_prefers_longest_phrase():
    found, phrase = geo.extract_from_text("Construction companies in the United States")
    assert found == {"us"} and phrase == "united states"


# ---------------------------------------------------------------------------
# three-valued filtering
# ---------------------------------------------------------------------------
def _co(**kw) -> Company:
    base = dict(idx=0, operational_name="X", country_code="de", description="d")
    base.update(kw)
    return Company(**base)


def test_missing_numeric_is_unknown_not_fail():
    spec = QuerySpec(query="q", employees=NumericRange(maximum=199))
    out = filters.apply_filters(_co(employee_count=None), spec)
    assert out.per_constraint["employee_count"] is Tri.UNKNOWN
    assert not out.failed
    assert out.unknowns == ["employee_count"]


def test_present_numeric_out_of_range_fails():
    spec = QuerySpec(query="q", employees=NumericRange(maximum=199))
    assert filters.apply_filters(_co(employee_count=900), spec).failed


def test_wrong_country_fails_but_missing_country_is_unknown():
    spec = QuerySpec(query="q", countries=["ro"])
    assert filters.apply_filters(_co(country_code="pl"), spec).failed
    out = filters.apply_filters(_co(country_code=None), spec)
    assert out.per_constraint["country"] is Tri.UNKNOWN and not out.failed


def test_unconstrained_spec_keeps_everyone():
    spec = QuerySpec(query="q")
    survivors, rejected = filters.filter_companies(
        [_co(idx=i, employee_count=None) for i in range(10)], spec
    )
    assert len(survivors) == 10 and rejected == {}


def test_filter_reports_why_companies_died():
    spec = QuerySpec(query="q", countries=["ro"])
    companies = [_co(idx=i, country_code="pl") for i in range(4)] + [_co(idx=9, country_code="ro")]
    survivors, rejected = filters.filter_companies(companies, spec)
    assert len(survivors) == 1
    assert rejected == {"country": 4}


def test_soften_drops_numeric_gates_but_keeps_geography():
    spec = QuerySpec(
        query="q", countries=["de"],
        employees=NumericRange(maximum=199), founded_year=NumericRange(minimum=2019),
    )
    soft = filters.soften(spec)
    assert soft.countries == ["de"]
    assert not soft.employees.is_set() and not soft.founded_year.is_set()
    assert spec.employees.is_set(), "soften must not mutate the original spec"


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def test_l2_normalise_makes_dot_product_cosine():
    """Gemini returns un-normalised vectors below 3072 dims; this is the guard."""
    raw = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    unit = l2_normalise(raw)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    assert np.isclose(float(unit[0] @ unit[0]), 1.0)


def test_l2_normalise_survives_a_zero_vector():
    assert not np.isnan(l2_normalise(np.zeros((1, 4), dtype=np.float32))).any()


@pytest.mark.parametrize(
    "codes,prefixes,expected_sign",
    [
        (["326199"], ["3261"], "high"),
        (["326199"], ["32"], "low"),
        (["541511"], ["3261"], "zero"),
        ([], ["3261"], "zero"),
        (["326199"], [], "zero"),
    ],
)
def test_naics_affinity_grades_by_prefix_length(codes, prefixes, expected_sign):
    score = retrieval.naics_affinity(codes, prefixes)
    if expected_sign == "zero":
        assert score == 0.0
    elif expected_sign == "low":
        assert 0.0 < score < 0.5
    else:
        assert score >= 0.65


def test_naics_affinity_takes_the_best_match():
    assert retrieval.naics_affinity(["541511", "326199"], ["3261"]) >= 0.65


def test_bm25_ranks_the_document_containing_the_term():
    corpus = [
        "freight forwarding customs brokerage warehousing",
        "cosmetics skincare retail brand",
        "glass bottles jars packaging manufacturer",
    ]
    bm = retrieval.BM25(corpus)
    scores = bm.score(retrieval.tokenize("packaging bottles jars"))
    assert int(np.argmax(scores)) == 2


def test_bm25_returns_zeros_for_unknown_terms():
    bm = retrieval.BM25(["alpha beta", "gamma delta"])
    assert not bm.score(retrieval.tokenize("zzzz qqqq")).any()


def test_rrf_rewards_agreement_across_signals():
    # doc 0 is top of both lists, doc 2 is top of neither
    fused = retrieval.rrf([[0, 1, 2], [0, 2, 1]], [1.0, 1.0], k=60, size=3)
    assert int(np.argmax(fused)) == 0


def test_rrf_ignores_a_zero_weighted_signal():
    with_signal = retrieval.rrf([[2, 1, 0], [0, 1, 2]], [1.0, 0.0], k=60, size=3)
    assert int(np.argmax(with_signal)) == 2


def test_retrieve_prefers_the_on_topic_company():
    cfg = PipelineConfig()
    companies = [
        Company(idx=0, operational_name="Glass Pack AB", description="bottles and jars",
                primary_naics=Naics("327213", "Glass Container Manufacturing"),
                core_offerings=["glass bottles", "jars"]),
        Company(idx=1, operational_name="Lumi Cosmetics", description="skincare brand",
                primary_naics=Naics("325620", "Toilet Preparation Manufacturing"),
                core_offerings=["serums"]),
    ]
    docs = [company_document(c) for c in companies]
    spec = QuerySpec(
        query="packaging suppliers", keywords=["bottles", "jars", "packaging"],
        naics_prefixes=["3272"],
        role_statement="manufactures bottles jars and closures",
    )
    survivors = [FilterOutcome(company=c) for c in companies]
    # identical embeddings, so dense contributes nothing and bm25+naics decide
    emb = l2_normalise(np.ones((2, 4), dtype=np.float32))
    out = retrieval.retrieve(survivors, spec, emb, emb[0], docs, top_k=2, cfg=cfg)
    assert out[0].company.operational_name == "Glass Pack AB"


# ---------------------------------------------------------------------------
# planner post-processing
# ---------------------------------------------------------------------------
def test_build_spec_expands_regions_and_clamps_enums():
    spec = build_spec("q", {"complexity": "nonsense", "geography": ["Scandinavia"]})
    assert spec.complexity == "SEMANTIC"
    assert set(spec.countries) == {"se", "no", "dk"}


def test_build_spec_recovers_geography_the_planner_dropped():
    """Safety net: the model sometimes returns an empty geography array."""
    spec = build_spec("Logistic companies in Romania", {"geography": []})
    assert spec.countries == ["ro"]


def test_build_spec_ignores_non_numeric_naics_and_bad_criteria():
    """A malformed criterion is dropped, not coerced into a nameless one.

    Keeping it would put a criterion with no description into the rubric and
    into the scoring denominator, quietly penalising every company for failing
    a requirement that says nothing.
    """
    spec = build_spec("q", {
        "naics_prefixes": ["3261", "not-a-code", ""],
        "criteria": [{"name": "ok", "description": "d", "weight": 0.5, "must_have": True},
                     {"weight": "bad"}],
    })
    assert spec.naics_prefixes == ["3261"]
    assert [c.name for c in spec.criteria] == ["ok"]


def test_build_spec_handles_a_totally_empty_response():
    spec = build_spec("q", {})
    assert spec.complexity == "SEMANTIC" and spec.countries == []
    assert not spec.has_hard_constraints()


# ---------------------------------------------------------------------------
# heuristic planner (the no-LLM degraded path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,check",
    [
        ("Public software companies with more than 1,000 employees.",
         lambda s: s.employees.minimum == 1001 and s.is_public is True),
        ("Clean energy startups founded after 2018 with fewer than 200 employees",
         lambda s: s.employees.maximum == 199 and s.founded_year.minimum == 2019),
        ("Construction companies in the United States with revenue over $50 million",
         lambda s: s.countries == ["us"] and s.revenue_usd.minimum == 50_000_000),
        ("Logistic companies in Romania", lambda s: s.countries == ["ro"]),
        ("Renewable energy equipment manufacturers in Scandinavia",
         lambda s: set(s.countries) == {"se", "no", "dk"}),
    ],
)
def test_heuristic_planner_recovers_constraints_without_an_llm(query, check):
    """The degraded path must still honour what a regex can see."""
    assert check(heuristic_spec(query))


def test_heuristic_planner_invents_nothing_for_a_vague_query():
    spec = heuristic_spec("Fast-growing fintech companies competing with traditional banks")
    assert not spec.employees.is_set()
    assert not spec.revenue_usd.is_set()
    assert spec.is_public is None
    assert spec.countries == []


def test_heuristic_planner_always_yields_a_usable_probe():
    spec = heuristic_spec("anything at all")
    assert spec.ideal_profile and spec.role_statement
    assert spec.complexity == "SEMANTIC"
    assert spec.criteria == []  # no rubric is honest; a fake one would not be


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def _cand(idx: int, fused: float = 0.5, unknowns: list[str] | None = None) -> Candidate:
    return Candidate(
        company=Company(idx=idx, operational_name="C{}".format(idx)),
        fused_score=fused, unknown_constraints=unknowns or [],
    )


def test_qualified_always_outranks_partial_regardless_of_confidence():
    cfg = PipelineConfig()
    cands = [_cand(0), _cand(1)]
    quals = {
        0: Qualification(verdict=Verdict.QUALIFIED, confidence=0.05),
        1: Qualification(verdict=Verdict.PARTIAL, confidence=0.99),
    }
    out = rank(cands, quals, QuerySpec(query="q"), cfg)
    assert out[0].company.idx == 0


def test_rejected_companies_are_dropped_by_default():
    cfg = PipelineConfig()
    cfg.no_match_fallback = 0  # isolate the filter from the recovery path
    quals = {0: Qualification(verdict=Verdict.REJECTED, confidence=0.9)}
    assert rank([_cand(0)], quals, QuerySpec(query="q"), cfg) == []
    assert len(rank([_cand(0)], quals, QuerySpec(query="q"), cfg, include_rejected=True)) == 1


def test_zero_results_fall_back_to_labelled_near_misses():
    """An empty page is worse than an honest "closest we found"."""
    cfg = PipelineConfig()
    cfg.no_match_fallback = 3
    cands = [_cand(i, fused=i / 10) for i in range(6)]
    quals = {i: Qualification(verdict=Verdict.REJECTED, confidence=0.95) for i in range(6)}
    out = rank(cands, quals, QuerySpec(query="q"), cfg)
    assert len(out) == 3
    assert all(r.qualification.verdict is Verdict.PARTIAL for r in out)
    assert all("closest available match" in r.qualification.reason for r in out)
    assert all("fallback" in r.qualification.tier for r in out)


def test_fallback_does_not_fire_when_something_qualifies():
    cfg = PipelineConfig()
    quals = {
        0: Qualification(verdict=Verdict.QUALIFIED, confidence=0.9),
        1: Qualification(verdict=Verdict.REJECTED, confidence=0.9),
    }
    out = rank([_cand(0), _cand(1)], quals, QuerySpec(query="q"), cfg)
    assert [r.company.idx for r in out] == [0]


def test_scores_do_not_all_saturate_for_confident_verdicts():
    """Regression: 50 companies once tied at exactly 1.000, destroying the rank."""
    cfg = PipelineConfig()
    cands = [_cand(i, fused=i / 20) for i in range(20)]
    quals = {i: Qualification(verdict=Verdict.QUALIFIED, confidence=1.0) for i in range(20)}
    scores = [r.score for r in rank(cands, quals, QuerySpec(query="q"), cfg)]
    assert max(scores) <= 1.0
    assert len(set(round(s, 4) for s in scores)) > 10, "retrieval must break ties"


def test_verdict_bands_cannot_overlap_under_any_modifier():
    """The ordering invariant must hold for every reachable combination."""
    cfg = PipelineConfig()
    best_partial = _cand(0, fused=1.0)
    worst_qualified = _cand(1, fused=0.0, unknowns=["employee_count", "revenue", "year_founded"])
    quals = {
        0: Qualification(verdict=Verdict.PARTIAL, confidence=1.0, escalated=True,
                         criteria_met={"a": True}),
        1: Qualification(verdict=Verdict.QUALIFIED, confidence=0.0,
                         criteria_met={"a": False}),
    }
    spec = QuerySpec(query="q", criteria=[Criterion(name="a", description="d", weight=1.0)])
    out = rank([best_partial, worst_qualified], quals, spec, cfg, include_rejected=True)
    assert out[0].company.idx == 1, "a maxed-out PARTIAL must not outrank any QUALIFIED"


def test_unknown_constraints_cost_score():
    cfg = PipelineConfig()
    q = {
        0: Qualification(verdict=Verdict.QUALIFIED, confidence=0.9),
        1: Qualification(verdict=Verdict.QUALIFIED, confidence=0.9),
    }
    out = rank([_cand(0), _cand(1, unknowns=["employee_count", "revenue"])],
               q, QuerySpec(query="q"), cfg, include_rejected=True)
    assert out[0].company.idx == 0


def test_candidate_without_a_verdict_is_surfaced_not_dropped():
    cfg = PipelineConfig()
    out = rank([_cand(0)], {}, QuerySpec(query="q"), cfg, include_rejected=True)
    assert out[0].qualification.tier == "missing"
    assert out[0].qualification.reason == "no verdict returned"


def test_ranking_is_stable_for_tied_scores():
    cfg = PipelineConfig()
    quals = {i: Qualification(verdict=Verdict.QUALIFIED, confidence=0.5) for i in range(5)}
    cands = [_cand(i, fused=0.5) for i in range(5)]
    first = [r.company.idx for r in rank(cands, quals, QuerySpec(query="q"), cfg)]
    second = [r.company.idx for r in rank(list(reversed(cands)), quals, QuerySpec(query="q"), cfg)]
    assert first == second


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------
def test_rate_limiter_allows_a_burst_up_to_capacity_then_throttles():
    """The embedding endpoint bills per text, so the budget is spent in bulk."""
    import asyncio
    import time as _time

    async def scenario() -> tuple[float, float]:
        limiter = RateLimiter(per_minute=60)  # 1 token/second
        start = _time.monotonic()
        await limiter.acquire(60)             # drains the full bucket instantly
        burst = _time.monotonic() - start
        await limiter.acquire(2)              # must now wait ~2s for a refill
        return burst, _time.monotonic() - start

    burst, total = asyncio.run(scenario())
    assert burst < 0.2, "a full bucket should not block"
    assert total >= 1.5, "an empty bucket must throttle rather than overrun"


def test_rate_limiter_never_deadlocks_on_an_oversized_request():
    """A cost larger than the bucket is clamped, not waited on forever."""
    import asyncio

    async def scenario() -> None:
        limiter = RateLimiter(per_minute=10)
        await asyncio.wait_for(limiter.acquire(10_000), timeout=5.0)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_precision_does_not_punish_an_honestly_short_list():
    assert metrics.precision_at_k([3, 3, 2, 2], 10) == 1.0


def test_precision_counts_grade_one_as_irrelevant():
    assert metrics.precision_at_k([3, 1, 1, 1], 4) == 0.25


def test_ndcg_is_one_for_a_perfect_ordering():
    pool = [3, 3, 2, 0, 0]
    assert metrics.ndcg_at_k([3, 3, 2, 0, 0], pool, 5) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ordering():
    pool = [3, 2, 0]
    assert metrics.ndcg_at_k([0, 2, 3], pool, 3) < metrics.ndcg_at_k([3, 2, 0], pool, 3)


def test_kappa_is_one_for_identical_raters_and_near_zero_for_chance():
    assert metrics.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert metrics.cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(1.0)
    assert metrics.cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0.0


def test_binary_scores_handles_an_empty_prediction():
    out = metrics.binary_scores([1, 1, 0], [0, 0, 0])
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["fn"] == 2


def test_summarise_reports_pool_relative_recall():
    out = metrics.summarise([3, 2, 0], [3, 2, 2, 0, 0], ks=(3,))
    assert out["relevant_in_pool"] == 3.0
    assert out["relevant_found"] == 2.0
    assert out["R@3"] == pytest.approx(2 / 3, abs=1e-4)
