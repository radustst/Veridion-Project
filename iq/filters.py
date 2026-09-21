"""Stage 1 -- deterministic hard constraints with three-valued logic.

Free, instant, and identical on every run. Anything checkable from a field is
checked here so that the LLM never spends a token deciding whether 1979 is
after 2018.

The design decision that matters is what happens to missing data. In this
dataset 38% of companies have no employee_count and 27% have no year_founded.

  - Treat missing as FAIL and "clean energy startups founded after 2018 with
    fewer than 200 employees" deletes most of the correct answers, because the
    small young companies are exactly the ones with thin data.
  - Treat missing as PASS and every constraint becomes decorative.

So missing is UNKNOWN: the company survives, the gap is recorded, the LLM is
asked to infer it from the description, and the final score carries a small
penalty for each requirement that could never be verified. That keeps recall
without pretending to a certainty we do not have.
"""
from __future__ import annotations

from .schema import Company, FilterOutcome, QuerySpec, Tri


def apply_filters(company: Company, spec: QuerySpec) -> FilterOutcome:
    checks: dict[str, Tri] = {}

    if spec.countries:
        if company.country_code is None:
            checks["country"] = Tri.UNKNOWN
        else:
            checks["country"] = (
                Tri.PASS if company.country_code in set(spec.countries) else Tri.FAIL
            )

    if spec.employees.is_set():
        checks["employee_count"] = spec.employees.check(company.employee_count)

    if spec.revenue_usd.is_set():
        checks["revenue"] = spec.revenue_usd.check(company.revenue)

    if spec.founded_year.is_set():
        checks["year_founded"] = spec.founded_year.check(company.year_founded)

    if spec.is_public is not None:
        if company.is_public is None:
            checks["is_public"] = Tri.UNKNOWN
        else:
            checks["is_public"] = Tri.PASS if company.is_public == spec.is_public else Tri.FAIL

    return FilterOutcome(company=company, per_constraint=checks)


def filter_companies(
    companies: list[Company], spec: QuerySpec
) -> tuple[list[FilterOutcome], dict[str, int]]:
    """Split the database into survivors and a per-constraint rejection tally.

    The tally is returned rather than discarded because it is the first thing
    to look at when a query returns too few results: if 460 of 477 companies
    died on `country`, the geography was resolved wrongly, and no amount of
    prompt tuning downstream will fix that.
    """
    survivors: list[FilterOutcome] = []
    rejected_by: dict[str, int] = {}
    for c in companies:
        outcome = apply_filters(c, spec)
        if outcome.failed:
            for key, val in outcome.per_constraint.items():
                if val is Tri.FAIL:
                    rejected_by[key] = rejected_by.get(key, 0) + 1
            continue
        survivors.append(outcome)
    return survivors, rejected_by


# Geography is the one constraint we trust absolutely: country_code is present
# for every company in this dataset and is not a judgement call. Everything
# else can be softened, but a company in Poland is not in Germany.
HARD_FAIL_ALWAYS = frozenset({"country"})


def relax(spec: QuerySpec, survivors: int, minimum: int = 12) -> bool:
    """Whether the filter pass was so aggressive it is worth re-running relaxed.

    Called by the pipeline when almost nothing survives. Returns True if the
    spec has soft-able numeric constraints that could be demoted to ranking
    signals rather than gates -- an over-eager planner reading "startups" as a
    headcount cap should not be allowed to empty the result set.
    """
    if survivors >= minimum:
        return False
    return spec.employees.is_set() or spec.revenue_usd.is_set() or spec.founded_year.is_set()


def soften(spec: QuerySpec) -> QuerySpec:
    """Return a copy of the spec with numeric gates removed.

    Geography and is_public stay: those are cheap, reliable and rarely the
    cause of an empty result set.
    """
    import copy

    relaxed = copy.deepcopy(spec)
    relaxed.employees.minimum = relaxed.employees.maximum = None
    relaxed.revenue_usd.minimum = relaxed.revenue_usd.maximum = None
    relaxed.founded_year.minimum = relaxed.founded_year.maximum = None
    return relaxed
