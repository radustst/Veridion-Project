"""Load and normalise the company dataset.

The raw file is messier than it looks: `address` and `primary_naics` arrive
sometimes as real JSON objects and sometimes as Python-repr strings
("{'country_code': 'ro', ...}"), numbers arrive as floats where they are
conceptually integers, and six of the thirteen fields are frequently absent.
Everything downstream assumes a clean `Company`, so all of that is dealt with
exactly once, here.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from .schema import Company, Naics


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a field that may be a dict, a Python-repr string, or JSON."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(value)
                if isinstance(out, dict):
                    return out
            except (ValueError, SyntaxError, TypeError):
                continue
    return {}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def _as_naics(value: Any) -> Optional[Naics]:
    m = _as_mapping(value)
    code = str(m.get("code", "")).strip()
    if not code:
        return None
    return Naics(code=code, label=str(m.get("label", "")).strip(), share=_as_float(m.get("share")))


def _as_naics_list(value: Any) -> list[Naics]:
    if value is None:
        return []
    items = value
    if isinstance(value, str):
        try:
            items = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, (list, tuple)):
        return []
    out = [_as_naics(i) for i in items]
    return [n for n in out if n is not None]


def parse_company(row: dict[str, Any], idx: int) -> Company:
    from .geo import normalise_code

    addr = _as_mapping(row.get("address"))
    pub = row.get("is_public")
    return Company(
        idx=idx,
        operational_name=(row.get("operational_name") or None),
        website=(row.get("website") or None),
        year_founded=_as_int(row.get("year_founded")),
        country_code=normalise_code(addr.get("country_code")),
        region_name=(addr.get("region_name") or None),
        town=(addr.get("town") or None),
        employee_count=_as_int(row.get("employee_count")),
        revenue=_as_float(row.get("revenue")),
        primary_naics=_as_naics(row.get("primary_naics")),
        secondary_naics=_as_naics_list(row.get("secondary_naics")),
        description=(row.get("description") or "").strip(),
        business_model=_as_list(row.get("business_model")),
        target_markets=_as_list(row.get("target_markets")),
        core_offerings=_as_list(row.get("core_offerings")),
        is_public=bool(pub) if isinstance(pub, bool) else None,
        raw=row,
    )


def _dedupe_key(c: Company) -> tuple[str, str] | None:
    """Identity key for duplicate detection, or None if we cannot tell."""
    if c.website:
        host = c.website.strip().lower().removeprefix("www.")
        if host:
            return ("web", host)
    if c.operational_name:
        name = " ".join(c.operational_name.split()).lower()
        if name:
            return ("name", "{}|{}".format(name, c.country_code or "?"))
    return None


def dedupe_companies(companies: list[Company]) -> tuple[list[Company], int]:
    """Collapse duplicate records, keeping the most complete version.

    This dataset ships genuine duplicates -- 13 websites appear on two rows
    each, and some pairs are byte-identical. Left alone they cost real money
    (the same company is sent to the LLM twice) and they corrupt the output,
    because a user asking for renewable energy manufacturers in Scandinavia
    should not be shown "CIRKEL Energi" at ranks 7 and 8.

    Duplicates are matched on website first, since that is the closest thing
    to a real identifier here, and on name+country only when no website
    exists. Fields absent from the surviving record are back-filled from the
    one being dropped, so merging never loses information.
    """
    best: dict[tuple[str, str], Company] = {}
    order: list[Company] = []
    dropped = 0

    for c in companies:
        key = _dedupe_key(c)
        if key is None:
            order.append(c)
            continue
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = c
            order.append(c)
            continue

        dropped += 1
        keep, lose = (
            (incumbent, c)
            if incumbent.completeness() >= c.completeness()
            else (c, incumbent)
        )
        for field_name in (
            "operational_name", "website", "year_founded", "country_code",
            "region_name", "town", "employee_count", "revenue", "primary_naics",
            "description", "is_public",
        ):
            if getattr(keep, field_name) in (None, "") and getattr(lose, field_name) not in (None, ""):
                setattr(keep, field_name, getattr(lose, field_name))
        for list_field in ("secondary_naics", "business_model", "target_markets", "core_offerings"):
            if not getattr(keep, list_field):
                setattr(keep, list_field, getattr(lose, list_field))

        if keep is not incumbent:
            order[order.index(incumbent)] = keep
            best[key] = keep

    return order, dropped


def load_companies(path: str | Path, deduplicate: bool = True) -> list[Company]:
    companies: list[Company] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            companies.append(parse_company(json.loads(line), idx))
    if deduplicate:
        companies, _ = dedupe_companies(companies)
        # Re-index so Company.idx stays a dense row index into the embedding
        # matrix; retrieval indexes that matrix by idx directly.
        for new_idx, c in enumerate(companies):
            c.idx = new_idx
    return companies


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------
def company_document(c: Company) -> str:
    """The text used for embedding and BM25.

    Field labels are kept in the string on purpose. "Serves: automotive" and
    "Provides: freight forwarding" embed differently from a bare bag of words,
    and that distinction is exactly what separates a packaging supplier from a
    cosmetics brand -- both mention cosmetics, but only one *provides*
    packaging. Structure in, structure out.
    """
    parts: list[str] = []
    if c.operational_name:
        parts.append(c.operational_name)
    if c.primary_naics:
        parts.append("Industry: {} ({})".format(c.primary_naics.label, c.primary_naics.code))
    for n in c.secondary_naics:
        parts.append("Also: {} ({})".format(n.label, n.code))
    if c.core_offerings:
        parts.append("Provides: " + "; ".join(c.core_offerings[:12]))
    if c.target_markets:
        parts.append("Serves: " + ", ".join(c.target_markets[:10]))
    if c.business_model:
        parts.append("Model: " + ", ".join(c.business_model[:8]))
    if c.description:
        parts.append(c.description)
    parts.append("Location: " + c.location())
    return "\n".join(parts)


def company_card(c: Company, include_description: bool = True, desc_chars: int = 460) -> str:
    """Compact record shown to the qualifying LLM.

    Token budget is the binding constraint on the cheap tier -- eight of these
    share one prompt -- so numbers are rendered densely and the description is
    truncated. Absent fields are printed as "unknown" rather than omitted,
    because the model needs to distinguish "this company has 40 employees" from
    "we do not know how many employees this company has" in order to reason
    honestly about a headcount constraint.
    """
    lines = ["Name: {}".format(c.name)]
    lines.append("Location: {}".format(c.location()))
    if c.primary_naics:
        lines.append("NAICS: {} {}".format(c.primary_naics.code, c.primary_naics.label))
    if c.secondary_naics:
        lines.append(
            "Secondary NAICS: "
            + "; ".join("{} {}".format(n.code, n.label) for n in c.secondary_naics)
        )
    lines.append(
        "Employees: {} | Revenue USD: {} | Founded: {} | Public: {}".format(
            c.employee_count if c.employee_count is not None else "unknown",
            "{:,.0f}".format(c.revenue) if c.revenue is not None else "unknown",
            c.year_founded if c.year_founded is not None else "unknown",
            c.is_public if c.is_public is not None else "unknown",
        )
    )
    if c.business_model:
        lines.append("Business model: " + ", ".join(c.business_model[:6]))
    if c.core_offerings:
        lines.append("Core offerings: " + "; ".join(c.core_offerings[:8]))
    if c.target_markets:
        lines.append("Target markets: " + ", ".join(c.target_markets[:8]))
    if include_description and c.description:
        desc = c.description
        if len(desc) > desc_chars:
            desc = desc[:desc_chars].rsplit(" ", 1)[0] + "..."
        lines.append("Description: " + desc)
    return "\n".join(lines)


def dataset_summary(companies: Iterable[Company]) -> dict[str, Any]:
    companies = list(companies)
    n = len(companies)
    if not n:
        return {}
    countries: dict[str, int] = {}
    for c in companies:
        if c.country_code:
            countries[c.country_code] = countries.get(c.country_code, 0) + 1
    return {
        "n": n,
        "countries": dict(sorted(countries.items(), key=lambda kv: -kv[1])),
        "has_employee_count": sum(c.employee_count is not None for c in companies) / n,
        "has_revenue": sum(c.revenue is not None for c in companies) / n,
        "has_year_founded": sum(c.year_founded is not None for c in companies) / n,
        "has_secondary_naics": sum(bool(c.secondary_naics) for c in companies) / n,
        "mean_completeness": sum(c.completeness() for c in companies) / n,
    }
