"""The 12 benchmark queries, tagged by the kind of difficulty they pose.

The tags are not decoration. Averaging one number over all twelve hides the
only interesting result: a system can be near-perfect on structured filters
and still fail badly at inferring a company's role in a supply chain, and it
is that gap -- not the mean -- that tells you what to build next.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    text: str
    kind: str   # structured | semantic | reasoning
    note: str   # what specifically makes this one hard


QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        "q01", "Logistic companies in Romania", "structured",
        "Geography plus a clean industry. The trap is software vendors selling "
        "logistics tools, and in-house distribution arms of non-logistics firms.",
    ),
    BenchmarkQuery(
        "q02", "Public software companies with more than 1,000 employees.", "structured",
        "Three field-level constraints. is_public is present for every row but "
        "employee_count is missing for 38%, so the headcount gate is where "
        "recall is won or lost.",
    ),
    BenchmarkQuery(
        "q03", "Food and beverage manufacturers in France", "structured",
        "NAICS 311/312 plus country. The trap is retailers and distributors of "
        "food that do not manufacture it.",
    ),
    BenchmarkQuery(
        "q04",
        "Companies that could supply packaging materials for a direct-to-consumer "
        "cosmetics brand",
        "reasoning",
        "The canonical embedding failure: naive similarity returns cosmetics "
        "brands. Correct answers are packaging manufacturers, which may never "
        "mention cosmetics at all.",
    ),
    BenchmarkQuery(
        "q05",
        "Construction companies in the United States with revenue over $50 million",
        "structured",
        "Clean on paper. Tests unit handling ($50 million -> 50000000) and "
        "whether engineering/architecture firms are wrongly counted as construction.",
    ),
    BenchmarkQuery(
        "q06", "Pharmaceutical companies in Switzerland", "structured",
        "Easy geography, but Switzerland is 43 rows here and heavy in life "
        "sciences, so precision depends on separating pharma from medtech, "
        "diagnostics and CROs.",
    ),
    BenchmarkQuery(
        "q07", "B2B SaaS companies providing HR solutions in Europe", "semantic",
        "Four simultaneous conditions: B2B, SaaS delivery, HR domain, Europe. "
        "Partial matches (B2B SaaS that is not HR) are the main false positive.",
    ),
    BenchmarkQuery(
        "q08",
        "Clean energy startups founded after 2018 with fewer than 200 employees",
        "semantic",
        "Two numeric gates over fields missing in 27% and 38% of rows -- the "
        "hardest test of the UNKNOWN path. 'Startup' is also vague and must "
        "not become an invented extra filter.",
    ),
    BenchmarkQuery(
        "q09",
        "Fast-growing fintech companies competing with traditional banks in Europe.",
        "reasoning",
        "'Fast-growing' is unmeasurable from this schema and 'competing with "
        "traditional banks' is a judgement about market position. Tests whether "
        "the system admits what it cannot know instead of inventing growth data.",
    ),
    BenchmarkQuery(
        "q10", "E-commerce companies using Shopify or similar platforms", "reasoning",
        "Asks about a technographic fact the dataset does not contain. The "
        "correct behaviour is graceful degradation to 'SMB-scale D2C e-commerce', "
        "flagged as inferred -- not confident fabrication.",
    ),
    BenchmarkQuery(
        "q11", "Renewable energy equipment manufacturers in Scandinavia", "semantic",
        "Region word needing a stable definition (se/no/dk, not fi/is), and a "
        "manufacturer-vs-operator distinction: a wind farm operator is not an "
        "equipment manufacturer.",
    ),
    BenchmarkQuery(
        "q12",
        "Companies that manufacture or supply critical components for electric "
        "vehicle battery production",
        "reasoning",
        "Multi-tier supply chain. Correct answers span cathode chemicals, "
        "lithium refining, separators, foils and cell equipment -- most of which "
        "never use the phrase 'electric vehicle'.",
    ),
]

BY_ID = {q.id: q for q in QUERIES}
TEXTS = [q.text for q in QUERIES]
