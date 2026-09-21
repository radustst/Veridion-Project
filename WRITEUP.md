# Intent Qualification — Writeup

*Numbers come from `results/eval_report.md`, `results/results.json`,
`results/calibration.json` and `results/cold_run.json`, all produced by the
commands in the README. Where a measurement was planned but not completed,
this document says so rather than estimating.*

---

## 3.1 Approach

### The problem underneath the problem

The brief frames this as a cost problem — LLM-per-company is accurate but
expensive. I think that framing is only half right, and the more interesting
half is accuracy.

Take the query the brief itself flags:

> *Companies that could supply packaging materials for a D2C cosmetics brand*

Embedding search fails here not because embeddings are weak, but because the
query and the answer are about **different companies**. The query is written
from the buyer's perspective; the answer is a seller. The words "cosmetics"
and "brand" describe the customer, and a vector model — correctly, on its own
terms — returns documents that look like the query. Spending more money on a
bigger model at rerank time does not fix a retrieval stage that never
surfaced a single packaging manufacturer.

So the central design decision is not *where do I save tokens*. It is:
**translate the query into a description of the answer before doing anything
else.** Everything cheap downstream works better once you do, and the cost
savings mostly fall out for free.

### Architecture

```
                                                   companies that
   user query                                      satisfy the intent
       │                                                    ▲
       ▼                                                    │
┌──────────────┐                                     ┌──────────────┐
│ ① PLANNER    │  1 LLM call, fixed per query        │ ⑥ RANKING    │
│              │                                     │              │
│ hard filters │                                     │ verdict band │
│ rubric       │                                     │ + confidence │
│ NAICS priors │                                     │ + rubric     │
│ ideal profile│                                     │ + retrieval  │
└──────┬───────┘                                     └──────▲───────┘
       │ QuerySpec                                          │
       ▼                                                    │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│ ② FILTERS    │────▶│ ③ RETRIEVAL  │────▶│ ④ QUALIFY    │──┤
│              │     │              │     │              │  │
│ PASS/FAIL/   │     │ dense + BM25 │     │ cheap model  │  │
│ UNKNOWN      │     │ + NAICS, RRF │     │ 8 per call   │  │
│              │     │              │     │              │  │
│ free         │     │ free         │     │ 4-10 calls   │  │
│ 456 -> 26-456│     │ -> 30-80     │     │              │  │
└──────────────┘     └──────────────┘     └──────┬───────┘  │
                                                 │          │
                                          ┌──────▼───────┐  │
                                          │ ⑤ ESCALATE   │──┘
                                          │ stronger LLM │
                                          │ only if      │
                                          │ uncertain    │
                                          │ ≤25% capped  │
                                          └──────────────┘
```

Only stages ② and ③ touch every company, and neither calls a model. **Cost per
query is therefore roughly flat in database size** — which is the property
that actually matters for scaling, more than any constant-factor saving.

### ① Planner — one call, the most leverage in the system

One LLM call converts the query into a `QuerySpec`:

| Output | Purpose |
|---|---|
| `complexity` | STRUCTURED / SEMANTIC / REASONING → sets the downstream budget |
| hard filters | country set, employee/revenue/founding ranges, is_public |
| `role_statement` | what a matching company must **do**, from the seller's side |
| `ideal_profile` | a synthetic company record — this is what gets embedded |
| `naics_prefixes` | industry codes a match would plausibly carry |
| `keywords` | vocabulary for lexical matching |
| `criteria` | 2–5 weighted, individually checkable requirements |
| `disqualifiers` | the specific near-misses a careless matcher would accept |

Two of these deserve justification.

**`ideal_profile` (the HyDE move).** Instead of embedding the query, the
planner writes the record of a company that would perfectly answer it, and we
embed *that*. For the packaging query it produced:

> *"Manufacturer and distributor of commercial packaging solutions,
> specializing in primary and secondary containers for the beauty, personal
> care, and cosmetics industries. Core offerings include custom glass and
> plastic bottles, jars, squeeze tubes, droppers, pumps, caps, folding
> cartons…"*

That text lives in the same region of embedding space as actual packaging
manufacturers, where the raw query does not.

**But I should report what actually happened, not what I expected.** I built
this specifically to beat the failure the brief describes, and on *this*
dataset that failure barely occurs. Baseline B — raw-query cosine, no planning
at all — returns packaging manufacturers in its top 12 for the packaging query
and scores P@10 of 0.900 against the cascade's 1.000. Not one cosmetics brand
appears.

The reason is a property of the data rather than of the method: the packaging
suppliers here describe themselves as *"cosmetic packaging manufacturers"*, so
the query's own words appear in the correct answers. The buyer/seller
confusion the brief warns about needs a corpus where the supplier never names
the customer's industry, and this corpus mostly is not that.

The HyDE profile still earns its place — it improves the *ordering*
(nDCG@10 0.781 vs 0.658) and it is what makes the EV-battery supply-chain
query work, where the answers genuinely do not say "electric vehicle". But the
honest headline is that on this data the largest wins come from the structured
constraints, not the semantic rewrite. §Results has the numbers.

**`disqualifiers`.** The planner names the traps explicitly, e.g. *"A
direct-to-consumer cosmetics brand that uses packaging is not a packaging
supplier."* These go verbatim into the qualifier prompt. Telling a cheap model
precisely which mistake to avoid is far more effective per token than
instructing it to "be careful".

### ② Filters — three-valued, because missing data is the norm

`employee_count` is absent for 38% of companies, `year_founded` for 27%,
`revenue` for 18%. The obvious binary filter is a trap in both directions:

- **missing → FAIL** deletes the right answers. For *"clean energy startups
  founded after 2018 with fewer than 200 employees"*, the small young
  companies are precisely the ones with thin records.
- **missing → PASS** makes the constraint decorative.

So constraints evaluate to `PASS`, `FAIL`, or **`UNKNOWN`**. UNKNOWN survives,
is recorded, is shown to the LLM as an explicit `[unverifiable from data: …]`
annotation, and costs the company a small amount of final score. The user sees
which requirements could not be checked rather than being told a confident lie.

Geography resolves from a **static table**, not the LLM. Ask a model twice
whether Finland is in Scandinavia and you can get two answers; `geo.py` always
says se/no/dk. It is free, instant, auditable, and removes an entire class of
run-to-run instability.

### ③ Retrieval — three weak signals, fused by rank

| Signal | Catches | Blind to |
|---|---|---|
| Dense (vs `ideal_profile`) | paraphrase, role similarity | exact codes, rare proper nouns |
| BM25 over a structured document | "Shopify", "airless pump", "customs brokerage" | synonymy |
| NAICS prefix affinity | taxonomy truth, independent of prose | mis-coded companies |

Fused with **Reciprocal Rank Fusion**. RRF consumes ranks, not scores, which
matters: a cosine of 0.71 and a BM25 of 14.3 are not comparable, and
normalising them into a weighted sum needs per-query calibration I have no
labels to fit. RRF needs none and degrades gracefully when one signal is
uninformative.

The document embedded is *structured*, not a bag of words — `"Provides: …"`,
`"Serves: …"`, `"Industry: …"`. A cosmetics brand and a packaging supplier both
contain the token "cosmetics"; only one **provides** bottles. Keeping the field
labels preserves that distinction.

Shortlist depth is set by the planner's complexity label — 30 for STRUCTURED,
55 for SEMANTIC, 80 for REASONING. This is the direct answer to the brief's
"simple queries receive the same expensive treatment" complaint.

### ④⑤ Qualification and selective escalation

Eight companies share one prompt. The rubric and disqualifiers are the bulk of
the tokens and are identical across companies, so batching removes roughly 80%
of input tokens versus one-call-per-company. Temperature 0, a response schema,
and a content-addressed cache handle the consistency complaint.

Escalation to a stronger model fires only on evidence that the cheap verdict is
*unreliable*, not merely negative:

1. confidence inside the uncertainty band (0.35–0.72);
2. verdict is PARTIAL — the model saying so itself;
3. **retrieval and the LLM disagree** — ranked top-decile but rejected, or
   ranked low but qualified. Disagreement between independent signals is the
   classic cheap uncertainty proxy;
4. accepted despite a constraint that could never be verified.

Capped at 25% of the shortlist. The cap is the point: without it, a badly
planned query where the cheap tier is unsure about everything would quietly
cost as much as Baseline A — the exact failure the architecture exists to
prevent. The cap being hit is itself a signal worth logging.

### ⑥ Ranking

Verdict picks a **non-overlapping band** (QUALIFIED 0.60–1.00, PARTIAL
0.25–0.55, REJECTED 0.00–0.20). Confidence, rubric coverage, retrieval strength
and the missing-data penalty position a company *within* its band and can never
move it across one. So the ordering is explainable — "it ranks lower because
two of its criteria are unproven" — which for this use case is worth more than
a marginal nDCG gain from a cleverer blend.

---

### What the funnel actually looks like

From `results/results.json`, one row per benchmark query:

| Query | Detected | Passed filters | Shortlist | Escalated | Returned | Qualified |
|---|---|---:|---:|---:|---:|---:|
| Logistics in Romania | STRUCTURED | 26 | 26 | 3 | 7 | 6 |
| Public software >1000 emp | STRUCTURED | 36 | 30 | 1 | 20 | 20 |
| F&B manufacturers France | STRUCTURED | 40 | 30 | 1 | 20 | 20 |
| Packaging for cosmetics | REASONING | 456 | 80 | 2 | 21 | 21 |
| US construction >$50M | STRUCTURED | 55 | 30 | 1 | 20 | 20 |
| Pharma in Switzerland | STRUCTURED | 43 | 30 | 7 | 27 | 27 |
| B2B SaaS HR Europe | SEMANTIC | 285 | 55 | 13 | 46 | 46 |
| Clean energy startups | SEMANTIC | 222 | 55 | 13 | 50 | 19 |
| Fintech vs banks Europe | REASONING | 285 | 80 | 5 | 7 | 4 |
| E-commerce on Shopify | SEMANTIC | 456 | 55 | 13 | 31 | 4 |
| Renewables Scandinavia | SEMANTIC | 63 | 55 | 13 | 53 | 45 |
| EV battery components | REASONING | 456 | 80 | 8 | 51 | 47 |

Three things to read off this table:

- **The hard filters do the heavy lifting where they can.** Geography alone
  takes 456 → 26 for Romania and 456 → 63 for Scandinavia, for free. Where the
  query has no structural constraint (packaging, Shopify, EV batteries) all
  456 survive and retrieval carries the whole load — which is exactly where
  the HyDE profile earns its place.
- **Complexity routing is working.** STRUCTURED queries shortlist 30 and
  escalate 1–7; REASONING queries shortlist 80. Simple queries genuinely cost
  less.
- **Escalation stays narrow.** Typically 1–13 companies, comfortably inside the
  25% cap, so the expensive tier is a small addition rather than a second full
  pass.

The two queries where `qualified` collapses far below `returned` — Shopify
(4 of 31) and clean energy startups (19 of 50) — are the system declining to
vouch for companies it cannot verify. That is the intended behaviour, and
§3.3 shows one case where it worked and one where it did not.

---

## Results

Method: every system's top 20 is pooled per query and graded 0–3 by an LLM
judge that never sees which system produced what, nor the planner's reading of
the query. Relevant means grade ≥ 2. Full output in
`results/eval_report.md`; the judge's own reliability is examined below,
and it matters.

**10 of 12 queries were judged.** Queries 11 and 12 hit an exhausted API quota
during judging and are excluded from every number here rather than being
scored as zeros.

### Cascade vs embedding-only, 10 judged queries

| System | P@5 | P@10 | P@20 | nDCG@10 | nDCG@20 | MAP |
|---|---:|---:|---:|---:|---:|---:|
| **Cascade** | **0.840** | **0.816** | **0.796** | **0.874** | **0.903** | **0.834** |
| Baseline B (embedding only) | 0.660 | 0.660 | 0.605 | 0.748 | 0.800 | 0.711 |

A consistent and meaningful margin — +0.156 P@10, +0.126 nDCG@10 — but the
aggregate hides where it comes from. Per query, the cascade and Baseline B are
*identical* on the easy industry+geography queries (France F&B, US
construction, Swiss pharma, EU HR SaaS: both 1.000 P@10). The whole margin is
earned on two queries:

| Query | Cascade P@10 | Baseline B P@10 |
|---|---:|---:|
| Clean energy startups, founded >2018, <200 employees | **0.900** | 0.200 |
| Logistics in Romania | **0.857** | 0.600 |
| Fintech competing with banks in Europe | **1.000** | 0.700 |
| Packaging for a D2C cosmetics brand | **1.000** | 0.900 |
| E-commerce using Shopify | 0.400 | 0.200 |

The numeric-constraint query is the decisive one, for the reason given in
§3.3E: embeddings cannot represent "founded after 2018".

### Cascade vs LLM-per-company

Baseline A ran over the **judged pool** rather than all 456 companies. One
call per company at the free tier's 15 requests/minute is **over 30 minutes
for a single query**, which was not a sensible use of the remaining quota. The
pooled variant still answers the architectural question — is adjudicating
companies one at a time better than eight to a prompt? — but says nothing
about recall over the full corpus. Compared like-for-like on the three
queries where it ran:

| System | P@5 | P@10 | P@20 | nDCG@10 | MAP | Avg returned |
|---|---:|---:|---:|---:|---:|---:|
| **Cascade** | **1.000** | **0.952** | **0.902** | 0.865 | 0.976 | 24.7 |
| Baseline A (pooled) | 0.933 | 0.889 | 0.856 | **0.876** | **0.981** | 16.3 |
| Baseline B | 0.800 | 0.833 | 0.733 | 0.799 | 0.854 | 20.0 |

Batching costs nothing measurable in accuracy here: the cascade is ahead on
precision at every depth, and Baseline A's tiny leads on nDCG@10 and MAP come
from returning a shorter, more conservative list.

**One caution about my own table.** Across *all* queries the cascade averages
P@10 0.816 while Baseline A shows 0.889 — which looks like a loss until you
notice they were scored on different query sets. Baseline A only ran on three
comparatively easy queries; the cascade's average is dragged down by the two
hardest. Restricted to the same three, the cascade wins. Comparing means over
different subsets is an easy way to publish a wrong conclusion, and I nearly
did.

### Cost and speed

| | Cascade | Baseline A |
|---|---:|---:|
| LLM calls per query | **6–13, mean 9.3** | 456 |
| LLM calls, all 12 queries | **112** | 5,472 (**49× more**) |
| Cost per query | **~$0.003** | ~$0.030 (projected) |
| Time per query | ~19 s | >30 min at free-tier RPM |

Call counts are exact, derived from the recorded stage counts of the
benchmark run: `1 planner + ceil(shortlist/8) + ceil(escalated/5)`. They scale
with query complexity, not database size — 6 calls for a structured query with
a 26-company shortlist, 13 for the EV-battery supply-chain query with 80.

Cost and time are measured cold with the cache disabled, but on **one query
only** (`results/cold_run.json`: Romania logistics, 6 calls, $0.00321,
18.8 s) — the remaining cold measurements were abandoned when the daily quota
ran out. Treat the per-query cost as an order of magnitude, not a precise
figure. Baseline A's cost is projected from 102 real uncached calls ($0.00679)
scaled to 456 companies.

**The call count is the claim I would defend**: 49× fewer requests, exactly
measured, independent of quota and provider pricing. The wall-clock gap is
real here but inflated by free-tier limits and would narrow substantially on a
paid tier with proper concurrency.

### Is the judge trustworthy? Partly, and the failures are correlated

47 pairs across three queries were hand-adjudicated by reading the full
records, independently of both the pipeline and the judge
(`results/calibration.json`):

| Measure | Value |
|---|---|
| Exact grade agreement | 0.617 |
| Agreement within one grade | 0.979 |
| **Cohen's κ (4-point)** | **0.239** |
| Cohen's κ (binary relevant/not) | 0.230 |
| Judge vs human, binary F1 | 0.943 |

The binary F1 of 0.943 looks reassuring and mostly is not. κ of 0.23 is only
"fair" agreement: nearly everything in a pooled top-20 is relevant, so two
raters agree often by luck, and κ strips that out. Quoting the F1 alone would
overstate the judge considerably.

More importantly, **the disagreements are not random — they run the same way
the pipeline errs.** On the logistics query the judge graded *Brasov
Industrial Portfolio* a **3**; I graded it **1**, because every one of its
core offerings is warehouse *leasing* (§3.3C). The judge rewarded precisely
the false positive the cascade made. It also graded *OSCAR*, a fuel
wholesaler, a 2 against my 1.

So the reported P@10 of 0.816 is **optimistic**: on at least one query the
judge and the system share a blind spot, and a grader that shares your errors
cannot detect them. This is the concrete, measured instance of failure mode 6
in §3.5 — predicted, then confirmed.

Two further judge findings worth recording:

- On the packaging query the judge is systematically *stricter* than me,
  grading seven cosmetic-packaging manufacturers 2 where I gave 3. Directional
  bias, no effect on binary relevance.
- On query 2 the judge graded **all 34 pooled companies 0 or 1**, reasoning
  that Capgemini, TCS, Atos and Concentrix are *"IT consulting, not a software
  publisher"*. That is the same over-narrow reading of "software company" that
  broke my planner (§3.3B). Had I trusted the judge, I would have concluded
  my fix failed. It did not — the judge inherited the identical blind spot.
  This single case is the best argument in the whole project for keeping a
  human-labelled set.

---

## 3.2 Tradeoffs

**Optimised for, in order: intent accuracy on hard queries → cost predictability
→ explainability → latency.** Latency is last deliberately; this is a research
/ prospecting workload where a user waits a few seconds for a considered
answer, not an autocomplete.

| Decision | Bought | Paid |
|---|---|---|
| Planner as a hard dependency | Every downstream stage gets a machine-readable intent | Single point of failure — a bad plan poisons everything (see 3.3) |
| Embed a synthetic profile, not the query | The supply-chain queries work at all | One extra call; the profile can hallucinate a too-narrow archetype |
| Static geography table | Free, instant, perfectly reproducible | Needs manual upkeep; unknown region words silently resolve to nothing |
| Three-valued filters | Recall preserved on sparse rows | More candidates reach the LLM, so higher cost |
| Batching 8 per prompt | ~80% fewer input tokens | Position bias and cross-contamination between companies in a batch |
| Escalation capped at 25% | Bounded worst-case cost | Some genuinely hard cases never get the better model |
| RRF over a tuned weighted sum | No labels needed, robust | Leaves accuracy on the table if labels ever exist |
| Flat numpy matrix, no vector DB | Two dependencies total | Rewrite needed at ~10⁶ companies (see 3.4) |
| No vendor SDK | Retry/fallback/rate-limit logic is visible and portable | More code to own |

**What I deliberately did not build.** A learned reranker (no labels, and 456
companies cannot support training one). A knowledge graph of supply-chain
relationships (the right answer for query 12, and far beyond scope). Query
decomposition into sub-queries (the planner's rubric already covers the
multi-condition cases). An ANN index (see 3.4 — it would be strictly worse
here).

**The honest cost caveat.** This system is cheaper per query than Baseline A
and the gap widens with database size, because Baseline A is linear in
companies and the cascade is roughly constant. But it is *not* cheaper than
Baseline B, and never will be — Baseline B does almost no work. The claim is
that Baseline B does not answer the question.

---

## 3.3 Error analysis

Four failures worth the space, all from `results/results.json`. Two I fixed
and two I did not, because they are properties of the design rather than bugs.

### A. The unanswerable constraint, silently dropped — *the worst one*

> **Query 10:** *"E-commerce companies using Shopify or similar platforms"*

The dataset contains no technographic data. Nothing records what platform a
company runs on. The planner handled this by quietly rewriting the role as
*"operates an online retail or direct-to-consumer store"* — dropping the part
of the query it could not satisfy — and the pipeline then answered that
easier question perfectly.

The four confidently QUALIFIED results:

| Rank | Company | Score | Problem |
|---|---|---|---|
| 1 | H&M Home | 0.924 | Global fashion retailer, enterprise commerce stack |
| 2 | Dell Technologies | 0.920 | Builds its own commerce infrastructure |
| 3 | Decathlon | 0.919 | Multinational, custom platform |
| 4 | Flextribe | 0.898 | Plausible — small D2C |

Three of the top four are the *least* likely Shopify users in the database.
Shopify implies SMB/D2C scale; the system returned the largest retailers it
could find, because "operates an online store" selects for exactly that.

What makes this the most dangerous failure is not that it is wrong — it is
that the output looks **excellent**. Well-known companies, fluent rationales,
high confidence. Nothing anywhere in the pipeline signals that the actual
question went unanswered.

And the system is inconsistent about it. Lululemon, Forever 21 and Home Depot
were all marked PARTIAL with reasons like *"the record does not specify the
e-commerce platform"* — the honest answer. The escalation tier reached that
conclusion for rank 5 onward but left the confident, wrong verdicts at the top
untouched. So the machinery to notice was present; it just did not fire where
it mattered.

**Fix I did not build:** a constraint-coverage check that asserts every
constraint in the query maps to a hard filter or a criterion, and tells the
user "platform usage could not be checked — no such data" instead of silently
answering a different question. This is item 2 on the next-steps list and I
think it is the single highest-value addition to the system.

### B. Planner over-specification — *fixed, and the most instructive*

> **Query 2:** *"Public software companies with more than 1,000 employees."*

Originally returned **zero companies**. The filters were perfect — 36
companies passed `is_public AND employee_count > 1000`, including Capgemini,
Wipro, TCS, Fujitsu, NTT DATA and EPAM. The qualifier then rejected every
single one.

The cause was one phrase in the plan. The planner had written the role as
*"development, licensing, and support of **proprietary software products**"*.
But this dataset's software companies are overwhelmingly NAICS 541512,
*Computer Systems Design Services* — IT services firms. Measured against
"proprietary software products", they are all correctly rejected.

One invented adjective, never present in the user's query, cost the entire
result set. Nothing downstream could catch it, because every stage after the
planner is *supposed* to faithfully enforce the plan. This is the structural
risk of putting a planner at the front of the pipeline, and it is the price
paid for everything the planner buys.

Two changes: the planner prompt now forbids narrowing past the query and names
this exact case; and `ranking.py` gained a zero-result recovery path that
surfaces labelled near misses rather than a blank page. **0 → 20 results**,
led by TCS, Concentrix, CGI, Fujitsu and Capgemini.

### C. Role confusion: adjacent is not the same as in

> **Query 1:** *"Logistic companies in Romania"* — rank 1: **Brasov Industrial
> Portfolio** (score 0.978)

It is an industrial real-estate portfolio that leases warehouse space. The
rationale — *"operates warehousing and logistics facilities"* — is factually
true and still wrong: owning a warehouse is not providing logistics. The
planner's disqualifiers caught the software-vendor trap it anticipated, but not
the property-owner trap it did not.

This is the residual form of exactly the error the whole architecture targets.
Reduced, not eliminated: disqualifiers only defend against the near-misses the
planner thinks to name in advance.

A related judgement call in the same list: *Compania Națională de Căi Ferate*
(national rail infrastructure) and *Poșta Română* (postal service). Both are
defensible as freight/parcel logistics, both are arguably infrastructure or
postal rather than logistics providers. I would not call either clearly wrong,
which is itself the point — a meaningful fraction of this task has no crisp
ground truth, and any single accuracy number papers over that.

### D. Duplicates and score saturation — *both fixed*

On *"Renewable energy equipment manufacturers in Scandinavia"* the first run
returned CIRKEL Energi, DEIF Wind Power Technology, ENERCON and World Wide
Wind **twice each** — the source data has 13 duplicated websites and 25
duplicated names, some rows byte-identical. They were being paid for twice and
displayed twice. Deduplication on website, then name+country, keeping the most
complete row: 477 → 456.

Worse, on the same query **fifty companies tied at exactly 1.000**. The cheap
model returns confidence 1.0 for almost everything it accepts, and the
QUALIFIED band ran to the clamp, so the list was qualified but not *ranked* —
half the deliverable missing. Verdict bands are now strictly non-overlapping
and the modifiers position companies within a band. Scores now spread properly
(0.989, 0.986, 0.980, 0.979…).

### E. Where the system is strongest — and it is not where I expected

The largest measured win is **query 8**, *"Clean energy startups founded after
2018 with fewer than 200 employees"*: cascade P@10 **0.900** against Baseline
B's **0.200**. Looking at what embedding-only returned explains the gap
completely — its top eight are companies the judge graded 1, with reasons like:

- *Tesla Ocean Turbine* — "founded date is unknown, preventing verification"
- *MantaWind* — "founded in 2018, failing the 'after 2018' requirement"
- *PACE* — "a public-private project/advocacy group, not a company"
- *Ventum Dynamics, windainergy, Sol Industries, Logic-Energy* — all "founded
  date unknown"

Every one of these is topically perfect clean-energy content and every one
fails the query. No amount of semantic similarity fixes an off-by-one on a
year, because "founded after 2018" is not a concept that lives in embedding
space. A deterministic filter answers it exactly, for free, every time. The
unglamorous stage does the heavy lifting.

The supply-chain reasoning is genuinely good too — query 12 returns cathode
and anode makers (BTR New Material, Stratus Materials, Hunan Yueneng), a
separator specialist (Sepion) and a cell manufacturer (QuantumScape), a
correct multi-tier read from records that mostly never say "electric vehicle".
But query 12 went unjudged, so I cannot put a number on it, and on the
packaging query Baseline B was nearly as good (§3.1).

**The honest summary: the headline result is that structured constraints and
role-aware retrieval beat similarity, and it is the boring deterministic
filter — not the clever synthetic-profile trick — that produces the biggest
measured gap on this dataset.** I would not have predicted that, and it is the
kind of thing you only find by measuring rather than reasoning about the
architecture.

---

## 3.4 Scaling to 100,000 companies per query

Today's shape at 456 companies: the corpus embedding is a one-off; per query,
filtering and retrieval are free and an LLM sees only 30–80 companies.

**What already scales.** The per-query LLM cost. Filtering is O(n) over
integers. Retrieval is a 456×768 matrix product — microseconds. Going to 100k
changes none of the cost structure, only the constants.

**What breaks, and the fix:**

1. **Corpus embedding cost and time.** 100k companies ≈ 25M tokens ≈ $4 one-off
   — fine. But the free tier's 100 requests/minute means 17 hours. *Fix:* the
   paid tier and `asyncBatchEmbedContent`. This is a quota problem, not a
   design problem. Embeddings are already incremental and content-addressed, so
   only changed records re-embed.

2. **Brute-force search, eventually.** 100k×768 float32 is 300MB and a full
   scan is ~50ms — still fine, and I would **not** add an ANN index at 100k.
   The recall cliff and operational cost are not worth it. Past ~1M I would use
   HNSW (`hnswlib`/FAISS) with the country filter pushed into the index as a
   pre-filter rather than applied after.

3. **BM25 is the real bottleneck.** The current implementation rebuilds the
   index over the filtered subset on *every query* — O(n) tokenisation per
   query. At 100k that is seconds. *Fix:* build one inverted index at load time
   and query it with a posting-list intersection; restrict to the filtered
   doc-id set rather than re-indexing. This is the first thing I would change.

4. **Filter selectivity becomes load-bearing.** At 456, "Europe" leaves 306
   companies and retrieval handles it. At 100k it leaves ~70k, and the
   shortlist is still 80 — so retrieval quality, not the LLM, determines the
   answer. *Fix:* deepen the shortlist for low-selectivity queries, and add a
   cheap cross-encoder rerank tier between ③ and ④ so the expensive model sees
   a better-ordered 80.

5. **Pooled evaluation stops working.** With 100k companies, judging the union
   of top-20s samples a vanishing fraction. *Fix:* stratified sampling with
   importance weighting, plus a fixed labelled regression set that must not
   degrade between releases.

**Architecturally I would change one thing:** split the offline and online
paths properly. Corpus embedding, NAICS enrichment and dedup become a batch
job writing to a real store (pgvector or similar, with `country_code`,
`employee_count`, `revenue` as indexed columns so stage ② is a SQL `WHERE`
rather than a Python loop). The online path keeps exactly the six stages it has
now.

---

## 3.5 Failure modes — when it is confidently wrong

The dangerous failures are not the ones that return nothing. They are the ones
that return a clean, plausible, well-reasoned list that is wrong.

**1. Planner over-specification (highest risk, observed).** The planner adds a
qualifier the user never wrote, and everything downstream faithfully enforces
it. Observed on query 2 — see 3.3. Nothing downstream can detect this, because
the pipeline's job is to be faithful to the plan. *Monitor:* rate of queries
returning zero or near-zero results; distribution of shortlist→qualified ratio
per query; alert when a query's ratio collapses relative to its historical
value.

**2. Silent constraint dropping (observed).** A constraint the data cannot
express gets quietly reinterpreted as something weaker — "using Shopify" became
"has an online store". The output looks confident and is well-reasoned; it just
answers a different question. This is the worst failure mode because there is
no error signal at all. *Monitor:* diff the query's noun phrases against the
`QuerySpec` fields and flag constraints that appear in neither a hard filter
nor a criterion; surface "interpreted as" to the user.

**3. Confidence is not calibrated.** The cheap model returns 1.0 for nearly
everything it accepts. Confidence is usable as an *ordering* signal and as an
escalation trigger, but its absolute value means little. Any threshold tuned
against it is fragile. *Monitor:* reliability curve of confidence vs judged
grade, recomputed per release.

**4. Rich-record bias.** Qualification reads a description. A company with a
long, well-written description is easier to qualify than an equally relevant
one with two lines. The system therefore systematically favours companies with
good marketing copy — a bias correlated with company size and
English-language presence. *Monitor:* mean `completeness()` and description
length of returned vs non-returned companies; if returned companies are
consistently richer, recall is being lost on thin records.

**5. Batch contamination.** Eight companies share a prompt, and which eight
depends on retrieval order, so a strong match early in a batch may shift
verdicts for its neighbours. `evaluation/consistency.py` measures this --
repeat runs with the cache disabled, reporting set Jaccard, top-10 overlap and
per-company verdict flips -- but **I did not get to run it**: the free-tier
quota was exhausted by the benchmark and the judge. So this risk is argued,
not measured, and I would not claim the system is stable until that number
exists. *Monitor:* verdict flip rate under batch reshuffling.

**6. Judge/qualifier correlation.** The evaluation judge is the same model
family as the system it grades, so shared blind spots are invisible. The
hand-labelled calibration set exists precisely to bound this, and it is the
reason I would not quote the headline metrics to a customer without saying how
they were produced.

**7. Dataset-shape assumptions.** The geography table, the NAICS grading curve
and the shortlist depths were chosen against a 456-company, Europe-heavy,
manufacturing-heavy sample. A database of 100k US SMBs would need all three
revisited.

### What I would build next, in order

1. **Fix BM25 to a persistent inverted index** — the only component that
   genuinely does not scale.
2. **Constraint-coverage check** — assert every constraint in the query landed
   somewhere in the `QuerySpec`; surface unmapped ones to the user as "could
   not be checked". Directly attacks failure mode 2, the most dangerous one.
3. **Plan verification pass** — a second cheap call asking "is this plan
   narrower than the query?" before spending anything. Attacks failure mode 1.
4. **Confidence calibration** — fit verdict probability against the labelled
   set so escalation thresholds mean something.
5. **A real labelled regression set** — a few hundred hand-adjudicated pairs
   is the difference between tuning and guessing.

---

## Appendix: operating constraints that shaped the results

Everything here ran against a free-tier Google AI Studio key, and that had
real consequences for the numbers. Stating them so the results are not read
as more general than they are.

**The model tiers are narrower than designed.** `gemini-2.5-flash-lite` and
`gemini-2.5-pro` are retired for new keys; `gemini-3.5-flash` and the Gemini 3
Pro models returned a hard 429 (no free-tier allowance) on every attempt. The
larger flash models exhaust a small daily allowance quickly. In practice the
cheap tier, the escalation tier and the judge all sometimes resolve to lite
models. **When that happens, escalation is not escalating to a stronger model
— only to a different one thinking harder, and the measured benefit of the
escalation tier is therefore a lower bound.** On a paid key with a genuine
pro tier available for stages ⑤ and the judge, I would expect both the
escalation gain and the judge's reliability to improve.

**Rate limits are the dominant latency term, and they flatter the cascade.**
The embedding endpoint meters each *text* in a batch separately, capped at 100
per minute, so the one-off corpus embedding takes ~5 minutes for 456
companies. Generation is far tighter: the 429 response body names the figure,
and for `gemini-3.5-flash-lite` it is **15 requests per minute**.

That number reframes the cost argument entirely. Baseline A needs one call per
company, so 456 calls per query is **over 30 minutes for a single query** at
the free tier's ceiling — not slow, but structurally infeasible. The cascade
needs about seven calls and finishes in seconds. The architecture is not
merely cheaper; under a realistic quota it is the difference between a system
that works and one that does not.

It also cost me real time to learn. I initially set the client's limiter to
110 requests/minute, seven times the actual ceiling, which converted the
entire allowance into 429s and backoff — the client spent its time being
rejected rather than working. The honest general claim, independent of quota,
is the **call count**: 9.3 versus 456 per query on average, a 49x reduction. On a
paid tier with real concurrency the wall-clock gap would narrow considerably,
and I would not quote these seconds-vs-minutes figures as if they were
intrinsic.

**Caching makes repeat runs free, which can mislead.** Results are cached by
content hash, so a second run of an unchanged configuration reports near-zero
cost and near-zero time. Any cost or latency figure quoted from a run with
cache hits understates a cold run; `results/results.json` records
`cache_hits` per query so this is checkable rather than hidden.

**Single-annotator labels.** The calibration set was labelled by one person
(me), so it measures judge-vs-author agreement, not judge-vs-consensus. It
bounds how far the judge can be trusted; it does not establish ground truth.

**The dataset is small and skewed.** 456 companies after deduplication,
heavily weighted to Europe (285), manufacturing and professional services,
with 86 US companies. Several benchmark queries have only a handful of true
answers in the data, so per-query metrics move a lot on one or two decisions.
Treat the per-query table as directional and the aggregate as weakly
supported.
