# Evaluation report

Judge: `gemini-3.8-flash` | pool depth 20 | relevance threshold grade >= 2

## Systems (mean over queries)

| System | Queries | P@5 | P@10 | P@20 | nDCG@10 | nDCG@20 | MAP | Avg returned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cascade (ours)** | 10 | 0.840 | 0.816 | 0.796 | 0.874 | 0.903 | 0.834 | 24.5 |
| Baseline A (LLM per company, pooled) | 3 | 0.933 | 0.889 | 0.856 | 0.876 | 0.896 | 0.981 | 16.3 |
| Baseline B (embedding only) | 10 | 0.660 | 0.660 | 0.605 | 0.748 | 0.800 | 0.711 | 20.0 |

## Cost and latency

| System | Queries | Wall seconds | LLM calls | Cost USD |
|---|---:|---:|---:|---:|
| **Cascade (ours)** | 10 | 0.2 | 0 | $0.00000 |
| Baseline B (embedding only) | 10 | 0.0 | 0 | $0.00000 |
| Baseline A (LLM per company, pooled) | 3 | 0.0 | 0 | $0.00000 |

## Per-query

| Query | Kind | Detected | Pool | Relevant | Cascade P@10 | B P@10 | Cascade nDCG@10 | B nDCG@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic companies in Romania | structured | STRUCTURED | 20 | 6 | 0.857 | 0.600 | 0.950 | 0.855 |
| Public software companies with more than 1,000 employees | structured | STRUCTURED | 34 | 0 | 0.000 | 0.000 | 1.000 | 0.691 |
| Food and beverage manufacturers in France | structured | STRUCTURED | 20 | 20 | 1.000 | 1.000 | 0.962 | 1.000 |
| Companies that could supply packaging materials for a di | reasoning | REASONING | 21 | 18 | 1.000 | 0.900 | 0.781 | 0.658 |
| Construction companies in the United States with revenue | structured | STRUCTURED | 21 | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pharmaceutical companies in Switzerland | structured | STRUCTURED | 24 | 24 | 1.000 | 1.000 | 0.926 | 0.901 |
| B2B SaaS companies providing HR solutions in Europe | semantic | SEMANTIC | 33 | 33 | 1.000 | 1.000 | 0.864 | 0.883 |
| Clean energy startups founded after 2018 with fewer than | semantic | SEMANTIC | 35 | 20 | 0.900 | 0.200 | 0.883 | 0.273 |
| Fast-growing fintech companies competing with traditiona | reasoning | REASONING | 20 | 8 | 1.000 | 0.700 | 0.839 | 0.805 |
| E-commerce companies using Shopify or similar platforms | reasoning | SEMANTIC | 9 | 7 | 0.400 | 0.200 | 0.533 | 0.409 |

> **2 of 12 queries could not be judged** and are excluded from every
> number above: `q11, q12`. The judge hit an exhausted API quota.

*Recall is measured against the judged pool, not the whole database: a
relevant company that no system retrieved cannot appear in these numbers.*