# RSI-Audit analysis

**SYNTHETIC DEMO — no GPU experiment; no real benchmark-performance claim.**

The values below are generated test fixtures that exercise the pipeline. They are not research findings.

Mode: `demo`. Audit tasks: **16**; task families: **16**; search seeds: **3**.

## Prespecified primary endpoint

Only declared final deployed checkpoints enter the primary contrast. Historical best audit scores are never selected.

Fresh − Reuse: **+8.33 percentage points**; paired crossed-bootstrap 95% interval **[-2.08, +33.33] pp**.

Bootstrap replicates: 2000; deterministic analysis seed: 20261028. No significance or general safety label is assigned.

| Seed | Fresh solve rate (%) | Reuse solve rate (%) | Difference (pp) |
| --- | ---: | ---: | ---: |
| 11 | 68.75 | 68.75 | +0.00 |
| 29 | 43.75 | 43.75 | +0.00 |
| 47 | 68.75 | 43.75 | +25.00 |

## Final checkpoints and initial baseline

| Arm | Final solve rate (%) | Initial solve rate (%) | Paired gain over initial (pp) |
| --- | ---: | ---: | ---: |
| reuse | 52.08 | 43.75 | +8.33 |
| fresh | 60.42 | 43.75 | +16.67 |
| terminal | 52.08 | 43.75 | +8.33 |

## Descriptive promotion audit

Negative differences describe this audit sample. They are not certified population errors; checkpoint comparisons are dependent. Zero accepted promotions have an undefined (NA) negative fraction.

| Arm | Gate decisions | Accepted | Observed negative audit differences | Negative fraction (%) |
| --- | ---: | ---: | ---: | ---: |
| reuse | 9 | 1 | 0 | 0.00 |
| fresh | 9 | 2 | 0 | 0.00 |
| terminal | 3 | 1 | 0 | 0.00 |

Every candidate/parent audit difference and validation-minus-audit optimism value is retained in `synthetic-analysis.json`. Terminal development updates do not count as gate promotions.

## Actual cost ledger

| Phase | Logical input/output tokens | Physical input/output tokens | Physical model calls | Cache hits | Reported GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| audit | 0 / 0 | 0 / 0 | 0 | 320 | 0.000000 |
| freeze | 0 / 0 | 0 / 0 | 0 | 0 | 0.000000 |
| search | 0 / 0 | 0 / 0 | 0 | 728 | 0.000000 |
| unspecified | 0 / 0 | 0 / 0 | 0 | 0 | NA |

Reported interval GPU hours: **0.000000**. This is wall time × GPU count from explicit phase intervals; model decoding seconds are not GPU hours. Coverage is limited to recorded intervals. Unquantified time intervals: 0.

Aborted operations: 0; attempts with incomplete token telemetry: 0. Physical-token totals are a lower bound if any attempt is unquantified; phase wall time still includes handled failures.

## Integrity and limitations

All declared seed/arm pairs and all referenced policy/task combinations were required. Missing observations were not converted to failures or dropped. Input content hashes are reported in `synthetic-analysis.json`; hashes alone are not proof of authentic execution.

- SYNTHETIC DEMO: no GPU experiment and no real benchmark-performance claim. Synthetic successes are not empirical research results.
- The interval is a finite-data crossed bootstrap description, not a universal safety, no-regression, or population guarantee.
- Repeated tasks across seeds are not independent new tasks; six seeds do not remove limited task diversity.
- Checkpoint negative fractions are descriptive and dependent; no Holm-adjusted support labels or global false-promotion guarantee are implemented.
- A null or wide interval does not establish equivalence; actual costs can differ under common ceilings.
- Fewer than six search seeds are present; this configuration must not be represented as the full six-seed study.

## Implementation amendment

This version implements descriptive checkpoint audit differences, not the proposal's prospective Holm-adjusted support labels. A negative point estimate is labeled observed regression only; inferential support remains unassessed. The primary interval is not a global no-regression guarantee.
