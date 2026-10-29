# Analysis implementation and interpretation

`rsi_audit.analysis` uses the Python standard library. It reads an already
audited run; it never requests model inference, opens a hidden task bundle,
changes a policy, or selects a checkpoint using audit scores.

```python
from pathlib import Path
from rsi_audit.analysis import analyze_run, write_report

report = analyze_run(Path("runs/example"), bootstrap_samples=2000,
                     bootstrap_seed=20261028)
write_report(report, Path("runs/example/report"))
```

The outputs are `analysis.json` and `report.md`. A `demo` run is labeled
**synthetic, with no GPU experiment and no benchmark-performance claim**.
A `pilot` run is labeled development feasibility work, not the formal endpoint.

## Required input contract

- `config.json`: the actual `search.arms`, `search.proposer_seeds`, and
  `search.rounds` used in this run. Demo and pilot configurations must state
  their smaller actual scope rather than retain formal-study counts.
- `state.json`: schema version 1, mode `demo`, `pilot`, or `formal`, phase
  `audited`, and one run for every configured arm/seed pair. Policy references
  are hash strings. Every configured round has parent/candidate references,
  an `accepted` boolean, and its observed confirmation gain. Terminal
  development-only updates must set `phase: "dev_update"` and may have a null
  confirmation gain.
- `audit_manifest.json`: an object with `tasks`, each containing `task_id`,
  `source`, and `family_id`. This expected denominator is independent of the
  observed audit rows. For compatibility, `config.engine.audit_tasks` can
  supply the same list when no independent manifest exists.
- `audit.jsonl`: one boolean `passed` record for every expected task and every
  referenced initial, final, parent, and candidate policy in each arm/seed.
  Extra complete policy audits are disclosed but never used to select the
  primary endpoint. Incomplete extra policy matrices are also rejected.
- `ledger.jsonl`: trusted engine events, optionally under an `event` wrapper.

When state contains `audit_manifest_sha256` and `audit_jsonl_sha256`, their
byte hashes must match. The engine owns freezing, audit authorization, and
ledger authenticity. Hash matching alone does not prove that execution was
authentic. The report discloses whether each optional hash was available.

Missing tasks, missing seed pairs, duplicate audit keys, metadata disagreement,
nonboolean outcomes, incomplete rounds, and unfinished phases raise
`AnalysisError`. Even a task missing from **every** observed checkpoint is
detected using the independent manifest. There is no imputation or silent
complete-case deletion.

## Primary endpoint and resampling

The sole primary comparison is the mean paired task difference between
`fresh` and `reuse`, using each run's `final_policy`. Historical best scores
never enter endpoint selection. Per-seed effects and each arm's difference
from its own initial policy are retained.

Each bootstrap replicate resamples the search seeds once, then resamples
whole task families once within source strata. Those same selections apply
to both arms and to every task in a selected family. Unequal-size families
contribute all their tasks rather than one family-average vote. The original
task-count weight of each source is retained while its within-source mean is
recomputed from the resampled families.

A family that spans sources is assigned to the corresponding combination
stratum, so source stratification cannot split correlated tasks. This conditions
on the observed source-combination structure; sparse strata provide limited
resampling information and should not be interpreted as high certainty.

The interval is the 2.5th and 97.5th percentiles of these paired crossed
bootstrap effects. It is an uncertainty description for a finite collection
of tasks and searches. Small seed counts, single-family strata, and few
bootstrap repetitions limit its interpretation. It is not a no-regression
guarantee, a proof of equivalence, or a guarantee about other models.

## Checkpoint audit: explicit implementation amendment

For the same candidate and parent pair, the module reports:

```text
audit_gain = candidate_audit_rate - parent_audit_rate
optimism = observed_confirmation_gain - audit_gain
```

Development-only Terminal updates are included in the descriptive checkpoint
table but excluded from gate-promotion counts. The fraction of accepted
modifications with a negative audit point estimate is descriptive. When no
modification was accepted, this fraction is JSON `null` and Markdown `NA`.

**This implementation does not implement the proposal's prospective Holm-
adjusted checkpoint support labels.** It reports observed improvement,
regression, or tie, with inferential support `not_assessed`. It does not label
any checkpoint statistically safe or harmful. A negative audit point estimate
is not a known population false-promotion event. This amendment is also included
in every machine-readable and Markdown report.

## Cost accounting

Only `solve` and `proposal` events contribute model-call and token totals;
`gate` summaries do not duplicate those costs. Logical totals include exact
cache reuses. Physical totals exclude events marked `cache_hit: true`.
Model decoding and candidate execution seconds remain separate measurements.

GPU hours come only from explicit `phase_time` intervals:

```text
reported_GPU_hours = sum(wall_seconds * gpu_count / 3600)
```

Intervals must be nonoverlapping engine measurements. Missing telemetry is
not reconstructed from token throughput or decoding seconds. Reports show
the sum of recorded quantified intervals and disclose unquantified intervals;
they do not certify that those intervals cover the entire project. A synthetic
demo rejects nonzero GPU-count intervals.

## Validation

Run the focused offline checks with:

```bash
python -m unittest discover -s tests -p test_analysis.py -v
```

These tests use hand-computable fixtures to check the primary paired effect,
initial baseline, final-versus-historical selection, zero-acceptance handling,
Terminal update accounting, complete denominators, duplicate detection,
family and seed correlation, source weighting, byte-hash mismatches, proposer
costs, cache accounting, GPU time provenance, and synthetic report labeling.
Passing them validates these analytics behaviors; it is not a GPU benchmark
result or an end-to-end security attestation.
