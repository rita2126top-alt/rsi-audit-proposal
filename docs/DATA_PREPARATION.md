# Data preparation and split commitments

The data path is implemented with the Python standard library. It does not run
reference programs, generated programs, shell commands, or model inference.
Reference outputs are prepared separately through the isolated execution layer.

The allocator and bundle format additionally accept `source="synthetic"` for
explicitly labeled offline demos. The real EvalPlus loader never produces that
source. Main-study and real-data pilot orchestration must reject synthetic
tasks; synthetic demonstrations are not benchmark results.

## Pinned sources and exact input semantics

The authoritative archive hashes, decoded JSONL hashes, byte counts, task counts,
and release URLs are in `evidence/baseline-lock.json`:

| Source | Release | Expected tasks |
| --- | --- | ---: |
| HumanEval+ | v0.1.10 | 164 |
| MBPP+ | v0.2.0 | 378 |

Input restoration was checked against EvalPlus commit
`e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`, specifically
[data/mbpp.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/data/mbpp.py),
[data/humaneval.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/data/humaneval.py), and
[evaluate.py](https://github.com/evalplus/evalplus/blob/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/evaluate.py).

For both datasets, the reference program is `prompt + canonical_solution`.
HumanEval typically supplies a function header and docstring in the prompt and
the indented body in the solution. MBPP supplies a module docstring prompt and
a complete implementation. Simply executing a HumanEval solution body, or
assuming every reference is a complete standalone function, is incorrect.

MBPP's JSON representation loses some Python types. The loader implements the
pinned upstream task-ID conversion table, restoring tuples, nested tuples,
sets, complex numbers, and tuple-valued dictionaries. In particular, task 115's
empty list becomes an empty dictionary, matching the upstream implementation;
124 restores a float and a complex argument; 252 restores a complex argument;
580/615/791 recursively restore tuples. The module never `eval`s a value.
Unrecognized malformed argument structures fail rather than being skipped.

The exact locked MBPP+ archive has one explicit format exception: `Mbpp/793`
stores its empty `plus_input` as `{}`. Upstream iteration executes zero added
cases. The loader records a `data_notes` entry and converts only that known
record's empty added-test collection to `[]`, retaining its base cases and the
task itself. Other non-list collections fail validation.

## Obtain and verify archives

From the repository root, on a machine with network access:

```bash
python tools/download_assets.py --output private-assets
```

The destination must not already exist. The downloader uses the exact lockfile
URLs, caps download bytes, and checks both hashes and the complete task count.
It creates a mode-0700 directory and mode-0600 files. It downloads no model
weights and incurs no inference calls.

Archives obtained separately can be loaded directly:

```python
from pathlib import Path
from rsi_audit.data import load_evalplus

tasks = load_evalplus(
    Path("private-assets/HumanEvalPlus.jsonl.gz"),
    Path("private-assets/MbppPlus.jsonl.gz"),
    Path("evidence/baseline-lock.json"),
)
```

An already decompressed JSONL file is accepted only when its exact payload hash
matches. A decompressed file cannot establish the original gzip wrapper hash;
the locked JSONL digest still authenticates the benchmark payload. Gzip inputs
must match both the archive and decoded payload hashes. Do not silently change
line endings, filter records, or use a different release.

Each task contains `task_id`, `source` (`humaneval` or `mbpp`), `family_id`,
`prompt`, `entry_point`, `canonical_code`, `inputs`, `atol`, `public_examples`,
base/added case counts, and data notes. `inputs` is the full base-plus-added
sequence of positional-argument lists with original Python types restored.

## Public examples and private tests

Structured public examples are extracted **only from the original prompt**,
using AST literal parsing of direct-call equality assertions and direct-call
doctests. Their shape is `{"args": [...], "kwargs": {...}, "expected": value}`.
No reference code, canonical solution, hidden `base_input`, or `plus_input` is
used to construct a public example.

An expression such as `set(f(x)) == {1, 2}` or
`math.isclose(f(x), y, rel_tol=...)` cannot be reduced to the exact return value
of `f` without changing its meaning. These compound examples remain visible
in the full prompt but are omitted from the structured example list. Likewise,
variable-dependent, nonliteral, or multiline assertions that cannot be safely
represented are omitted. A task can legitimately have an empty structured list;
this must never trigger a fallback to hidden tests. No prompt code is executed
during extraction.

## Whole-family deterministic allocation

```python
from rsi_audit.data import partition_tasks

parts = partition_tasks(tasks, dev_count=128, panel_count=6,
                        panel_size=32, audit_min=222, seed=42,
                        family_overrides=None)
```

The loader and allocator merge normalized exact prompt matches and exact
reference-function AST matches (ignoring the entry-point name and docstring).
Existing family labels are preserved. This is conservative exact-match grouping;
it does not prove semantic deduplication or training-data decontamination.

A human-reviewed near-duplicate map may additionally merge families:

```json
{"HumanEval/12": "reviewed-group-A", "Mbpp/42": "reviewed-group-A"}
```

Keys must be existing task IDs; values must be nonempty shared string labels.
These illustrative IDs do not assert a real duplicate pair. Overrides can merge
families but cannot split automatic families. Record the review and map hash in
the private preregistration metadata.

All variants of a task and all members of a family stay in exactly one dev,
confirmation panel, or final-audit partition. Confirmation panels are also
mutually family-disjoint. The allocator uses exact subset feasibility and
backtracking over family size/source-count types, preferring source proportions
close to those of the input pool. The same seed and task content produce the
same allocation regardless of input record order. Family constraints can make
exact source quotas impossible; actual counts are reported rather than hidden.

Dev and every panel must have the requested exact sizes. The remainder is the
audit set and must meet `audit_min`. If no whole-family allocation exists, the
operation fails and the preregistered design must be revised. It never breaks a
family, silently drops a task, or refills a split from an unrelated source.

## Private bundles and public commitments

```python
from rsi_audit.data import write_bundle, read_partition, verify_commitment

manifest = write_bundle(parts, Path("private-bundle"),
                        {"split_seed": 42, "protocol_version": "pre-main-1"})
dev = read_partition(Path("private-bundle"), "dev")
panels = read_partition(Path("private-bundle"), "panels")
# Explicit audit action, outside the search process:
assert verify_commitment(Path("private-bundle"))
```

`write_bundle` refuses an existing destination, creates mode-0700 directory and
mode-0600 files, and preserves non-JSON Python types through the shared wire
codec. Private files are `dev.json`, `panels.json`, `audit.json`,
`metadata.private.json`, and `secret.private.json`. Metadata stays private.

`public_manifest.json` contains only schema, counts, file hashes, and a split
commitment. It exposes no task IDs, prompts, tests, reference code, or salt.
The split commitment is SHA-256 of canonical UTF-8 JSON containing the ordered
split task IDs and a cryptographically random 32-byte salt. Canonical JSON sorts
object keys, uses compact separators, and preserves Unicode. File hashes bind
the full serialized contents, not just the split IDs.

Publish or otherwise externally timestamp the public manifest before the main
run. Local verification proves internal consistency against that manifest; it
cannot detect an adversary replacing the manifest and every file together.
The published manifest is the external commitment. Do not expose the private
salt or private split files to the searching agent.

`read_partition(..., "dev")` reads only `dev.json` and the public manifest.
`read_partition(..., "panels")` likewise does not open `audit.json` or the
secret. `verify_commitment` intentionally reads every partition, including the
audit set, and therefore belongs to preparation/audit orchestration rather than
the search runtime. Filesystem mode bits do not isolate processes sharing the
same operating-system user; the main runner must withhold audit files from the
model's tool/container mounts and capabilities.

## Pilot isolation

`pilot_bundle_from_dev(parent, destination, dev_count=16, panel_count=2,
panel_size=8, audit_min=8, audit_count=8, seed=0, metadata=None)` loads only the parent's dev
partition and creates a separately salted **40-task** bundle from it: 16 dev,
two panels of 8, and exactly 8 pilot audit tasks. All four bins are allocated
jointly as whole families; unused parent-dev families are excluded from the
pilot, never silently reassigned to its audit set. Pilot confirmation
and pilot audit sets therefore contain only main-development tasks. The
parent's confirmation panels and final audit are never opened by this helper.
The helper retains whole-family allocation rules and may fail for infeasible
pilot sizes. Pilot metrics must never be presented as the main study's audit.
The `audit_count` argument is optional for API compatibility. If omitted, the
helper retains all parent-dev tasks and uses the remaining tasks for pilot audit;
the study CLI always passes `audit_count=8` for the registered 40-task pilot.

## Tests and limits

```bash
python -m unittest discover -s tests -p 'test_data.py' -v
```

Tests cover hash/count corruption, plain-versus-gzip verification, official
MBPP type restorations, public-example extraction, duplicate IDs, family
isolation, a non-greedy feasible allocation, impossible allocations, file
permissions, salted commitments, content tampering, and pilot read isolation.
The loader uses only bounded public benchmark assets but Python AST parsing is
not a general security sandbox. All actual reference and generated-code
execution must go through the separate isolated executor.
