"""Read-only, dependency-free analysis of an already frozen and audited run.

The primary endpoint always uses each run's declared final deployed policy.
No model calls, checkpoint selection, or missing-as-failure substitutions occur
here.  Checkpoint regressions are descriptive, not certified error probabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class AnalysisError(ValueError):
    """An input is inconsistent, incomplete, or unsafe to summarize."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Cannot read valid JSON from {path.name}: {exc}") from exc


def _jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f"{path.name}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise AnalysisError(f"{path.name}:{line_number}: expected an object")
                yield value
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"Cannot read {path.name}: {exc}") from exc


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise AnalysisError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AnalysisError(f"Cannot hash {path.name}: {exc}") from exc


def _manifest(run_dir: Path, config: dict, state: dict) -> tuple[list[dict], dict]:
    path = run_dir / "audit_manifest.json"
    manifest_hash = None
    if path.is_file():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise AnalysisError("audit_manifest.json must be an object with a tasks list")
        tasks = payload.get("tasks")
        manifest_hash = _sha256(path)
        expected = state.get("audit_manifest_sha256")
        if expected is not None and expected != manifest_hash:
            raise AnalysisError("audit_manifest.json SHA-256 does not match audited state")
        origin = "audit_manifest.json"
    else:
        tasks = config.get("engine", {}).get("audit_tasks")
        if state.get("audit_manifest_sha256") is not None:
            raise AnalysisError("State references a missing audit_manifest.json")
        origin = "config.engine.audit_tasks"
    if not isinstance(tasks, list) or not tasks:
        raise AnalysisError("A complete expected audit task manifest is required; rows cannot define their own denominator")
    result = []
    seen = set()
    for item in tasks:
        if not isinstance(item, dict):
            raise AnalysisError("Audit manifest task entries must be objects")
        row = {key: _text(item.get(key), f"manifest.{key}") for key in ("task_id", "source", "family_id")}
        if row["task_id"] in seen:
            raise AnalysisError(f"Duplicate task in audit manifest: {row['task_id']}")
        seen.add(row["task_id"])
        result.append(row)
    result.sort(key=lambda item: item["task_id"])
    return result, {"manifest_origin": origin, "manifest_sha256": manifest_hash,
                    "manifest_hash_checked": state.get("audit_manifest_sha256") is not None}


def _runs(config: dict, state: dict) -> tuple[list[str], list[int], dict]:
    search = config.get("search", {})
    arms = search.get("arms")
    seeds = search.get("proposer_seeds")
    if not isinstance(arms, list) or not arms or not all(isinstance(a, str) and a for a in arms):
        raise AnalysisError("config.search.arms must list actual study arms")
    if len(set(arms)) != len(arms) or not {"fresh", "reuse"}.issubset(arms):
        raise AnalysisError("Unique arms including fresh and reuse are required for the primary contrast")
    if not isinstance(seeds, list) or not seeds:
        raise AnalysisError("config.search.proposer_seeds must list actual study seeds")
    seeds = [_integer(seed, "proposer seed") for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise AnalysisError("Duplicate configured proposer seed")
    rounds_count = _integer(search.get("rounds"), "config.search.rounds", 1)
    items = state.get("runs")
    if not isinstance(items, list):
        raise AnalysisError("state.runs must be a list")
    runs = {}
    expected_keys = {(arm, seed) for arm in arms for seed in seeds}
    for run in items:
        if not isinstance(run, dict):
            raise AnalysisError("Each run must be an object")
        arm = _text(run.get("arm"), "run.arm")
        seed = _integer(run.get("seed"), "run.seed")
        key = (arm, seed)
        if key not in expected_keys:
            raise AnalysisError(f"Unexpected run/seed pair: {key}")
        if key in runs:
            raise AnalysisError(f"Duplicate run/seed pair: {key}")
        _text(run.get("initial_policy"), "initial_policy")
        _text(run.get("final_policy"), "final_policy")
        rounds = run.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != rounds_count:
            raise AnalysisError(f"Incomplete rounds for {key}; expected {rounds_count}")
        seen_rounds = set()
        for transition in rounds:
            if not isinstance(transition, dict):
                raise AnalysisError("Round entries must be objects")
            round_number = _integer(transition.get("round"), "round number", 1)
            if round_number in seen_rounds or round_number > rounds_count:
                raise AnalysisError(f"Duplicate or out-of-range round for {key}")
            seen_rounds.add(round_number)
            for field in ("parent_policy", "candidate_policy"):
                _text(transition.get(field), field)
            if not isinstance(transition.get("accepted"), bool):
                raise AnalysisError("Round accepted must be a boolean")
            for field in ("confirm_gain", "dev_gain"):
                if transition.get(field) is not None:
                    gain = _number(transition[field], field)
                    if not -1 <= gain <= 1:
                        raise AnalysisError(f"{field} must be a rate difference in [-1, 1]")
            if transition.get("phase") != "dev_update" and transition.get("confirm_gain") is None:
                raise AnalysisError("A gate round must record confirm_gain; mark non-gate updates phase='dev_update'")
        runs[key] = run
    missing = expected_keys - set(runs)
    if missing:
        raise AnalysisError(f"Missing run/seed pairs: {sorted(missing)}")
    for seed in seeds:
        if len({runs[arm, seed]["initial_policy"] for arm in arms}) != 1:
            raise AnalysisError(f"Arms do not share the initial policy for seed {seed}")
    return arms, seeds, runs


def _audit_matrix(run_dir: Path, manifest: list[dict], state: dict, runs: dict) -> tuple[dict, dict]:
    path = run_dir / "audit.jsonl"
    file_hash = _sha256(path)
    if state.get("audit_jsonl_sha256") is not None and state["audit_jsonl_sha256"] != file_hash:
        raise AnalysisError("audit.jsonl SHA-256 does not match audited state")
    task_meta = {task["task_id"]: (task["source"], task["family_id"]) for task in manifest}
    referenced = {}
    for key, run in runs.items():
        policies = {run["initial_policy"], run["final_policy"]}
        for transition in run["rounds"]:
            policies.update((transition["parent_policy"], transition["candidate_policy"]))
        referenced[key] = policies
    matrix = defaultdict(dict)
    row_count = 0
    for row in _jsonl(path):
        arm = _text(row.get("arm"), "audit.arm")
        seed = _integer(row.get("seed"), "audit.seed")
        run_key = (arm, seed)
        if run_key not in runs:
            raise AnalysisError(f"Audit row has an unexpected run/seed: {run_key}")
        policy = _text(row.get("policy_hash"), "audit.policy_hash")
        task = _text(row.get("task_id"), "audit.task_id")
        if task not in task_meta:
            raise AnalysisError(f"Unexpected task outside the audit manifest: {task}")
        if (row.get("source"), row.get("family_id")) != task_meta[task]:
            raise AnalysisError(f"Audit task source/family metadata mismatch: {task}")
        if not isinstance(row.get("passed"), bool):
            raise AnalysisError(f"Audit passed must be boolean for {task}")
        checkpoint_key = (arm, seed, policy)
        if task in matrix[checkpoint_key]:
            raise AnalysisError(f"Duplicate audit key: {checkpoint_key + (task,)}")
        matrix[checkpoint_key][task] = int(row["passed"])
        row_count += 1
    expected_tasks = set(task_meta)
    for key, policies in referenced.items():
        for policy in policies:
            checkpoint_key = key + (policy,)
            observed = set(matrix.get(checkpoint_key, {}))
            if observed != expected_tasks:
                missing = sorted(expected_tasks - observed)
                raise AnalysisError(f"Incomplete audit matrix for {checkpoint_key}: missing {len(missing)} tasks {missing[:5]}")
    extra_checkpoints = []
    for checkpoint, values in matrix.items():
        if set(values) != expected_tasks:
            raise AnalysisError(f"Incomplete audit matrix for additional checkpoint {checkpoint}")
        if checkpoint[2] not in referenced[checkpoint[:2]]:
            extra_checkpoints.append({"arm": checkpoint[0], "seed": checkpoint[1], "policy_hash": checkpoint[2]})
    return dict(matrix), {"audit_rows": row_count, "audit_sha256": file_hash,
                          "audit_hash_checked": state.get("audit_jsonl_sha256") is not None,
                          "checkpoint_count": len(matrix), "additional_checkpoint_rows_not_used_for_selection": extra_checkpoints}


def _cluster_layout(manifest: list[dict]) -> tuple[dict, dict, dict]:
    """Keep cross-source families together, while retaining source task weights."""
    families = defaultdict(list)
    source_indices = defaultdict(list)
    for index, task in enumerate(manifest):
        families[task["family_id"]].append(index)
        source_indices[task["source"]].append(index)
    strata = defaultdict(list)
    for family, indices in sorted(families.items()):
        source_set = tuple(sorted({manifest[index]["source"] for index in indices}))
        strata[source_set].append(family)
    return dict(families), dict(source_indices), dict(strata)


def _paired_bootstrap(values_by_seed: list[list[float]], manifest: list[dict], samples: int, seed: int) -> list[float]:
    """Crossed seed/family bootstrap; both axes are shared across the contrast.

    Resample seeds once per replicate and families once within each source
    stratum. Families spanning sources form a combination stratum, so they are
    never split. Each original source retains its original task-count weight;
    unequal-size families contribute all their tasks, not one averaged vote.
    """
    families, source_indices, strata = _cluster_layout(manifest)
    rng = random.Random(seed)
    seed_count = len(values_by_seed)
    task_count = len(manifest)
    draws = []
    for _ in range(samples):
        selected_seeds = [rng.randrange(seed_count) for _ in range(seed_count)]
        seed_means = [sum(values_by_seed[s][i] for s in selected_seeds) / seed_count for i in range(task_count)]
        multiplicity = [0] * task_count
        for source_set in sorted(strata):
            family_ids = strata[source_set]
            for _family_draw in family_ids:
                chosen = family_ids[rng.randrange(len(family_ids))]
                for index in families[chosen]:
                    multiplicity[index] += 1
        value = 0.0
        for source in sorted(source_indices):
            indices = source_indices[source]
            denominator = sum(multiplicity[index] for index in indices)
            if denominator <= 0:
                raise AnalysisError("Bootstrap lost a source; invalid family/source stratification")
            source_mean = sum(multiplicity[index] * seed_means[index] for index in indices) / denominator
            value += len(indices) / task_count * source_mean
        draws.append(value)
    return draws


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _empty_cost() -> dict:
    return {"solve_events": 0, "proposal_events": 0, "cache_hits": 0,
            "aborted_operations": 0, "unquantified_token_operations": 0,
            "logical_input_tokens": 0, "logical_output_tokens": 0, "logical_model_calls": 0,
            "physical_input_tokens": 0, "physical_output_tokens": 0, "physical_model_calls": 0,
            "physical_model_seconds": 0.0, "physical_execution_seconds": 0.0,
            "reported_gpu_hours": None, "timed_intervals": 0, "unquantified_time_intervals": 0}


def _costs(path: Path, mode: str) -> dict:
    total = _empty_cost()
    by_phase = defaultdict(_empty_cost)
    by_arm = defaultdict(_empty_cost)
    kinds = Counter()
    for wrapper in _jsonl(path):
        event = wrapper.get("event", wrapper)
        if not isinstance(event, dict):
            raise AnalysisError("ledger event must be an object")
        kind = _text(event.get("kind"), "ledger.kind")
        kinds[kind] += 1
        phase = str(event.get("phase", "unspecified"))
        arm = str(event.get("arm", "shared"))
        buckets = (total, by_phase[phase], by_arm[arm])
        if kind in {"solve", "proposal"}:
            cache_hit = event.get("cache_hit", False)
            if not isinstance(cache_hit, bool):
                raise AnalysisError("ledger.cache_hit must be boolean")
            counts = {field: _integer(event.get(field, 0), f"ledger.{field}")
                      for field in ("input_tokens", "output_tokens", "model_calls")}
            model_seconds = _number(event.get("model_seconds", 0), "ledger.model_seconds", True)
            execution_seconds = _number(event.get("execution_seconds", 0), "ledger.execution_seconds", True)
            for bucket in buckets:
                bucket[f"{kind}_events"] += 1
                bucket["cache_hits"] += int(cache_hit)
                for field, value in counts.items():
                    bucket[f"logical_{field}"] += value
                    if not cache_hit:
                        bucket[f"physical_{field}"] += value
                if not cache_hit:
                    bucket["physical_model_seconds"] += model_seconds
                    bucket["physical_execution_seconds"] += execution_seconds
        elif kind == "operation_aborted":
            usage = event.get("known_usage")
            for bucket in buckets:
                bucket["aborted_operations"] += 1
                if usage is None:
                    bucket["unquantified_token_operations"] += 1
                    continue
                if not isinstance(usage, dict):
                    raise AnalysisError("Aborted-operation usage must be an object or null")
                for field in ("input_tokens", "output_tokens", "model_calls"):
                    bucket[f"physical_{field}"] += _integer(usage.get(field, 0), f"aborted.{field}")
                for field in ("model_seconds", "execution_seconds"):
                    bucket[f"physical_{field}"] += _number(usage.get(field, 0), f"aborted.{field}", True)
        elif kind == "phase_time":
            wall = event.get("wall_seconds")
            gpu_count = event.get("gpu_count")
            if wall is None or gpu_count is None:
                for bucket in buckets:
                    bucket["unquantified_time_intervals"] += 1
                continue
            wall = _number(wall, "phase_time.wall_seconds", True)
            gpu_count = _integer(gpu_count, "phase_time.gpu_count")
            if mode == "demo" and gpu_count:
                raise AnalysisError("Synthetic demo cannot claim GPU execution; phase_time.gpu_count must be zero")
            gpu_hours = wall * gpu_count / 3600.0
            for bucket in buckets:
                bucket["reported_gpu_hours"] = (bucket["reported_gpu_hours"] or 0.0) + gpu_hours
                bucket["timed_intervals"] += 1
    return {"total": total, "by_phase": dict(sorted(by_phase.items())), "by_arm": dict(sorted(by_arm.items())),
            "ledger_events_by_kind": dict(sorted(kinds.items())),
            "gpu_time_definition": "Sum of reported non-overlapping phase_time.wall_seconds * gpu_count / 3600; decoding seconds are never GPU hours.",
            "gpu_time_coverage": "reported_phase_intervals_only; absent or unquantified intervals are not inferred",
            "logical_token_definition": "solve/proposal event token counts including exact cache reuses",
            "physical_token_definition": "Uncached solve/proposal tokens plus known usage from aborted operations; incomplete telemetry is explicitly counted",
            "provenance_note": "Reads engine-produced trusted ledger; a content hash alone does not establish authentic execution."}


def analyze_run(run_dir: Path, bootstrap_samples: int = 2000, bootstrap_seed: int = 20261028) -> dict:
    """Analyze a complete audited run without starting or modifying an experiment."""
    run_dir = Path(run_dir)
    bootstrap_samples = _integer(bootstrap_samples, "bootstrap_samples", 1)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    config, state = _read_json(run_dir / "config.json"), _read_json(run_dir / "state.json")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise AnalysisError("config.json and state.json must be objects")
    if state.get("schema_version") != 1:
        raise AnalysisError("Unsupported state schema_version")
    if state.get("phase") != "audited":
        raise AnalysisError("Analysis requires phase='audited'; it cannot run on partial or unfrozen searches")
    mode = state.get("mode")
    if mode not in {"demo", "pilot", "formal"}:
        raise AnalysisError("State mode must be demo, pilot, or formal")
    manifest, integrity = _manifest(run_dir, config, state)
    arms, seeds, runs = _runs(config, state)
    matrix, matrix_integrity = _audit_matrix(run_dir, manifest, state, runs)
    integrity.update(matrix_integrity)
    integrity["complete_expected_task_matrix"] = True
    task_ids = [task["task_id"] for task in manifest]
    task_count = len(task_ids)
    def vector(arm, seed, policy):
        return [matrix[arm, seed, policy][task] for task in task_ids]
    def rate(arm, seed, policy):
        return sum(vector(arm, seed, policy)) / task_count
    arm_results = {}
    for arm in arms:
        per_seed = []
        for seed in seeds:
            run = runs[arm, seed]
            final = rate(arm, seed, run["final_policy"])
            initial = rate(arm, seed, run["initial_policy"])
            per_seed.append({"seed": seed, "initial_policy": run["initial_policy"], "final_policy": run["final_policy"],
                             "initial_solve_rate": initial, "final_solve_rate": final,
                             "gain_over_initial": final - initial})
        arm_results[arm] = {"final_solve_rate": sum(item["final_solve_rate"] for item in per_seed) / len(seeds),
                            "initial_solve_rate": sum(item["initial_solve_rate"] for item in per_seed) / len(seeds),
                            "gain_over_initial": sum(item["gain_over_initial"] for item in per_seed) / len(seeds),
                            "per_seed": per_seed}
    differences = []
    per_seed_contrast = []
    for seed in seeds:
        fresh = vector("fresh", seed, runs["fresh", seed]["final_policy"])
        reuse = vector("reuse", seed, runs["reuse", seed]["final_policy"])
        delta = [f - r for f, r in zip(fresh, reuse)]
        differences.append(delta)
        per_seed_contrast.append({"seed": seed, "fresh_solve_rate": sum(fresh) / task_count,
                                  "reuse_solve_rate": sum(reuse) / task_count,
                                  "difference": sum(delta) / task_count})
    estimate = sum(item["difference"] for item in per_seed_contrast) / len(seeds)
    draws = _paired_bootstrap(differences, manifest, bootstrap_samples, bootstrap_seed)
    interval = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    checkpoints = []
    promotion_summary = {}
    for arm in arms:
        accepted_deltas = []
        gate_count = 0
        for seed in seeds:
            for transition in sorted(runs[arm, seed]["rounds"], key=lambda item: item["round"]):
                parent_rate = rate(arm, seed, transition["parent_policy"])
                candidate_rate = rate(arm, seed, transition["candidate_policy"])
                audit_gain = candidate_rate - parent_rate
                is_gate = transition.get("phase") != "dev_update"
                confirm_gain = transition.get("confirm_gain") if is_gate else None
                gate_count += int(is_gate)
                promotion = is_gate and transition["accepted"]
                if promotion:
                    accepted_deltas.append(audit_gain)
                checkpoints.append({"arm": arm, "seed": seed, "round": transition["round"],
                                    "phase": transition.get("phase", "gate"),
                                    "parent_policy": transition["parent_policy"], "candidate_policy": transition["candidate_policy"],
                                    "accepted": transition["accepted"], "counts_as_gate_promotion": promotion,
                                    "parent_audit_solve_rate": parent_rate, "candidate_audit_solve_rate": candidate_rate,
                                    "audit_gain": audit_gain, "confirm_gain": confirm_gain,
                                    "optimism": None if confirm_gain is None else confirm_gain - audit_gain,
                                    "descriptive_direction": "observed_regression" if audit_gain < 0 else ("observed_improvement" if audit_gain > 0 else "observed_tie"),
                                    "inferential_support": "not_assessed"})
        negative_count = sum(delta < 0 for delta in accepted_deltas)
        promotion_summary[arm] = {"gate_decisions": gate_count, "accepted_count": len(accepted_deltas),
                                   "rejected_count": gate_count - len(accepted_deltas),
                                   "observed_audit_regressions": negative_count,
                                   "observed_negative_fraction": negative_count / len(accepted_deltas) if accepted_deltas else None,
                                   "mean_audit_gain_of_accepted": sum(accepted_deltas) / len(accepted_deltas) if accepted_deltas else None,
                                   "inferential_support": "not_assessed; descriptive sample outcomes only"}
    families, sources, strata = _cluster_layout(manifest)
    warnings = [
        "The interval is a finite-data crossed bootstrap description, not a universal safety, no-regression, or population guarantee.",
        "Repeated tasks across seeds are not independent new tasks; six seeds do not remove limited task diversity.",
        "Checkpoint negative fractions are descriptive and dependent; no Holm-adjusted support labels or global false-promotion guarantee are implemented.",
        "A null or wide interval does not establish equivalence; actual costs can differ under common ceilings.",
    ]
    if mode == "demo":
        warnings.insert(0, "SYNTHETIC DEMO: no GPU experiment and no real benchmark-performance claim. Synthetic successes are not empirical research results.")
    elif mode == "pilot":
        warnings.insert(0, "PILOT ONLY: this is development-partition feasibility work, not the preregistered formal audit result.")
    if len(seeds) < 2 or len(families) < 2:
        warnings.append("At least one resampling axis has fewer than two units; a narrow or degenerate interval is not evidence of certainty.")
    if len(seeds) < 6:
        warnings.append("Fewer than six search seeds are present; this configuration must not be represented as the full six-seed study.")
    if bootstrap_samples < 500:
        warnings.append("Fewer than 500 bootstrap replicates were requested; interval endpoints have substantial Monte Carlo instability.")
    if not integrity["manifest_hash_checked"] or not integrity["audit_hash_checked"]:
        warnings.append("One or more audit content hashes were unavailable; matrix completeness is checked but no missing hash is invented.")
    return {"schema_version": 1, "analysis_version": "0.1.0", "project": config.get("project", "RSI-Audit"),
            "mode": mode, "synthetic": mode == "demo", "checkpoint_rule": "declared_final_deployed_policy_only",
            "data": {"tasks": task_count, "families": len(families), "seeds": seeds, "arms": arms,
                     "source_task_counts": {source: len(indices) for source, indices in sorted(sources.items())},
                     "family_resampling_strata": [{"sources": list(source_set), "families": len(ids)} for source_set, ids in sorted(strata.items())]},
            "integrity": integrity, "arms": arm_results,
            "primary": {"contrast": "fresh-minus-reuse", "estimate": estimate, "estimate_percentage_points": estimate * 100,
                        "ci95": interval, "ci95_percentage_points": [value * 100 for value in interval],
                        "method": "source-weighted, crossed paired seed/family percentile bootstrap",
                        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed, "per_seed": per_seed_contrast,
                        "p_value": None, "significance_claim": None},
            "checkpoints": checkpoints, "promotions": promotion_summary,
            "costs": _costs(run_dir / "ledger.jsonl", mode),
            "protocol_amendments": [{"id": "descriptive-checkpoint-analysis-v1",
                                      "description": "This implementation reports observed checkpoint differences only. Proposed multiplicity-adjusted inferential labels are not implemented; no transition is certified safe or harmful."}],
            "warnings": warnings}


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value: Any, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return format(float(value) * 100, "+.2f" if signed else ".2f")


def write_report(report: dict, output_dir: Path) -> None:
    """Write analysis.json and report.md without implying synthetic significance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    mode = report["mode"]
    lines = ["# RSI-Audit analysis", ""]
    if mode == "demo":
        lines.extend(["**SYNTHETIC DEMO — no GPU experiment; no real benchmark-performance claim.**", "",
                      "The values below are generated test fixtures that exercise the pipeline. They are not research findings.", ""])
    elif mode == "pilot":
        lines.extend(["**DEVELOPMENT PILOT — feasibility results, not the formal study endpoint.**", ""])
    else:
        lines.extend(["**Formal-run summary.** Interpret the observed sample effects with the recorded limitations below.", ""])
    data, primary = report["data"], report["primary"]
    lines.extend([f"Mode: `{mode}`. Audit tasks: **{data['tasks']}**; task families: **{data['families']}**; search seeds: **{len(data['seeds'])}**.", "",
                  "## Prespecified primary endpoint", "",
                  "Only declared final deployed checkpoints enter the primary contrast. Historical best audit scores are never selected.", "",
                  f"Fresh − Reuse: **{_percent(primary['estimate'], True)} percentage points**; paired crossed-bootstrap 95% interval **[{_percent(primary['ci95'][0], True)}, {_percent(primary['ci95'][1], True)}] pp**.", "",
                  f"Bootstrap replicates: {primary['bootstrap_samples']}; deterministic analysis seed: {primary['bootstrap_seed']}. No significance or general safety label is assigned.", "",
                  "| Seed | Fresh solve rate (%) | Reuse solve rate (%) | Difference (pp) |",
                  "| --- | ---: | ---: | ---: |"])
    for row in primary["per_seed"]:
        lines.append(f"| {row['seed']} | {_percent(row['fresh_solve_rate'])} | {_percent(row['reuse_solve_rate'])} | {_percent(row['difference'], True)} |")
    lines.extend(["", "## Final checkpoints and initial baseline", "",
                  "| Arm | Final solve rate (%) | Initial solve rate (%) | Paired gain over initial (pp) |",
                  "| --- | ---: | ---: | ---: |"])
    for arm, row in report["arms"].items():
        lines.append(f"| {_cell(arm)} | {_percent(row['final_solve_rate'])} | {_percent(row['initial_solve_rate'])} | {_percent(row['gain_over_initial'], True)} |")
    lines.extend(["", "## Descriptive promotion audit", "",
                  "Negative differences describe this audit sample. They are not certified population errors; checkpoint comparisons are dependent. Zero accepted promotions have an undefined (NA) negative fraction.", "",
                  "| Arm | Gate decisions | Accepted | Observed negative audit differences | Negative fraction (%) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for arm, row in report["promotions"].items():
        lines.append(f"| {_cell(arm)} | {row['gate_decisions']} | {row['accepted_count']} | {row['observed_audit_regressions']} | {_percent(row['observed_negative_fraction'])} |")
    lines.extend(["", "Every candidate/parent audit difference and validation-minus-audit optimism value is retained in `analysis.json`. Terminal development updates do not count as gate promotions.", "",
                  "## Actual cost ledger", "",
                  "| Phase | Logical input/output tokens | Physical input/output tokens | Physical model calls | Cache hits | Reported GPU hours |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase, cost in report["costs"]["by_phase"].items():
        gpu = "NA" if cost["reported_gpu_hours"] is None else f"{cost['reported_gpu_hours']:.6f}"
        lines.append(f"| {_cell(phase)} | {cost['logical_input_tokens']} / {cost['logical_output_tokens']} | {cost['physical_input_tokens']} / {cost['physical_output_tokens']} | {cost['physical_model_calls']} | {cost['cache_hits']} | {gpu} |")
    total = report["costs"]["total"]
    gpu_total = "NA (no quantified intervals)" if total["reported_gpu_hours"] is None else f"{total['reported_gpu_hours']:.6f}"
    lines.extend(["", f"Reported interval GPU hours: **{gpu_total}**. This is wall time × GPU count from explicit phase intervals; model decoding seconds are not GPU hours. Coverage is limited to recorded intervals. Unquantified time intervals: {total['unquantified_time_intervals']}.", "",
                  f"Aborted operations: {total['aborted_operations']}; attempts with incomplete token telemetry: {total['unquantified_token_operations']}. Physical-token totals are a lower bound if any attempt is unquantified; phase wall time still includes handled failures.", "",
                  "## Integrity and limitations", "",
                  "All declared seed/arm pairs and all referenced policy/task combinations were required. Missing observations were not converted to failures or dropped. Input content hashes are reported in `analysis.json`; hashes alone are not proof of authentic execution.", ""])
    for warning in report["warnings"]:
        lines.append(f"- {_cell(warning)}")
    lines.extend(["", "## Implementation amendment", "",
                  "This version implements descriptive checkpoint audit differences, not the proposal's prospective Holm-adjusted support labels. A negative point estimate is labeled observed regression only; inferential support remains unassessed. The primary interval is not a global no-regression guarantee.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
