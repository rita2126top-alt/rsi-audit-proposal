"""Trusted study coordinator: fixed comparisons, immutable checkpoints and audit gate.

Candidates never receive this directory. This is local integrity evidence, not a
remote attestation system and not protection against the machine administrator.
"""
from __future__ import annotations

import copy
import fcntl
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from .common import BudgetExceeded, IntegrityError, atomic_json, canonical, digest, file_digest, read_json, source_fingerprint
from .data import read_partition, verify_commitment
from .ledger import Ledger
from .policy import BASELINE, Policy
from .wire import encode


def profile(config: dict, mode: str) -> dict:
    """Materialize every reduced-study override; never relabel a pilot as formal."""
    result = copy.deepcopy(config)
    if mode not in {"demo", "pilot", "formal"}:
        raise ValueError("mode must be demo, pilot or formal")
    result["engine"] = {
        "implementation_version": "0.2.0", "mode": mode,
        "cache_rule": "exact_policy_task_model_solver_identity",
        "repair_order_semantics": "order_of_diagnostic_blocks_in_repair_prompt",
        "inference": {"torch": "2.6.0", "transformers": "4.51.3", "dtype": "bfloat16"},
        "limits": {"timeout_seconds": 30, "memory_mb": 512, "cpus": 1.0},
        "regression_labels": "descriptive_only_no_Holm_certification",
        "analysis": {"bootstrap_samples": 2000, "bootstrap_seed": 20261028},
        "phase_gpu_hours": {"search": 240.0, "audit": 144.0},
    }
    if mode in {"demo", "pilot"}:
        demo = mode == "demo"
        rounds, panel, dev, audit = (3, 4, 12, 16) if demo else (2, 8, 16, 8)
        result["search"].update(arms=["reuse", "fresh", "terminal"], proposer_seeds=[11, 29, 47] if demo else [11, 29], rounds=rounds)
        result["data"].update(raw_tasks=dev + rounds * panel + audit, target_dev=dev,
                              target_confirm=rounds * panel, target_audit=audit, minimum_audit=audit)
        result["gate"].update(panel_tasks=panel, paired_comparisons_per_arm_seed=rounds * panel,
                              reuse_fresh_net_successes_required=1, terminal_tasks=rounds * panel,
                              terminal_net_successes_required=rounds if demo else 2)
        result["engine"]["phase_gpu_hours"] = {"search": 0.0 if demo else 36.0, "audit": 0.0 if demo else 12.0}
    result["gate"]["success_rate_margin"] = result["gate"]["reuse_fresh_net_successes_required"] / result["gate"]["panel_tasks"]
    validate_config(result)
    return result


def validate_config(config: dict) -> None:
    search, gate, solver = (config[key] for key in ("search", "gate", "solver"))
    if set(search["arms"]) not in ({"reuse", "fresh"}, {"reuse", "fresh", "terminal"}) or len(search["arms"]) != len(set(search["arms"])):
        raise ValueError("Study arms must uniquely include reuse and fresh, optionally terminal")
    seeds = search["proposer_seeds"]
    if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for value in [search["rounds"], search["candidates_per_round"], gate["panel_tasks"],
                  gate["terminal_tasks"], gate["reuse_fresh_net_successes_required"],
                  gate["terminal_net_successes_required"], solver["max_total_input_tokens_per_task"],
                  solver["max_total_output_tokens_per_task"]]:
        if type(value) is not int or value <= 0:
            raise ValueError("Counts and token limits must be positive integers")
    if search["candidates_per_round"] != 2 or solver["max_calls_per_task"] not in (1, 2):
        raise ValueError("Implementation fixes two proposed candidates and at most two solver calls")
    if gate["terminal_tasks"] != gate["panel_tasks"] * search["rounds"]:
        raise ValueError("Terminal gate must use the union of all confirmation panels")
    if gate["paired_comparisons_per_arm_seed"] != gate["terminal_tasks"] or gate["success_rate_margin"] != gate["terminal_net_successes_required"] / gate["terminal_tasks"]:
        raise ValueError("Gate comparison counts and success-rate margins disagree")
    if solver["repair_feedback"] != "public_examples_only" or solver["temperature"] != 0:
        raise ValueError("Frozen solver uses greedy decoding and public-example repair only")
    if config["engine"]["mode"] == "formal":
        if (search["rounds"] != 6 or search["proposer_seeds"] != [11, 29, 47, 71, 97, 131]
                or gate["panel_tasks"] != 32 or gate["reuse_fresh_net_successes_required"] != 2
                or gate["terminal_net_successes_required"] != 12):
            raise ValueError("Formal run deviates from registered rounds, seeds or gate; preregister an amendment first")


def _state_body(state: dict) -> dict:
    return {key: value for key, value in state.items() if key != "ledger_anchor"}


def verify_run(run_dir: Path, *, allow_dirty: bool = False) -> dict:
    run_dir = Path(run_dir)
    marker_path = run_dir / "command.inprogress"
    marker = read_json(marker_path) if marker_path.exists() else None
    if marker is not None and not isinstance(marker, dict):
        raise IntegrityError("Malformed command journal")
    if marker is not None and marker.get("status") != "completed" and not allow_dirty:
        raise IntegrityError("Interrupted command detected. Preserve this run; no automatic crash recovery or unmetered replay is allowed")
    config, identity, state = [read_json(run_dir / name) for name in ("config.json", "identity.json", "state.json")]
    ledger = Ledger(run_dir)
    anchor = state["ledger_anchor"]
    ledger.assert_prefix(anchor["count"], anchor["head"])
    checkpoints = [record for record in ledger.records if record["event"].get("kind") == "checkpoint"]
    if not checkpoints or checkpoints[-1]["hash"] != anchor["head"]:
        raise IntegrityError("State is not anchored to the latest authenticated checkpoint")
    if checkpoints[-1]["event"]["state_sha256"] != digest(_state_body(state)):
        raise IntegrityError("State differs from its authenticated checkpoint")
    if not allow_dirty:
        if marker is None:
            if len(ledger.records) != 1 or state["phase"] != "searching" or state["runs"]:
                raise IntegrityError("Completed-command authentication is missing")
        else:
            if (marker.get("record_count") != len(ledger.records) or marker.get("record_hash") != ledger.head
                    or marker.get("state_anchor") != anchor):
                raise IntegrityError("Completed-command marker is stale or altered")
            completion = ledger.records[-1]["event"]
            if (completion.get("kind") != "command_complete" or completion.get("phase") != marker.get("phase")
                    or completion.get("state_anchor") != anchor):
                raise IntegrityError("Completed-command marker lacks authenticated ledger evidence")
            start_count = completion.get("start_count")
            if type(start_count) is not int or not 1 <= start_count < len(ledger.records):
                raise IntegrityError("Completed command has no valid start reference")
            start = ledger.records[start_count - 1]
            latest_start = next((record for record in reversed(ledger.records) if record["event"].get("kind") == "command_start"), None)
            if (start["hash"] != completion.get("start_hash") or latest_start != start
                    or start["event"].get("kind") != "command_start" or start["event"].get("phase") != marker["phase"]):
                raise IntegrityError("Command completion does not match the latest authenticated start")
    if state["config_sha256"] != digest(config) or state["identity_sha256"] != digest(identity):
        raise IntegrityError("Config or backend identity was changed")
    if identity["source_sha256"] != source_fingerprint():
        raise IntegrityError("Implementation changed after this run was initialized; use the original checkout")
    if digest(read_json(run_dir / "bundle_manifest.json")) != state["bundle_manifest_sha256"]:
        raise IntegrityError("Data commitment was changed")
    for path in (run_dir / "policies").glob("*.json"):
        if digest(read_json(path)) != path.stem:
            raise IntegrityError("Policy snapshot hash mismatch")
    frozen = None
    if state["phase"] in {"frozen", "audited"}:
        frozen = read_json(run_dir / "freeze.json")
        if digest(frozen) != state["freeze_sha256"]:
            raise IntegrityError("Freeze manifest changed")
        ledger.assert_prefix(frozen["ledger_anchor"]["count"], frozen["ledger_anchor"]["head"])
        for field in ("config_sha256", "identity_sha256", "bundle_manifest_sha256"):
            if frozen[field] != state[field]:
                raise IntegrityError("Freeze provenance mismatch")
        if frozen["runs"] != state["runs"]:
            raise IntegrityError("Search checkpoints changed after freeze")
        for policy_hash, expected_hash in frozen["policies"].items():
            path = run_dir / "policies" / f"{policy_hash}.json"
            if not path.is_file() or file_digest(path) != expected_hash:
                raise IntegrityError("Frozen policy snapshot is missing or changed")
    if state["phase"] == "audited":
        for filename, key in (("audit_manifest.json", "audit_manifest_sha256"), ("audit.jsonl", "audit_jsonl_sha256")):
            if file_digest(run_dir / filename) != state[key]:
                raise IntegrityError("Audit file changed after completion")
        manifest = read_json(run_dir / "audit_manifest.json")
        if manifest["freeze_sha256"] != state["freeze_sha256"]:
            raise IntegrityError("Audit manifest references another freeze")
    return {"phase": state["phase"], "mode": state["mode"], "ledger_records": len(ledger.records),
            "ledger_head": ledger.head, "freeze_sha256": state.get("freeze_sha256"),
            "source_sha256": identity["source_sha256"], "integrity": "verified_local_authenticated_artifacts"}


class Engine:
    def __init__(self, run_dir: Path, *, adapter=None, bundle_dir: Path | None = None):
        self.root = Path(run_dir)
        verify_run(self.root)
        self.config = read_json(self.root / "config.json")
        self.state = read_json(self.root / "state.json")
        self.identity = read_json(self.root / "identity.json")
        self.ledger = Ledger(self.root)
        self.adapter = adapter
        if adapter is not None and self.identity["adapter"] != adapter.identity:
            raise IntegrityError("Model/container/runtime identity differs from the initialized run")
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        if self.bundle_dir is not None:
            if digest(read_json(self.bundle_dir / "public_manifest.json")) != self.state["bundle_manifest_sha256"]:
                raise IntegrityError("Supplied bundle differs from the frozen data commitment")
        self.requests, self.cache = {}, {}
        for record in self.ledger.records:
            event = record["event"]
            if event.get("kind") in {"solve", "proposal"}:
                self.requests[event["request_key"]] = event
                if not event["cache_hit"]:
                    self.cache[event["cache_key"]] = event
        self._remaining = lambda: float("inf")

    @classmethod
    def create(cls, run_dir: Path, config: dict, bundle_dir: Path, adapter) -> "Engine":
        validate_config(config)
        mode = config["engine"]["mode"]
        if bool(adapter.synthetic) != (mode == "demo"):
            raise ValueError("Synthetic backend is permitted only for the explicitly labelled demo")
        root, bundle_dir = Path(run_dir), Path(bundle_dir)
        manifest = read_json(bundle_dir / "public_manifest.json")
        metadata = read_json(bundle_dir / "metadata.private.json")
        if file_digest(bundle_dir / "metadata.private.json") != manifest["files"]["metadata.private.json"]:
            raise IntegrityError("Bundle metadata hash mismatch")
        counts = manifest["counts"]
        if (counts["dev"]["tasks"] != config["data"]["target_dev"]
                or len(counts["panels"]) != config["search"]["rounds"]
                or any(x["tasks"] != config["gate"]["panel_tasks"] for x in counts["panels"])
                or counts["audit"]["tasks"] < config["data"]["minimum_audit"]):
            raise ValueError("Bundle sizes differ from this registered study profile")
        sources = set(counts["dev"]["sources"]) | set(counts["audit"]["sources"])
        for panel in counts["panels"]:
            sources.update(panel["sources"])
        if mode == "demo" and sources != {"synthetic"}:
            raise ValueError("Demo must use synthetic fixtures, never benchmark rows")
        if mode != "demo" and ("synthetic" in sources or not metadata.get("oracle_precomputed")):
            raise ValueError("Real runs require precomputed canonical oracle and real benchmark sources")
        if mode != "demo" and (metadata.get("oracle_executor_identity") != adapter.identity["executor"]
                               or metadata.get("oracle_execution_limits") != config["engine"]["limits"]):
            raise IntegrityError("Oracle preparation and solver execution must use the same frozen image and limits")
        if mode == "pilot" and metadata.get("purpose") != "pilot-only-from-parent-dev":
            raise ValueError("Pilot must be derived solely from parent development data")
        if mode == "formal" and metadata.get("purpose") != "formal-study":
            raise ValueError("Formal study requires a formal data bundle")
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        (root / "policies").mkdir(mode=0o700)
        (root / "cache").mkdir(mode=0o700)
        identity = {"adapter": adapter.identity, "source_sha256": source_fingerprint()}
        for filename, value in (("config.json", config), ("identity.json", identity), ("bundle_manifest.json", manifest)):
            atomic_json(root / filename, value, exclusive=True)
        ledger = Ledger(root, create=True)
        state = {"schema_version": 1, "mode": mode, "phase": "searching", "runs": [],
                 "config_sha256": digest(config), "identity_sha256": digest(identity),
                 "bundle_manifest_sha256": digest(manifest)}
        event = ledger.append({"kind": "checkpoint", "state_sha256": digest(state)})
        state["ledger_anchor"] = {"count": len(ledger.records), "head": event["hash"]}
        atomic_json(root / "state.json", state, exclusive=True)
        instance = cls(root, adapter=adapter, bundle_dir=bundle_dir)
        instance._save_policy(BASELINE)
        return instance

    def _checkpoint(self) -> None:
        record = self.ledger.append({"kind": "checkpoint", "state_sha256": digest(_state_body(self.state))})
        self.state["ledger_anchor"] = {"count": len(self.ledger.records), "head": record["hash"]}
        atomic_json(self.root / "state.json", self.state)

    @contextmanager
    def _command(self, phase: str, startup_seconds: float = 0.0):
        """One writer; hard crashes fail closed, handled errors retain measured costs."""
        with (self.root / "run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise IntegrityError("Another coordinator is writing this run") from exc
            verify_run(self.root)
            if read_json(self.root / "state.json") != self.state:
                raise IntegrityError("Stale coordinator; reload before writing")
            start_record = self.ledger.append({"kind": "command_start", "phase": phase, "state_anchor": self.state["ledger_anchor"]})
            start_count = len(self.ledger.records)
            atomic_json(self.root / "command.inprogress", {"status": "active", "phase": phase, "pid": os.getpid(),
                                                            "start_count": start_count, "start_hash": start_record["hash"]})
            started = time.monotonic() - startup_seconds
            gpu_count = self.adapter.gpu_count if self.adapter else 0
            consumed = sum(event["wall_seconds"] * event["gpu_count"] / 3600.0
                           for event in self.ledger.events("phase_time") if event["phase"] == phase)
            cap = self.config["engine"]["phase_gpu_hours"].get(phase)
            def remaining():
                if gpu_count == 0 or cap is None:
                    return float("inf")
                return (cap - consumed) * 3600.0 / gpu_count - (time.monotonic() - started)
            self._remaining = remaining
            try:
                yield
            finally:
                self.ledger.append({"kind": "phase_time", "phase": phase,
                                    "wall_seconds": max(0.0, time.monotonic() - started), "gpu_count": gpu_count})
                self._checkpoint()
                completion = self.ledger.append({"kind": "command_complete", "phase": phase,
                                                 "state_anchor": self.state["ledger_anchor"],
                                                 "start_count": start_count, "start_hash": start_record["hash"]})
                # A durable authenticated completion record avoids treating
                # directory-entry deletion as the sole evidence of clean exit.
                atomic_json(self.root / "command.inprogress", {
                    "status": "completed", "phase": phase, "record_count": len(self.ledger.records),
                    "record_hash": completion["hash"], "state_anchor": self.state["ledger_anchor"]})
                self._remaining = lambda: float("inf")

    def _ensure_budget(self) -> float | None:
        remaining = self._remaining()
        if remaining <= 0:
            raise BudgetExceeded("Registered phase GPU-hour allowance exhausted; no result is imputed")
        return None if remaining == float("inf") else remaining

    def _save_policy(self, policy: Policy) -> None:
        path = self.root / "policies" / f"{policy.sha256}.json"
        if path.exists():
            if digest(read_json(path)) != policy.sha256:
                raise IntegrityError("Existing policy file was modified")
        else:
            atomic_json(path, policy.to_dict(), exclusive=True)

    def _policy(self, value: str) -> Policy:
        obj = read_json(self.root / "policies" / f"{value}.json")
        if digest(obj) != value:
            raise IntegrityError("Policy hash mismatch")
        return Policy.from_dict(obj, self.adapter.count_tokens, self.config["search"]["strategy_prompt_max_tokens"])

    def _read_result(self, event: dict) -> dict:
        path = self.root / "cache" / f"{event['cache_key']}.json"
        if file_digest(path) != event["result_sha256"]:
            raise IntegrityError("Authenticated cached result was modified")
        return read_json(path)

    def _operation(self, kind: str, context: dict, key_content: dict, perform) -> dict:
        cache_key = digest({"kind": kind, "identity": self.identity, "operation": key_content})
        request_key = digest({"kind": kind, "context": context, "cache_key": cache_key})
        if request_key in self.requests:
            return self._read_result(self.requests[request_key])
        cached = self.cache.get(cache_key)
        if cached:
            result = self._read_result(cached)
            result_hash = cached["result_sha256"]
        else:
            self._ensure_budget()
            try:
                result = perform()
            except BaseException as exc:
                self.ledger.append({"kind": "operation_aborted", **context, "operation_kind": kind,
                                    "error_type": type(exc).__name__, "known_usage": getattr(exc, "usage", None),
                                    "cost_note": "Phase wall time includes this attempt; token telemetry may be incomplete"})
                raise
            path = self.root / "cache" / f"{cache_key}.json"
            atomic_json(path, result, exclusive=True)
            result_hash = file_digest(path)
        usage = {key: result.get(key, 0) for key in ("input_tokens", "output_tokens", "model_calls", "model_seconds", "execution_seconds")}
        event = dict(context, **usage, kind=kind, request_key=request_key, cache_key=cache_key,
                     cache_hit=cached is not None, result_sha256=result_hash)
        self.ledger.append(event)
        self.requests[request_key] = event
        if not cached:
            self.cache[cache_key] = event
        return result

    def _evaluate(self, policy: Policy, tasks: list[dict], context: dict) -> list[dict]:
        results = []
        for task in tasks:
            item_context = dict(context, policy_hash=policy.sha256, task_id=task["task_id"])
            result = self._operation("solve", item_context,
                {"policy": policy.to_dict(), "task_sha256": digest(encode(task)), "solver": self.config["solver"]},
                lambda task=task: self.adapter.solve(policy, task, remaining_seconds=self._remaining))
            if type(result.get("passed")) is not bool or result.get("synthetic") != (self.state["mode"] == "demo"):
                raise IntegrityError("Solver returned an invalid or incorrectly labelled outcome")
            results.append(result)
        return results

    def _nominate(self, parent: Policy, dev: list[dict], context: dict) -> tuple[Policy, int, int, dict]:
        parent_results = self._evaluate(parent, dev, dict(context, label="parent_dev"))
        parent_successes = sum(result["passed"] for result in parent_results)
        failures = [{"task_id": task["task_id"], "prompt": task["prompt"][:1200], "status": result["status"],
                     "public_feedback": result["public_feedback"][:800], "program": result["code"][:1200]}
                    for task, result in zip(dev, parent_results) if not result["passed"]][:6]
        candidates, invalid = [], []
        for index in range(2):
            # Common random seeds across arms; parent/failure differences remain causal.
            seed = int(digest([context["seed"], context["round"], index])[:8], 16)
            proposal_context = dict(context, label=f"candidate_{index}", policy_hash=parent.sha256)
            def propose():
                try:
                    return self.adapter.propose(parent, failures, seed=seed, max_seconds=self._ensure_budget())
                except BudgetExceeded as exc:
                    # A global phase stop must propagate; an oversized meta prompt is invalid.
                    self._ensure_budget()
                    return {"text": "", "input_tokens": 0, "output_tokens": 0, "model_calls": 0,
                            "model_seconds": 0.0, "error": str(exc)}
            result = self._operation("proposal", proposal_context,
                {"parent": parent.to_dict(), "failures": failures, "seed": seed, "search": self.config["search"]}, propose)
            try:
                policy = Policy.from_response(result["text"], self.adapter.count_tokens,
                                              self.config["search"]["strategy_prompt_max_tokens"])
            except (ValueError, TypeError) as exc:
                invalid.append({"candidate": index, "reason": str(exc)[:400]})
                continue
            self._save_policy(policy)
            outcomes = self._evaluate(policy, dev, dict(context, label=f"candidate_dev_{index}"))
            successes = sum(item["passed"] for item in outcomes)
            tokens = sum(item["output_tokens"] for item in outcomes)
            candidates.append((policy, successes, tokens))
        if not candidates:
            return parent, parent_successes, parent_successes, {"invalid_candidates": invalid, "all_invalid": True}
        winner, successes, _ = min(candidates, key=lambda item: (-item[1], item[2], item[0].sha256))
        return winner, successes, parent_successes, {"invalid_candidates": invalid, "all_invalid": False}

    def search(self, *, startup_seconds: float = 0.0) -> dict:
        if self.state["phase"] not in {"searching", "searched"}:
            raise IntegrityError("Search is locked after freeze")
        if self.adapter is None or self.bundle_dir is None:
            raise ValueError("Search requires backend and bundle")
        if self.state["phase"] == "searched":
            return self.state
        # No audit content or salt is opened anywhere on this path.
        dev, panels = read_partition(self.bundle_dir, "dev"), read_partition(self.bundle_dir, "panels")
        spec, gate = self.config["search"], self.config["gate"]
        with self._command("search", startup_seconds):
            for seed in spec["proposer_seeds"]:
                panel_order = list(range(len(panels)))
                random.Random(seed).shuffle(panel_order)
                for arm in spec["arms"]:
                    run = next((x for x in self.state["runs"] if x["arm"] == arm and x["seed"] == seed), None)
                    if run is None:
                        run = {"arm": arm, "seed": seed, "initial_policy": BASELINE.sha256,
                               "current_policy": BASELINE.sha256, "final_policy": BASELINE.sha256,
                               "panel_order": panel_order, "rounds": []}
                        self.state["runs"].append(run)
                    for round_number in range(len(run["rounds"]) + 1, spec["rounds"] + 1):
                        parent = self._policy(run["current_policy"])
                        context = {"phase": "search", "arm": arm, "seed": seed, "round": round_number}
                        candidate, candidate_dev, parent_dev, diagnostics = self._nominate(parent, dev, context)
                        dev_gain = (candidate_dev - parent_dev) / len(dev)
                        if arm == "terminal":
                            accepted, confirm_gain, phase = candidate_dev > parent_dev, None, "dev_update"
                        else:
                            panel_index = panel_order[0 if arm == "reuse" else round_number - 1]
                            panel = panels[panel_index]
                            left = self._evaluate(parent, panel, dict(context, label="gate_parent"))
                            right = self._evaluate(candidate, panel, dict(context, label="gate_candidate"))
                            net = sum(x["passed"] for x in right) - sum(x["passed"] for x in left)
                            accepted = net >= gate["reuse_fresh_net_successes_required"]
                            confirm_gain, phase = net / len(panel), "gate"
                        selected = candidate if accepted else parent
                        transition = {"round": round_number, "parent_policy": parent.sha256,
                                      "candidate_policy": candidate.sha256, "accepted": accepted,
                                      "confirm_gain": confirm_gain, "dev_gain": dev_gain, "phase": phase,
                                      "selected_policy": selected.sha256, **diagnostics}
                        run["current_policy"] = selected.sha256
                        if arm == "terminal" and round_number == spec["rounds"]:
                            # Last terminal record represents the only deployment gate vs B0.
                            all_confirm = [task for panel in panels for task in panel]
                            left = self._evaluate(BASELINE, all_confirm, dict(context, label="terminal_parent"))
                            right = self._evaluate(selected, all_confirm, dict(context, label="terminal_candidate"))
                            net = sum(x["passed"] for x in right) - sum(x["passed"] for x in left)
                            approved = net >= gate["terminal_net_successes_required"]
                            transition["terminal_dev_update"] = {"parent_policy": parent.sha256, "candidate_policy": candidate.sha256,
                                                                 "accepted": accepted, "selected_policy": selected.sha256, "dev_gain": dev_gain}
                            transition.update(parent_policy=BASELINE.sha256, candidate_policy=selected.sha256,
                                              accepted=approved, confirm_gain=net / len(all_confirm),
                                              dev_gain=None, phase="gate", selected_policy=selected.sha256 if approved else BASELINE.sha256)
                            run["final_policy"] = transition["selected_policy"]
                        else:
                            run["final_policy"] = selected.sha256
                        run["rounds"].append(transition)
                        self.ledger.append({"kind": "decision", **context, "transition_sha256": digest(transition)})
                        self._checkpoint()
            self.state["phase"] = "searched"
        return self.state

    def freeze(self) -> dict:
        if self.state["phase"] != "searched":
            raise IntegrityError("All arms and seeds must complete search before freeze")
        spec = self.config["search"]
        expected = {(arm, seed) for arm in spec["arms"] for seed in spec["proposer_seeds"]}
        if {(run["arm"], run["seed"]) for run in self.state["runs"]} != expected or any(len(run["rounds"]) != spec["rounds"] for run in self.state["runs"]):
            raise IntegrityError("Incomplete study cannot be frozen")
        with self._command("freeze"):
            frozen = {"schema_version": 1, "mode": self.state["mode"],
                      **{key: self.state[key] for key in ("config_sha256", "identity_sha256", "bundle_manifest_sha256")},
                      "runs": copy.deepcopy(self.state["runs"]), "ledger_anchor": copy.deepcopy(self.state["ledger_anchor"]),
                      "policies": {path.stem: file_digest(path) for path in sorted((self.root / "policies").glob("*.json"))}}
            atomic_json(self.root / "freeze.json", frozen, exclusive=True)
            self.state.update(phase="frozen", freeze_sha256=digest(frozen))
            self.ledger.append({"kind": "freeze", "freeze_sha256": digest(frozen)})
        return frozen

    def audit(self, *, startup_seconds: float = 0.0) -> dict:
        if self.state["phase"] != "frozen":
            raise IntegrityError("Audit requires completed frozen search and cannot rerun an audited study")
        if self.adapter is None or self.bundle_dir is None:
            raise ValueError("Audit requires backend and bundle")
        with self._command("audit", startup_seconds):
            verify_commitment(self.bundle_dir)
            tasks = read_partition(self.bundle_dir, "audit")
            manifest = {"tasks": [{key: task[key] for key in ("task_id", "source", "family_id")} for task in tasks],
                        "freeze_sha256": self.state["freeze_sha256"], "bundle_commitment": self.state["bundle_manifest_sha256"]}
            atomic_json(self.root / "audit_manifest.json", manifest)
            temporary = self.root / "audit.partial.jsonl"
            rows = 0
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as stream:
                for run in self.state["runs"]:
                    policies = {run["initial_policy"], run["final_policy"]}
                    for item in run["rounds"]:
                        policies.update((item["parent_policy"], item["candidate_policy"]))
                    if len(policies) > self.config["audit"]["max_checkpoints_per_arm_seed"]:
                        raise IntegrityError("Referenced checkpoints exceed registered audit ceiling")
                    for policy_hash in sorted(policies):
                        policy = self._policy(policy_hash)
                        context = {"phase": "audit", "arm": run["arm"], "seed": run["seed"], "round": 0, "label": "checkpoint_audit"}
                        results = self._evaluate(policy, tasks, context)
                        for task, result in zip(tasks, results):
                            row = {"arm": run["arm"], "seed": run["seed"], "policy_hash": policy_hash,
                                   **{key: task[key] for key in ("task_id", "source", "family_id")},
                                   "passed": result["passed"], "status": result["status"], "synthetic": result["synthetic"]}
                            stream.write(canonical(row) + b"\n")
                            rows += 1
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "audit.jsonl")
            self.state.update(phase="audited", audit_manifest_sha256=file_digest(self.root / "audit_manifest.json"),
                              audit_jsonl_sha256=file_digest(self.root / "audit.jsonl"))
            self.ledger.append({"kind": "audit_complete", "rows": rows, "audit_jsonl_sha256": self.state["audit_jsonl_sha256"]})
        return self.state
