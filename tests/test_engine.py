"""End-to-end integrity tests using forty explicitly synthetic tasks only.

No GPU, model download, Docker worker, or candidate execution is used. A fixed
source fingerprint isolates these unit tests from concurrent checkout edits;
a separate test verifies that a changed fingerprint is rejected.
"""

import copy
import json
import tempfile
from types import SimpleNamespace
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from rsi_audit.analysis import AnalysisError, analyze_run, write_report
from rsi_audit.common import BudgetExceeded, IntegrityError, read_json
from rsi_audit.data import partition_tasks, write_bundle
from rsi_audit.engine import Engine, profile, validate_config, verify_run
from rsi_audit.ledger import Ledger
from rsi_audit.policy import BASELINE
from rsi_audit.solver import SyntheticAdapter
from rsi_audit.wire import encode


class InterruptibleSynthetic(SyntheticAdapter):
    """The same synthetic computation, with a simulated one-time interruption."""

    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.completed = Counter()
        self.completed_count = 0

    def solve(self, policy, task, *, remaining_seconds=None):
        if self.fail_after is not None and self.completed_count >= self.fail_after:
            self.fail_after = None
            raise BudgetExceeded("Simulated interruption before the next synthetic operation")
        result = super().solve(policy, task, remaining_seconds=remaining_seconds)
        self.completed[policy.sha256, task["task_id"]] += 1
        self.completed_count += 1
        return result


class TiedSynthetic(SyntheticAdapter):
    identity = {"model": "tied-synthetic-fixture", "executor": "none", "synthetic": True}

    def solve(self, policy, task, *, remaining_seconds=None):
        result = super().solve(policy, task, remaining_seconds=remaining_seconds)
        result["passed"] = int(task["task_id"].split("/")[-1]) % 2 == 0
        result["status"] = "synthetic_pass" if result["passed"] else "synthetic_fail"
        return result


class InvalidJSONSynthetic(SyntheticAdapter):
    identity = {"model": "invalid-json-synthetic-fixture", "executor": "none", "synthetic": True}

    def propose(self, parent, failures, *, seed, max_seconds=None):
        return {"text": "[" * 2000 + "0" + "]" * 2000, "input_tokens": 0,
                "output_tokens": 0, "model_calls": 0, "model_seconds": 0.0}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_patch = mock.patch("rsi_audit.engine.source_fingerprint", return_value="f" * 64)
        cls.source_patch.start()
        cls.addClassCleanup(cls.source_patch.stop)
        cls.original_config = read_json(Path(__file__).resolve().parents[1] / "configs" / "study.json")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.case_dir = Path(self.directory.name)
        self.config = profile(copy.deepcopy(self.original_config), "demo")
        tasks = [{"task_id": f"synthetic/{index}", "source": "synthetic", "family_id": f"synthetic-family/{index}",
                  "prompt": f"SYNTHETIC FIXTURE ONLY, specification {index}; no benchmark claim.",
                  "entry_point": "f", "inputs": [[index]], "expected": [encode(index)],
                  "atol": 0.0, "public_examples": []} for index in range(40)]
        partitions = partition_tasks(tasks, 12, 3, 4, 16, 73)
        self.bundle = self.case_dir / "synthetic_bundle"
        write_bundle(partitions, self.bundle, {"purpose": "synthetic-demo", "synthetic": True})
        self.initial, self.searched, self.frozen, self.audited = "initial", "searched", "frozen", "audited"

    def clone(self, template, name="run"):
        """Build an independent real lifecycle; never copy mutable run trees.

        Earlier shared on-disk templates could acquire a stale command marker
        during destructive-fixture tests in the workspace filesystem. Fresh
        runs avoid inheriting any prior test's ledger, key, cache, or marker.
        They still use the production lifecycle and all its integrity checks.
        """
        if template not in {self.initial, self.searched, self.frozen, self.audited}:
            raise ValueError("Unknown fixture phase")
        path = self.case_dir / name
        engine = Engine.create(path, copy.deepcopy(self.config), self.bundle, SyntheticAdapter())
        if template != self.initial:
            engine.search()
        if template in {self.frozen, self.audited}:
            engine.freeze()
        if template == self.audited:
            engine.audit()
        return path

    def test_completed_search_cli_does_not_load_a_model_again(self):
        from rsi_audit.cli import search
        path = self.clone(self.searched)
        args = SimpleNamespace(run=path, config=Path(__file__).resolve().parents[1] / "configs/study.json", mode="demo")
        with mock.patch("rsi_audit.cli.real_adapter", side_effect=AssertionError("Unnecessary GPU load")):
            result = search(args)
        self.assertEqual(result["phase"], "searched")
        self.assertFalse(result["model_loaded"])

    def test_formal_analysis_cli_rejects_unfrozen_resampling_settings(self):
        from rsi_audit.cli import analyze
        args = SimpleNamespace(run=self.case_dir, bootstrap_samples=100, bootstrap_seed=20261028)
        registered = {"engine": {"analysis": {"bootstrap_samples": 2000, "bootstrap_seed": 20261028}}}
        with mock.patch("rsi_audit.cli.verify_run", return_value={"mode": "formal"}), \
             mock.patch("rsi_audit.cli.read_json", return_value=registered), \
             mock.patch("rsi_audit.cli.analyze_run", side_effect=AssertionError("Must not analyze changed settings")):
            with self.assertRaisesRegex(IntegrityError, "frozen before audit"):
                analyze(args)

    def test_complete_synthetic_pipeline_and_analysis(self):
        path = self.clone(self.audited)
        status = verify_run(path)
        self.assertEqual(status["phase"], "audited")
        self.assertEqual(status["mode"], "demo")
        state = read_json(path / "state.json")
        self.assertEqual(len(state["runs"]), 9)
        expected_rows = 0
        for run in state["runs"]:
            policies = {run["initial_policy"], run["final_policy"]}
            for transition in run["rounds"]:
                policies.update((transition["parent_policy"], transition["candidate_policy"]))
            expected_rows += len(policies) * 16
        actual_rows = (path / "audit.jsonl").read_text().splitlines()
        self.assertEqual(len(actual_rows), expected_rows)
        report = analyze_run(path, bootstrap_samples=100, bootstrap_seed=41)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["data"]["tasks"], 16)
        self.assertEqual(report["costs"]["total"]["physical_model_calls"], 0)
        self.assertEqual(report["costs"]["total"]["reported_gpu_hours"], 0)
        self.assertGreater(report["costs"]["total"]["cache_hits"], 0)
        write_report(report, path / "report")
        self.assertIn("SYNTHETIC DEMO", (path / "report" / "report.md").read_text())

    def test_search_never_opens_audit_or_commitment_salt(self):
        path = self.clone(self.initial)
        original_open = Path.open
        forbidden = {self.bundle / "audit.json", self.bundle / "secret.private.json"}
        def guarded_open(target, *args, **kwargs):
            if target in forbidden:
                raise AssertionError(f"Search tried to read forbidden partition: {target.name}")
            return original_open(target, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded_open), \
                mock.patch("rsi_audit.engine.verify_commitment", side_effect=AssertionError("Full commitment verification reads audit")):
            engine = Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
            engine.search()
        self.assertEqual(verify_run(path)["phase"], "searched")

    def test_audit_requires_freeze_and_freeze_requires_complete_search(self):
        initial = self.clone(self.initial, "initial")
        with self.assertRaises(IntegrityError):
            Engine(initial).freeze()
        for template, name in ((self.initial, "before-search"), (self.searched, "before-freeze")):
            with self.subTest(stage=name):
                engine = Engine(self.clone(template, name), adapter=SyntheticAdapter(), bundle_dir=self.bundle)
                with mock.patch("rsi_audit.engine.verify_commitment", side_effect=AssertionError("Must not open audit")), \
                        self.assertRaises(IntegrityError):
                    engine.audit()

    def test_freeze_locks_search_and_completed_audit_cannot_rerun(self):
        for template, name in ((self.frozen, "frozen"), (self.audited, "audited")):
            engine = Engine(self.clone(template, name), adapter=SyntheticAdapter(), bundle_dir=self.bundle)
            with self.subTest(stage=name), self.assertRaisesRegex(IntegrityError, "locked"):
                engine.search()
        with self.assertRaises(IntegrityError):
            Engine(self.clone(self.audited, "repeat-audit"), adapter=SyntheticAdapter(), bundle_dir=self.bundle).audit()

    def test_state_and_complete_ledger_tail_rollback_are_detected(self):
        state_path = self.clone(self.searched, "state-tamper")
        state = read_json(state_path / "state.json")
        state["runs"][0]["final_policy"] = "0" * 64
        (state_path / "state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(IntegrityError, "State differs"):
            verify_run(state_path)
        ledger_path = self.clone(self.searched, "ledger-truncated")
        log = ledger_path / "ledger.jsonl"
        log.write_bytes(b"".join(log.read_bytes().splitlines(keepends=True)[:-1]))
        with self.assertRaises(IntegrityError):
            verify_run(ledger_path)

    def test_cached_result_tampering_is_rejected_before_reuse(self):
        path = self.clone(self.searched)
        engine = Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
        event = next(event for event in engine.ledger.events("solve") if not event["cache_hit"])
        cache_file = path / "cache" / (event["cache_key"] + ".json")
        result = read_json(cache_file)
        result["passed"] = not result["passed"]
        cache_file.write_text(json.dumps(result))
        with self.assertRaisesRegex(IntegrityError, "cached result"):
            engine._read_result(event)

    def test_frozen_policy_modification_and_deletion_are_detected(self):
        for change in ("modify", "delete", "format"):
            path = self.clone(self.frozen, change)
            policy_file = path / "policies" / (BASELINE.sha256 + ".json")
            if change == "delete":
                policy_file.unlink()
            elif change == "format":
                policy_file.write_text(json.dumps(read_json(policy_file), indent=4))
            else:
                value = read_json(policy_file)
                value["strategy_prompt"] = "Modified after freeze"
                policy_file.write_text(json.dumps(value))
            with self.subTest(change=change), self.assertRaises(IntegrityError):
                verify_run(path)

    def test_handled_search_interruption_reuses_completed_work(self):
        path = self.clone(self.initial)
        adapter = InterruptibleSynthetic(fail_after=15)
        with self.assertRaises(BudgetExceeded):
            Engine(path, adapter=adapter, bundle_dir=self.bundle).search()
        self.assertEqual(verify_run(path)["phase"], "searching")
        self.assertEqual(read_json(path / "command.inprogress")["status"], "completed")
        completed_before = set(adapter.completed)
        self.assertTrue(completed_before)
        Engine(path, adapter=adapter, bundle_dir=self.bundle).search()
        self.assertEqual(verify_run(path)["phase"], "searched")
        self.assertTrue(all(adapter.completed[key] == 1 for key in completed_before))
        self.assertTrue(all(count == 1 for count in adapter.completed.values()))
        requests = [event["request_key"] for event in Ledger(path).events() if event["kind"] in ("solve", "proposal")]
        self.assertEqual(len(requests), len(set(requests)))

    def test_exact_request_resume_is_idempotent_and_new_context_is_cached(self):
        path = self.clone(self.initial)
        context = {"phase": "search", "arm": "fresh", "seed": 11, "round": 1,
                   "policy_hash": BASELINE.sha256, "task_id": "unit-fixture"}
        content = {"unit_fixture": "same computation"}
        result = {"input_tokens": 7, "output_tokens": 3, "model_calls": 1,
                  "model_seconds": 0.1, "execution_seconds": 0.0}
        perform = mock.Mock(return_value=result)
        engine = Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
        with engine._command("search"):
            engine._operation("solve", context, content, perform)
        resumed = Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
        with resumed._command("search"):
            resumed._operation("solve", context, content, perform)
            resumed._operation("solve", dict(context, round=2), content, perform)
        self.assertEqual(perform.call_count, 1)
        events = Ledger(path).events("solve")
        self.assertEqual(len(events), 2)
        self.assertEqual(sum(event["output_tokens"] for event in events if not event["cache_hit"]), 3)
        self.assertEqual(sum(event["output_tokens"] for event in events), 6)

    def test_incomplete_audit_is_not_analyzed_and_can_resume_cached_tasks(self):
        path = self.clone(self.frozen)
        adapter = InterruptibleSynthetic(fail_after=3)
        with self.assertRaises(BudgetExceeded):
            Engine(path, adapter=adapter, bundle_dir=self.bundle).audit()
        self.assertEqual(verify_run(path)["phase"], "frozen")
        self.assertFalse((path / "audit.jsonl").exists())
        with self.assertRaisesRegex(AnalysisError, "phase='audited'"):
            analyze_run(path, 20)
        completed_before = set(adapter.completed)
        Engine(path, adapter=adapter, bundle_dir=self.bundle).audit()
        self.assertEqual(verify_run(path)["phase"], "audited")
        self.assertTrue(all(adapter.completed[key] == 1 for key in completed_before))
        analyze_run(path, 20)

    def test_audit_row_deletion_is_detected_after_completion(self):
        path = self.clone(self.audited)
        audit = path / "audit.jsonl"
        audit.write_bytes(b"".join(audit.read_bytes().splitlines(keepends=True)[:-1]))
        with self.assertRaises(IntegrityError):
            verify_run(path)
        with self.assertRaises(AnalysisError):
            analyze_run(path, 20)

    def test_terminal_keeps_current_policy_on_strict_dev_tie(self):
        path = self.case_dir / "tie-run"
        adapter = TiedSynthetic()
        engine = Engine.create(path, self.config, self.bundle, adapter)
        engine.search()
        terminal_runs = [run for run in engine.state["runs"] if run["arm"] == "terminal"]
        for run in terminal_runs:
            self.assertEqual(run["current_policy"], BASELINE.sha256)
            self.assertEqual(run["final_policy"], BASELINE.sha256)
            self.assertTrue(all(not row["accepted"] for row in run["rounds"]))
            self.assertFalse(run["rounds"][-1]["terminal_dev_update"]["accepted"])
            self.assertEqual(run["rounds"][-1]["terminal_dev_update"]["dev_gain"], 0)

    def test_malformed_deep_json_is_invalid_proposal_not_failed_search(self):
        path = self.case_dir / "invalid-run"
        engine = Engine.create(path, self.config, self.bundle, InvalidJSONSynthetic())
        engine.search()
        self.assertEqual(verify_run(path)["phase"], "searched")
        for run in engine.state["runs"]:
            for transition in run["rounds"]:
                self.assertTrue(transition["all_invalid"])
                self.assertEqual(len(transition["invalid_candidates"]), 2)
            self.assertEqual(run["final_policy"], BASELINE.sha256)

    def test_crash_marker_and_changed_source_fail_closed(self):
        path = self.clone(self.initial)
        (path / "command.inprogress").write_text('{"phase":"search"}')
        with self.assertRaisesRegex(IntegrityError, "Interrupted"):
            Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
        other = self.clone(self.initial, "changed-source")
        with mock.patch("rsi_audit.engine.source_fingerprint", return_value="0" * 64), \
                self.assertRaisesRegex(IntegrityError, "Implementation changed"):
            verify_run(other)

    def test_completion_journal_requires_latest_authenticated_record(self):
        path = self.clone(self.initial)
        engine = Engine(path, adapter=SyntheticAdapter(), bundle_dir=self.bundle)
        with engine._command("search"):
            pass
        journal = path / "command.inprogress"
        completed = read_json(journal)
        self.assertEqual(completed["status"], "completed")
        verify_run(path)
        changed = dict(completed, record_hash="0" * 64)
        journal.write_text(json.dumps(changed))
        with self.assertRaisesRegex(IntegrityError, "stale or altered"):
            verify_run(path)
        journal.write_text(json.dumps(completed))
        Ledger(path).append({"kind": "command_start", "phase": "search", "state_anchor": completed["state_anchor"]})
        with self.assertRaisesRegex(IntegrityError, "stale or altered"):
            verify_run(path)
        journal.unlink()
        with self.assertRaisesRegex(IntegrityError, "authentication is missing"):
            verify_run(path)

    def test_active_command_cannot_be_laundered_as_completed(self):
        path = self.clone(self.initial)
        state = read_json(path / "state.json")
        ledger = Ledger(path)
        ledger.append({"kind": "command_start", "phase": "search", "state_anchor": state["ledger_anchor"]})
        (path / "command.inprogress").write_text(json.dumps({
            "status": "completed", "phase": "search", "record_count": len(ledger.records),
            "record_hash": ledger.head, "state_anchor": state["ledger_anchor"]}))
        with self.assertRaisesRegex(IntegrityError, "lacks authenticated"):
            verify_run(path)

    def test_reduced_profiles_materialize_their_true_gate_margin(self):
        for mode in ("demo", "pilot"):
            config = profile(self.original_config, mode)
            gate = config["gate"]
            with self.subTest(mode=mode):
                self.assertEqual(gate["success_rate_margin"], gate["reuse_fresh_net_successes_required"] / gate["panel_tasks"])
                self.assertEqual(gate["success_rate_margin"], gate["terminal_net_successes_required"] / gate["terminal_tasks"])
                self.assertEqual(gate["paired_comparisons_per_arm_seed"], gate["panel_tasks"] * config["search"]["rounds"])


if __name__ == "__main__":
    unittest.main()
