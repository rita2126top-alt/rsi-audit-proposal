"""Hand-checkable tests of denominators, pairing, resampling, and accounting."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rsi_audit.analysis import AnalysisError, _paired_bootstrap, analyze_run, write_report


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.tasks = [
            {"task_id": "a", "source": "human", "family_id": "a"},
            {"task_id": "b", "source": "human", "family_id": "b"},
            {"task_id": "c", "source": "mbpp", "family_id": "c"},
            {"task_id": "d", "source": "mbpp", "family_id": "d"},
        ]

    def make_run(self, arms=("reuse", "fresh"), seeds=(11, 29), rounds=1, mode="demo", accepted=True):
        self.config = {"project": "RSI-Audit", "search": {"arms": list(arms), "proposer_seeds": list(seeds), "rounds": rounds}}
        self.state = {"schema_version": 1, "mode": mode, "phase": "audited", "runs": []}
        self.rows = []
        for arm in arms:
            for seed in seeds:
                parent = "base"
                transitions = []
                policies = [parent]
                for round_number in range(1, rounds + 1):
                    candidate = f"{arm}-{seed}-r{round_number}"
                    policies.append(candidate)
                    transitions.append({"round": round_number, "parent_policy": parent,
                                        "candidate_policy": candidate, "accepted": accepted,
                                        "confirm_gain": 0.0625, "dev_gain": 0.1})
                    if accepted:
                        parent = candidate
                self.state["runs"].append({"arm": arm, "seed": seed, "initial_policy": "base",
                                           "final_policy": parent, "rounds": transitions})
                for policy in set(policies):
                    for task in self.tasks:
                        self.rows.append({"arm": arm, "seed": seed, "policy_hash": policy, **task, "passed": False})
        self.ledger = []
        self.save()

    def save(self):
        (self.path / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        manifest_path = self.path / "audit_manifest.json"
        manifest_path.write_text(json.dumps({"tasks": self.tasks, "freeze_sha256": "fixture", "bundle_commitment": "fixture"}), encoding="utf-8")
        audit_path = self.path / "audit.jsonl"
        audit_path.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.state["audit_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.state["audit_jsonl_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        (self.path / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        (self.path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.ledger), encoding="utf-8")

    def set_passes(self, arm, seed, policy, passed_ids):
        for row in self.rows:
            if (row["arm"], row["seed"], row["policy_hash"]) == (arm, seed, policy):
                row["passed"] = row["task_id"] in passed_ids

    def test_paired_primary_and_initial_baseline_are_hand_computable(self):
        self.make_run()
        self.set_passes("fresh", 11, "fresh-11-r1", {"a", "b", "c"})
        self.set_passes("reuse", 11, "reuse-11-r1", {"a"})
        self.set_passes("fresh", 29, "fresh-29-r1", {"a", "b"})
        self.set_passes("reuse", 29, "reuse-29-r1", {"b"})
        self.save()
        report = analyze_run(self.path, bootstrap_samples=100, bootstrap_seed=17)
        self.assertEqual(report["primary"]["estimate"], 0.375)
        self.assertEqual(report["primary"]["estimate_percentage_points"], 37.5)
        self.assertEqual([item["difference"] for item in report["primary"]["per_seed"]], [0.5, 0.25])
        self.assertEqual(report["arms"]["fresh"]["final_solve_rate"], 0.625)
        self.assertEqual(report["arms"]["fresh"]["gain_over_initial"], 0.625)
        self.assertTrue(report["integrity"]["complete_expected_task_matrix"])
        first = next(row for row in report["checkpoints"] if row["arm"] == "fresh" and row["seed"] == 11)
        self.assertEqual(first["optimism"], 0.0625 - 0.75)
        self.assertIsNone(report["primary"]["p_value"])
        self.assertIsNone(report["primary"]["significance_claim"])

    def test_final_policy_is_used_instead_of_historical_best(self):
        self.make_run(seeds=(11,), rounds=2)
        self.set_passes("fresh", 11, "fresh-11-r1", {"a", "b", "c", "d"})
        self.set_passes("fresh", 11, "fresh-11-r2", {"a", "b"})
        self.set_passes("reuse", 11, "reuse-11-r2", {"a"})
        self.save()
        report = analyze_run(self.path, 50)
        self.assertEqual(report["primary"]["estimate"], 0.25)
        self.assertEqual(report["arms"]["fresh"]["final_solve_rate"], 0.5)
        self.assertEqual(report["promotions"]["fresh"]["observed_audit_regressions"], 1)
        second = next(row for row in report["checkpoints"] if row["arm"] == "fresh" and row["round"] == 2)
        self.assertEqual(second["audit_gain"], -0.5)
        self.assertEqual(second["inferential_support"], "not_assessed")

    def test_zero_accepted_has_undefined_negative_fraction(self):
        self.make_run(accepted=False)
        report = analyze_run(self.path, 50)
        for arm in ("fresh", "reuse"):
            self.assertEqual(report["promotions"][arm]["accepted_count"], 0)
            self.assertIsNone(report["promotions"][arm]["observed_negative_fraction"])
            self.assertIsNone(report["promotions"][arm]["mean_audit_gain_of_accepted"])

    def test_terminal_development_updates_are_not_gate_promotions(self):
        self.make_run(arms=("reuse", "fresh", "terminal"), seeds=(11,), rounds=2)
        terminal = next(run for run in self.state["runs"] if run["arm"] == "terminal")
        terminal["rounds"][0].update(phase="dev_update", confirm_gain=None)
        terminal["rounds"][1]["parent_policy"] = "base"
        self.save()
        report = analyze_run(self.path, 50)
        self.assertEqual(report["promotions"]["terminal"]["gate_decisions"], 1)
        self.assertEqual(report["promotions"]["terminal"]["accepted_count"], 1)
        update = next(row for row in report["checkpoints"] if row["arm"] == "terminal" and row["round"] == 1)
        self.assertFalse(update["counts_as_gate_promotion"])
        self.assertIsNone(update["optimism"])

    def test_missing_one_audit_row_is_rejected(self):
        self.make_run()
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Incomplete audit matrix"):
            analyze_run(self.path, 20)

    def test_task_missing_everywhere_is_not_lost_from_denominator(self):
        self.make_run()
        self.rows = [row for row in self.rows if row["task_id"] != "d"]
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Incomplete audit matrix"):
            analyze_run(self.path, 20)

    def test_duplicate_audit_key_is_rejected_even_if_identical(self):
        self.make_run()
        self.rows.append(dict(self.rows[0]))
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Duplicate audit key"):
            analyze_run(self.path, 20)

    def test_missing_seed_pair_is_rejected(self):
        self.make_run()
        self.state["runs"].pop()
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Missing run/seed"):
            analyze_run(self.path, 20)

    def test_different_task_metadata_between_arms_is_rejected(self):
        self.make_run()
        self.rows[0]["family_id"] = "not-the-declared-family"
        self.save()
        with self.assertRaisesRegex(AnalysisError, "metadata mismatch"):
            analyze_run(self.path, 20)

    def test_unknown_or_replaced_task_id_is_rejected(self):
        self.make_run()
        self.rows[0]["task_id"] = "replacement-task"
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Unexpected task"):
            analyze_run(self.path, 20)

    def test_passed_values_are_strict_booleans(self):
        self.make_run()
        self.rows[0]["passed"] = 1
        self.save()
        with self.assertRaisesRegex(AnalysisError, "must be boolean"):
            analyze_run(self.path, 20)

    def test_incomplete_phase_is_rejected_without_model_calls(self):
        self.make_run()
        self.state["phase"] = "frozen"
        self.save()
        with self.assertRaisesRegex(AnalysisError, "phase='audited'"):
            analyze_run(self.path, 20)

    def test_hash_change_is_rejected(self):
        self.make_run()
        with (self.path / "audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(AnalysisError, "SHA-256"):
            analyze_run(self.path, 20)

    def test_missing_manifest_is_rejected(self):
        self.make_run()
        (self.path / "audit_manifest.json").unlink()
        with self.assertRaisesRegex(AnalysisError, "missing audit_manifest"):
            analyze_run(self.path, 20)

    def test_source_stratification_preserves_original_task_weights(self):
        manifest = [
            {"task_id": "a", "source": "A", "family_id": "a"},
            {"task_id": "b", "source": "A", "family_id": "b"},
            {"task_id": "c", "source": "A", "family_id": "c"},
            {"task_id": "d", "source": "B", "family_id": "d"},
        ]
        draws = _paired_bootstrap([[1, 1, 1, 0]], manifest, 100, 101)
        self.assertEqual(set(draws), {0.75})

    def test_family_resampling_never_splits_correlated_tasks(self):
        manifest = [
            {"task_id": "a", "source": "A", "family_id": "large"},
            {"task_id": "b", "source": "A", "family_id": "large"},
            {"task_id": "c", "source": "A", "family_id": "large"},
            {"task_id": "d", "source": "A", "family_id": "small"},
        ]
        draws = _paired_bootstrap([[1, 1, 1, 0]], manifest, 200, 31)
        self.assertEqual(set(draws), {0, 0.75, 1})

    def test_seed_is_resampled_once_for_all_tasks(self):
        manifest = [{"task_id": str(i), "source": "A", "family_id": "one-correlated-family"} for i in range(4)]
        draws = _paired_bootstrap([[1, 1, 1, 1], [-1, -1, -1, -1]], manifest, 200, 37)
        self.assertEqual(set(draws), {-1, 0, 1})

    def test_cross_source_family_stays_a_single_cluster(self):
        manifest = [
            {"task_id": "a", "source": "A", "family_id": "positive"},
            {"task_id": "b", "source": "B", "family_id": "positive"},
            {"task_id": "c", "source": "A", "family_id": "negative"},
            {"task_id": "d", "source": "B", "family_id": "negative"},
        ]
        draws = _paired_bootstrap([[1, 1, -1, -1]], manifest, 200, 39)
        self.assertEqual(set(draws), {-1, 0, 1})

    def test_costs_include_proposer_cache_and_only_explicit_gpu_intervals(self):
        self.make_run(mode="formal")
        self.ledger = [
            {"event": {"kind": "solve", "phase": "dev", "arm": "fresh", "input_tokens": 100, "output_tokens": 20,
                       "model_calls": 2, "model_seconds": 0.5, "execution_seconds": 0.2, "cache_hit": False}},
            {"event": {"kind": "solve", "phase": "dev", "arm": "fresh", "input_tokens": 100, "output_tokens": 20,
                       "model_calls": 2, "model_seconds": 0.5, "execution_seconds": 0.2, "cache_hit": True}},
            {"event": {"kind": "proposal", "phase": "proposal", "arm": "fresh", "input_tokens": 30, "output_tokens": 10,
                       "model_calls": 1, "model_seconds": 0.1, "execution_seconds": 0, "cache_hit": False}},
            {"event": {"kind": "gate", "phase": "confirm", "arm": "fresh", "input_tokens": 999, "output_tokens": 999}},
            {"event": {"kind": "phase_time", "phase": "dev", "wall_seconds": 3600, "gpu_count": 1}},
            {"event": {"kind": "phase_time", "phase": "audit", "wall_seconds": 1800, "gpu_count": 1}},
        ]
        self.save()
        total = analyze_run(self.path, 50)["costs"]["total"]
        self.assertEqual(total["logical_input_tokens"], 230)
        self.assertEqual(total["physical_input_tokens"], 130)
        self.assertEqual(total["logical_output_tokens"], 50)
        self.assertEqual(total["physical_output_tokens"], 30)
        self.assertEqual(total["logical_model_calls"], 5)
        self.assertEqual(total["physical_model_calls"], 3)
        self.assertAlmostEqual(total["physical_model_seconds"], 0.6)
        self.assertAlmostEqual(total["physical_execution_seconds"], 0.2)
        self.assertEqual(total["reported_gpu_hours"], 1.5)
        self.assertEqual(total["cache_hits"], 1)
        self.assertEqual(total["proposal_events"], 1)

    def test_missing_gpu_telemetry_is_not_reported_as_zero(self):
        self.make_run()
        self.ledger = [{"event": {"kind": "phase_time", "phase": "audit", "wall_seconds": 1800}}]
        self.save()
        total = analyze_run(self.path, 50)["costs"]["total"]
        self.assertIsNone(total["reported_gpu_hours"])
        self.assertEqual(total["unquantified_time_intervals"], 1)

    def test_synthetic_demo_cannot_claim_gpu_execution(self):
        self.make_run()
        self.ledger = [{"event": {"kind": "phase_time", "phase": "audit", "wall_seconds": 100, "gpu_count": 1}}]
        self.save()
        with self.assertRaisesRegex(AnalysisError, "Synthetic demo cannot claim GPU"):
            analyze_run(self.path, 50)

    def test_output_reports_are_explicitly_synthetic_and_reproducible(self):
        self.make_run()
        first = analyze_run(self.path, 100, 42)
        second = analyze_run(self.path, 100, 42)
        self.assertEqual(first, second)
        output = self.path / "reports"
        write_report(first, output)
        markdown = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC DEMO", markdown)
        self.assertIn("no GPU experiment", markdown)
        self.assertIn("no real benchmark-performance claim", markdown)
        self.assertIn("Holm-adjusted", markdown)
        self.assertTrue(json.loads((output / "analysis.json").read_text())["synthetic"])

    def test_aborted_attempt_costs_are_not_silently_lost_or_imputed(self):
        self.make_run()
        self.ledger = [
            {"kind": "operation_aborted", "phase": "search", "arm": "fresh",
             "known_usage": {"input_tokens": 20, "output_tokens": 3, "model_calls": 1,
                             "model_seconds": 0.2, "execution_seconds": 0.0}},
            {"kind": "operation_aborted", "phase": "search", "arm": "reuse", "known_usage": None},
        ]
        self.save()
        report = analyze_run(self.path, 20, 42)
        total = report["costs"]["total"]
        self.assertEqual(total["physical_input_tokens"], 20)
        self.assertEqual(total["physical_output_tokens"], 3)
        self.assertEqual(total["logical_input_tokens"], 0)
        self.assertEqual(total["aborted_operations"], 2)
        self.assertEqual(total["unquantified_token_operations"], 1)
        self.assertIsNone(total["reported_gpu_hours"])


if __name__ == "__main__":
    unittest.main()
