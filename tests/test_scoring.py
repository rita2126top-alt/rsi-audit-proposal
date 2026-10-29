"""Actual CPU comparisons, including each pinned EvalPlus special oracle."""

import unittest
from unittest import mock

from rsi_audit import wire
from rsi_audit.scoring import (score_outputs, MBPP_OUTPUT_NOT_NONE_TASKS,
                               MBPP_OUTPUT_SET_EQ_TASKS)


def check(expected, actual, *, source="generic", entry="f", inputs=None, atol=0):
    if inputs is None:
        inputs = [[] for _ in expected]
    task = {"task_id": "test", "source": source, "entry_point": entry, "atol": atol,
            "inputs": inputs, "expected": [wire.encode(x) for x in expected]}
    execution = {"status": "ok", "outputs": [wire.encode(x) for x in actual],
                 "wall_seconds": 0.01, "cpu_seconds": 0.01}
    return score_outputs(task, execution)


class ScoringTests(unittest.TestCase):
    def test_exact_and_wrong_answers(self):
        self.assertTrue(check([3, {"x": (1, 2)}], [3, {"x": (1, 2)}])["passed"])
        result = check([3, 4], [3, 5])
        self.assertEqual(result, {"passed": False, "status": "wrong_answer", "tests_run": 2})
        self.assertFalse(check([(1, 2)], [[1, 2]])["passed"])
        # Preserve upstream Python exact equality, which permits 1 == True.
        self.assertTrue(check([1], [True])["passed"])

    def test_float_tolerance_rtol_and_strict_type(self):
        self.assertTrue(check([1.0], [1.0000005])["passed"])
        self.assertFalse(check([1.0], [1.0001])["passed"])
        self.assertTrue(check([[1.0, 2.0]], [[1.0000005, 2.0]])["passed"])
        self.assertFalse(check([[1.0, 2.0]], [(1.0000005, 2.0)])["passed"])
        self.assertTrue(check([1e8], [1e8 + 5], atol=1e-8)["passed"])
        self.assertFalse(check([1.2], [1], atol=1.0)["passed"])

    def test_numeric_broadcasting_matches_numpy_rule(self):
        self.assertTrue(check([[[1.0], [1.0]]], [[[1.0, 1.0], [1.0, 1.0]]], atol=1e-6)["passed"])
        self.assertFalse(check([[[1.0], [2.0]]], [[[1.0, 2.0], [1.0, 2.0]]], atol=1e-6)["passed"])

    def test_implicit_float_tolerance_persists_across_cases(self):
        result = check([1.0, 2], [1.0000005, 2])
        self.assertTrue(result["passed"])

    def test_all_set_equivalence_oracles(self):
        for entry in MBPP_OUTPUT_SET_EQ_TASKS:
            with self.subTest(entry=entry):
                self.assertTrue(check([[1, 2, 2]], [[2, 1]], source="mbpp", entry=entry)["passed"])
                self.assertFalse(check([[1, 2]], [[1, 3]], source="mbpp", entry=entry)["passed"])
                self.assertFalse(check([[1, 2]], [[[1], 2]], source="mbpp", entry=entry)["passed"])

    def test_all_non_none_oracles_and_bool_exception(self):
        for entry in MBPP_OUTPUT_NOT_NONE_TASKS:
            with self.subTest(entry=entry):
                for candidate in (True, "anything", 0, wire.NON_NONE):
                    self.assertTrue(check([wire.NON_NONE], [candidate], source="mbpp", entry=entry)["passed"])
                self.assertFalse(check([wire.NON_NONE], [False], source="mbpp", entry=entry)["passed"])
                self.assertFalse(check([wire.NON_NONE], [None], source="mbpp", entry=entry)["passed"])
                self.assertTrue(check([None], [False], source="mbpp", entry=entry)["passed"])
                self.assertTrue(check([None], [None], source="mbpp", entry=entry)["passed"])
                self.assertFalse(check([None], [0], source="mbpp", entry=entry)["passed"])

    def test_root_oracle_accepts_alternate_root(self):
        self.assertTrue(check([1.0], [-1.0], source="humaneval", entry="find_zero",
                              inputs=[[[-1, 0, 1]]], atol=1e-6)["passed"])
        self.assertFalse(check([1.0], [2.0], source="humaneval", entry="find_zero",
                               inputs=[[[-1, 0, 1]]], atol=1e-6)["passed"])
        # Upstream continues before its generic implicit-float tolerance.
        self.assertFalse(check([0.0, 0.0], [0.0, 1e-8], source="humaneval", entry="find_zero",
                               inputs=[[[0, 1]], [[0, 1]]], atol=0)["passed"])

    def test_surface_area_alternative(self):
        # Perpendicular-height interpretation: round(9 + 6*sqrt(18.25)) = 35.
        self.assertTrue(check([33], [35], source="mbpp", entry="surface_Area",
                              inputs=[[3, 4]])["passed"])
        self.assertFalse(check([33], [34], source="mbpp", entry="surface_Area",
                               inputs=[[3, 4]])["passed"])

    def test_digit_distance_zero_padding(self):
        self.assertTrue(check([99], [3], source="mbpp", entry="digit_distance_nums",
                              inputs=[[1, 22]])["passed"])
        self.assertFalse(check([99], [4], source="mbpp", entry="digit_distance_nums",
                               inputs=[[1, 22]])["passed"])
        # Upstream short-circuits on an exact match before attempting its
        # alternative interpretation (which would fail on minus signs).
        self.assertTrue(check([99], [99], source="mbpp", entry="digit_distance_nums",
                              inputs=[[-1, 22]])["passed"])

    def test_legacy_oracles_are_explicit(self):
        self.assertTrue(check([99], [0], source="mbpp", entry="sum_div")["passed"])
        self.assertFalse(check([99], [1], source="mbpp", entry="sum_div")["passed"])
        # Upstream legacy are_equivalent is vacuous; this is documented and
        # absent from the pinned v0.2.0 data, not presented as a useful test.
        self.assertTrue(check([True], [False], source="mbpp", entry="are_equivalent")["passed"])
        self.assertTrue(check([True], [float("nan")], source="mbpp", entry="are_equivalent")["passed"])

    def test_nonfinite_and_unsupported_fail_closed(self):
        self.assertFalse(check([1.0], [float("nan")])["passed"])
        self.assertTrue(check([float("inf")], [float("inf")])["passed"])
        self.assertEqual(check([float("nan")], [float("nan")])["status"], "unsupported_oracle")
        self.assertEqual(check([wire.NON_NONE], [wire.NON_NONE])["status"], "unsupported_oracle")
        self.assertEqual(check([1], [1], source="unknown")["status"], "unsupported_oracle")

    def test_missing_numpy_does_not_silently_change_comparison(self):
        with mock.patch.dict("sys.modules", {"numpy": None}):
            self.assertEqual(check([1.0], [1.0000005])["status"], "unsupported_numeric_backend")

    def test_self_reported_pass_cannot_change_score(self):
        task = {"task_id": "test", "source": "generic", "entry_point": "f", "atol": 0,
                "inputs": [[]], "expected": [wire.encode(7)]}
        for execution in (
            {"passed": True, "status": "pass", "outputs": [wire.encode(7)]},
            {"passed": True, "status": "ok", "outputs": [wire.encode(6)]},
            {"passed": True, "status": "ok", "outputs": []},
            {"passed": True, "status": "ok", "outputs": [{"t": "pickle", "v": "x"}]},
            {"passed": True, "status": "timeout", "outputs": [wire.encode(7)]},
        ):
            self.assertFalse(score_outputs(task, execution)["passed"])

    def test_bad_reference_is_setup_error(self):
        task = {"task_id": "test", "source": "generic", "entry_point": "f", "atol": 0,
                "inputs": [[]], "expected": []}
        with self.assertRaises(ValueError):
            score_outputs(task, {"status": "ok", "outputs": []})
        task["expected"] = [wire.encode(1)]
        task["atol"] = float("nan")
        with self.assertRaises(ValueError):
            score_outputs(task, {"status": "ok", "outputs": [wire.encode(1)]})


if __name__ == "__main__":
    unittest.main()
