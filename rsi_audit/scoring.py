"""Trusted host comparisons, adapted to a data-only execution boundary.

Oracle reference (read and checked, not imported/executed):
https://github.com/evalplus/evalplus/blob/
e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2/evalplus/eval/__init__.py
and ``evalplus/eval/_special_oracle.py`` at that same commit.

All named special oracles are handled here. The pinned MBPP+ v0.2.0 data no
longer includes the legacy ``are_equivalent`` and ``sum_div`` tasks, but their
upstream semantics are explicit below. ``are_equivalent`` is a vacuous oracle
in upstream: any normally returned value is accepted. Do not use that legacy
task as an integrity canary.

Supported runtime: standard-library Python built-in wire values; no arbitrary
objects, generators or NumPy arrays are materialized on the host. The three
MBPP regex tasks use an explicit non-None marker. NumPy is used ONLY for the
upstream numerical comparison on these inert values. Missing backend/unsupported
reference semantics fail closed, and the caller must abort/report that status.
Resource-timeout semantics are deliberately host-controlled and not claimed to
be bit-for-bit identical to EvalPlus's per-case reference-time multiplier.

Upstream license notice for the adapted oracle behavior:
The MIT License
Copyright (c) OpenAI (https://openai.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import math

from . import wire

EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
MBPP_OUTPUT_NOT_NONE_TASKS = frozenset({"check_str", "text_match_three", "text_starta_endb"})
MBPP_OUTPUT_SET_EQ_TASKS = frozenset({
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
})
SPECIAL_ORACLES = {
    "humaneval": frozenset({"find_zero"}),
    "mbpp": MBPP_OUTPUT_NOT_NONE_TASKS | MBPP_OUTPUT_SET_EQ_TASKS |
            frozenset({"are_equivalent", "sum_div", "surface_Area", "digit_distance_nums"}),
}


class _Unsupported(Exception):
    def __init__(self, status="unsupported_oracle"):
        self.status = status


def _contains_nan(value):
    typ = type(value)
    if typ is float:
        return math.isnan(value)
    if typ is complex:
        return math.isnan(value.real) or math.isnan(value.imag)
    if typ in (list, tuple, set, frozenset):
        return any(_contains_nan(x) for x in value)
    if typ is dict:
        return any(_contains_nan(k) or _contains_nan(v) for k, v in value.items())
    return False


def _contains_marker(value):
    if value is wire.NON_NONE:
        return True
    if type(value) in (list, tuple, set, frozenset):
        return any(_contains_marker(x) for x in value)
    if type(value) is dict:
        return any(_contains_marker(k) or _contains_marker(v) for k, v in value.items())
    return False


def _is_floats(value):
    # Upstream's built-in subset: scalar float or nonempty flat float sequence.
    return (type(value) is float or
            type(value) in (list, tuple) and bool(value) and
            all(type(x) is float for x in value))


def _arguments(record):
    if type(record) is list:
        return record, {}
    if (type(record) is dict and "args" in record and set(record) <= {"args", "kwargs"}
            and type(record["args"]) is list and type(record.get("kwargs", {})) is dict
            and all(type(k) is str for k in record.get("kwargs", {}))):
        return record["args"], record.get("kwargs", {})
    raise ValueError("task input is not an argument record")


def _float_compare(actual, expected, tolerance):
    if type(actual) is not type(expected):
        return False
    if type(expected) in (list, tuple) and len(actual) != len(expected):
        return False
    try:
        import numpy as np
    except ImportError as exc:
        raise _Unsupported("unsupported_numeric_backend") from exc
    try:
        # Inputs have already been decoded to the allowlisted wire types. No
        # user __array__, __eq__, __float__, or other hooks can run here.
        return bool(np.allclose(actual, expected, rtol=1e-7, atol=tolerance))
    except (TypeError, ValueError, OverflowError):
        return False


def _matches(source, entry, arguments, actual, expected, tolerance):
    args, kwargs = _arguments(arguments)
    if source == "mbpp" and entry in MBPP_OUTPUT_NOT_NONE_TASKS:
        # Canonical execution returns its actual result (or non-None marker),
        # while upstream trusted_exec first reduces it to `is not None`.
        expected_exists = expected is not None
        return ((actual == expected_exists) if type(actual) is bool
                else (actual is not None) == expected_exists)
    if source == "mbpp" and entry == "are_equivalent":
        return True  # Explicit upstream legacy oracle; absent in MBPP+ v0.2.0.
    if _contains_marker(expected) or _contains_marker(actual):
        raise _Unsupported()
    # There is no meaningful finite-reference comparison to NaN. Do not
    # silently count all solutions wrong if the canonical output is NaN.
    if _contains_nan(expected):
        raise _Unsupported()
    if _contains_nan(actual):
        return False
    exact = actual == expected
    if source == "mbpp":
        if entry == "sum_div":
            exact = exact or actual == 0
        elif entry in MBPP_OUTPUT_SET_EQ_TASKS:
            try:
                exact = set(actual) == set(expected)
            except (TypeError, ValueError):
                return False
        elif entry == "surface_Area":
            if exact:
                return True
            if kwargs or len(args) != 2:
                raise ValueError("surface_Area oracle requires two positional inputs")
            try:
                base, height = args
                slant = math.sqrt((base / 2) ** 2 + height ** 2)
                alternative = round(base ** 2 + 4 * (base * slant) / 2)
                exact = exact or abs(actual - alternative) <= tolerance
            except (ValueError, TypeError, OverflowError):
                return False
        elif entry == "digit_distance_nums":
            if exact:
                return True
            if kwargs or len(args) != 2:
                raise ValueError("digit_distance_nums oracle requires two positional inputs")
            try:
                left, right = str(args[0]), str(args[1])
                width = max(len(left), len(right))
                alternative = sum(abs(int(a) - int(b)) for a, b in
                                  zip(left.zfill(width), right.zfill(width)))
                exact = exact or actual == alternative
            except (ValueError, TypeError, OverflowError):
                return False
    if source == "humaneval" and entry == "find_zero":
        if kwargs or len(args) != 1 or type(args[0]) not in (list, tuple):
            raise ValueError("find_zero oracle requires a coefficient sequence")
        try:
            # Mirrors upstream sum(coeff * math.pow(x, i)), including its
            # absolute-root residual semantics; roots need not match the
            # particular root returned by the canonical implementation.
            residual = sum(coeff * math.pow(actual, i) for i, coeff in enumerate(args[0]))
            return abs(residual) <= tolerance
        except (ValueError, TypeError, OverflowError):
            return False
    if exact:
        return True
    return _float_compare(actual, expected, tolerance) if tolerance != 0 else False


def score_outputs(task: dict, execution: dict) -> dict:
    """Compare inert worker outputs to canonical outputs in the trusted host.

    ``task.expected`` must contain wire-encoded canonical outputs. The trusted
    caller must obtain these in a separate canonical-code execution, never from
    candidate self-reports. Missing/invalid reference data is a setup error.
    ``tests_run`` counts valid returned output records, not claimed tests.

    Unsupported statuses are evaluator limitations, not model failures; callers
    must stop instead of silently dropping such tasks from a benchmark.
    """
    if type(task) is not dict or type(execution) is not dict:
        raise ValueError("task and execution must be dictionaries")
    required = {"task_id", "source", "entry_point", "atol", "inputs", "expected"}
    if not required <= set(task):
        raise ValueError("missing task scoring metadata")
    if type(task["task_id"]) is not str or type(task["entry_point"]) is not str:
        raise ValueError("invalid task identity")
    source = task["source"]
    if type(source) is not str:
        raise ValueError("source must be a string")
    if source not in ("generic", "humaneval", "mbpp"):
        return {"passed": False, "status": "unsupported_oracle", "tests_run": 0}
    inputs, references = task["inputs"], task["expected"]
    if (type(inputs) is not list or not inputs or type(references) is not list or
            len(references) != len(inputs)):
        raise ValueError("canonical expected count must match nonempty inputs")
    tolerance = task["atol"]
    if (type(tolerance) not in (int, float) or not math.isfinite(tolerance) or
            tolerance < 0):
        raise ValueError("atol must be finite and nonnegative")
    for record in inputs:
        _arguments(record)
    try:
        expected = [wire.decode(x, max_nodes=1_000_000, max_bytes=16 * 1024 * 1024)
                    for x in references]
    except wire.WireError as exc:
        raise ValueError("invalid canonical wire output") from exc
    status = execution.get("status")
    outputs = execution.get("outputs")
    if type(outputs) is not list or len(outputs) > len(inputs):
        return {"passed": False, "status": "protocol_error", "tests_run": 0}
    try:
        actual = [wire.decode(x, max_nodes=1_000_000, max_bytes=16 * 1024 * 1024)
                  for x in outputs]
    except wire.WireError:
        return {"passed": False, "status": "protocol_error", "tests_run": 0}
    if status != "ok":
        known = status if status in ("timeout", "candidate_error", "protocol_error") else "protocol_error"
        return {"passed": False, "status": known, "tests_run": len(actual)}
    if len(actual) != len(inputs):
        return {"passed": False, "status": "protocol_error", "tests_run": len(actual)}
    passed = True
    for args, value, reference in zip(inputs, actual, expected):
        # Upstream changes its atol variable in-place after seeing a float
        # reference. Keep that behavior for subsequent cases in the same task.
        this_tolerance = tolerance
        root_oracle = source == "humaneval" and task["entry_point"] == "find_zero"
        if not root_oracle and tolerance == 0 and _is_floats(reference):
            tolerance = 1e-6
        try:
            # Special oracles are evaluated before the implicit-float atol
            # change in upstream; ordinary comparisons use the changed value.
            is_special = task["entry_point"] in SPECIAL_ORACLES.get(source, ())
            matched = _matches(source, task["entry_point"], args, value, reference,
                               this_tolerance if is_special else tolerance)
            # A special oracle can fall through to allclose. Its only float
            # alternative (surface_Area) has integer canonical outputs in the
            # pinned dataset; otherwise use the ordinary tolerance fallthrough.
            if not matched and is_special and tolerance != this_tolerance and task["entry_point"] != "find_zero":
                matched = _float_compare(value, reference, tolerance)
            passed = passed and matched
        except _Unsupported as exc:
            return {"passed": False, "status": exc.status, "tests_run": len(actual)}
    return {"passed": passed, "status": "pass" if passed else "wrong_answer",
            "tests_run": len(actual)}
