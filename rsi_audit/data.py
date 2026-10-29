"""Hash-verified EvalPlus inputs and private, family-disjoint data bundles.

No reference or candidate program is executed by this module. See
docs/DATA_PREPARATION.md for the pinned upstream semantics and trust boundary.

Modified input-restoration rules adapt EvalPlus v0.3.1 (Apache-2.0).
Original project: https://github.com/evalplus/evalplus
Changes add locked loading, public-example parsing and private family bundles.
See THIRD_PARTY_NOTICES.md and licenses/EvalPlus-LICENSE.txt.
"""
from __future__ import annotations

import ast
import copy
import doctest
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
from functools import lru_cache
import unicodedata
import warnings


class DataError(ValueError):
    """The input cannot satisfy the frozen data protocol."""


EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
_VERSIONS = {"HumanEval+": "v0.1.10", "MBPP+": "v0.2.0"}
_FILES = {"dev": "dev.json", "panels": "panels.json", "audit": "audit.json"}


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"Invalid JSON: {exc}") from exc


def _ast(code: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def _literal(node):
    """Literal-only conversion, never eval/compile/exec or arbitrary calls."""
    return ast.literal_eval(node)


def _example(call, expected, entry_point):
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.func.id != entry_point or any(k.arg is None for k in call.keywords):
        return None
    try:
        args = [_literal(a) for a in call.args]
        kwargs = {k.arg: _literal(k.value) for k in call.keywords}
        value = _literal(expected)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return {"args": args, "kwargs": kwargs, "expected": value}


def extract_public_examples(prompt: str, entry_point: str) -> list[dict]:
    """Extract conservative, literal direct-call examples from the prompt only.

    Transformed comparisons such as set(f(x)) and math.isclose(f(x), y) are
    deliberately not reinterpreted as exact-output examples. The complete
    original prompt still exposes those examples to the model.
    """
    examples = []
    texts = [prompt]
    try:
        tree = _ast(prompt)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    texts.append(doc)
    except SyntaxError:
        pass
    parsed = []
    for text in texts:
        try:
            parsed.extend(doctest.DocTestParser().get_examples(text))
        except ValueError:
            continue
    for example in parsed:
        if not example.want.strip():
            continue
        try:
            call = ast.parse(example.source.strip(), mode="eval").body
            expected = ast.parse(example.want.strip(), mode="eval").body
        except SyntaxError:
            continue
        value = _example(call, expected, entry_point)
        if value is not None:
            examples.append(value)
    # Assert examples in both module docstrings (MBPP) and ordinary prompt
    # lines. Only a single equality with the entry-point call is representable.
    for line in "\n".join(texts).splitlines():
        line = line.strip()
        if line.startswith(">>> "):
            line = line[4:]
        if not line.startswith("assert "):
            continue
        try:
            statements = _ast(line).body
        except SyntaxError:
            continue
        if len(statements) != 1 or not isinstance(statements[0], ast.Assert):
            continue
        expr = statements[0].test
        if not (isinstance(expr, ast.Compare) and len(expr.ops) == 1
                and isinstance(expr.ops[0], ast.Eq)):
            continue
        value = _example(expr.left, expr.comparators[0], entry_point)
        if value is None:
            value = _example(expr.comparators[0], expr.left, entry_point)
        if value is not None:
            examples.append(value)
    # repr is used only for duplicate removal, never for deserialization.
    seen, unique = set(), []
    for example in examples:
        key = repr(example)
        if key not in seen:
            seen.add(key)
            unique.append(example)
    return unique


def _mbpp_inputs(task_id: str, inputs: list) -> list[list]:
    """Type restoration matching pinned EvalPlus data/mbpp.py.

    This is an explicit implementation of the upstream task-id conversion
    table, including its unusual empty-list -> empty-dict rule for task 115.
    No dynamic evaluation of strings or reference programs is used.
    """
    try:
        number = int(task_id.split("/")[-1])
        if number in {2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424,
                      429, 470, 560, 579, 596, 616, 630, 726, 740, 744, 809}:
            result = [[tuple(x) for x in case] for case in inputs]
        elif number in {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417,
                        438, 473, 614, 780}:
            result = [[[tuple(x) for x in arg] for arg in case] for case in inputs]
        elif number in {75, 413, 444, 753}:
            result = [[[tuple(x) for x in case[0]], case[1]] for case in inputs]
        elif number in {106, 750}:
            result = [[case[0], tuple(case[1])] for case in inputs]
        elif number == 115:
            result = [[[set(x) if isinstance(x, list) and x else {}
                        for x in case[0]]] for case in inputs]
        elif number == 124:
            result = [[float(case[0]), complex(case[1])] for case in inputs]
        elif number in {250, 405, 446, 617, 720, 763, 808}:
            result = [[tuple(case[0]), case[1]] for case in inputs]
        elif number in {259, 401, 445}:
            result = [[tuple(tuple(x) for x in arg) for arg in case]
                      for case in inputs]
        elif number == 278:
            result = [[tuple(tuple(x) if isinstance(x, list) else x
                             for x in case[0])] for case in inputs]
        elif number == 307:
            result = [[tuple(case[0]), case[1], case[2]] for case in inputs]
        elif number == 722:
            result = [[{key: tuple(value) for key, value in case[0].items()}]
                      + case[1:] for case in inputs]
        elif number == 252:
            result = [[complex(case[0])] for case in inputs]
        elif number in {580, 615, 791}:
            def tuples(value):
                return tuple(tuples(x) for x in value) if isinstance(value, list) else value
            result = [tuples(case) for case in inputs]
        else:
            result = inputs
        if any(not isinstance(case, (list, tuple)) for case in result):
            raise TypeError("Each test case must be a positional-argument sequence")
        return [list(case) for case in result]
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise DataError(f"Cannot restore pinned MBPP input types for {task_id}: {exc}") from exc


def _read_locked(path: Path, record: dict) -> list[dict]:
    data = Path(path).read_bytes()
    if data.startswith(b"\x1f\x8b"):
        if _sha(data) != record["gzip_sha256"]:
            raise DataError(f"Gzip SHA-256 mismatch: {path}")
        if "gzip_bytes" in record and len(data) != record["gzip_bytes"]:
            raise DataError(f"Gzip byte count mismatch: {path}")
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError) as exc:
            raise DataError(f"Invalid gzip: {path}") from exc
    if _sha(data) != record["jsonl_sha256"]:
        raise DataError(f"JSONL SHA-256 mismatch: {path}")
    rows = [_json(line) for line in data.splitlines() if line.strip()]
    if len(rows) != record["task_count_verified"]:
        raise DataError(f"Task count mismatch for {record['name']}: {len(rows)}")
    if any(not isinstance(row, dict) for row in rows):
        raise DataError("Each JSONL record must be an object")
    return rows


def load_evalplus(humaneval_path: Path, mbpp_path: Path, lock_path: Path) -> list[dict]:
    """Load the exact locked data; verify both wrapper and payload when gzipped."""
    lock = _json(Path(lock_path).read_bytes())
    evaluator = lock.get("proposed_evaluator", {}).get("commit", EVALPLUS_COMMIT)
    if evaluator != EVALPLUS_COMMIT:
        raise DataError("Input restoration was audited for a different evaluator commit")
    records = {r["name"]: r for r in lock["datasets"]}
    if len(records) != len(lock["datasets"]):
        raise DataError("Duplicate dataset records in lock")
    tasks = []
    for name, path, source, prefix in [
        ("HumanEval+", humaneval_path, "humaneval", "HumanEval/"),
        ("MBPP+", mbpp_path, "mbpp", "Mbpp/"),
    ]:
        record = records.get(name)
        if record is None or record.get("version") != _VERSIONS[name]:
            raise DataError(f"Unsupported or missing locked dataset version: {name}")
        for field in ("gzip_sha256", "jsonl_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", record.get(field, "")):
                raise DataError(f"Missing or malformed {field} in {name} lock")
        for row in _read_locked(Path(path), record):
            required = {"task_id", "prompt", "entry_point", "canonical_solution",
                        "base_input", "plus_input", "atol"}
            if not required <= row.keys():
                raise DataError(f"Incomplete {name} task; missing {required - row.keys()}")
            task_id = row["task_id"]
            if not isinstance(task_id, str) or not re.fullmatch(re.escape(prefix) + r"\d+", task_id):
                raise DataError(f"Invalid task ID: {task_id!r}")
            if not all(isinstance(row[k], str) for k in ("prompt", "entry_point", "canonical_solution")):
                raise DataError(f"Non-text task fields: {task_id}")
            if not row["entry_point"].isidentifier():
                raise DataError(f"Invalid entry point: {task_id}")
            data_notes = []
            if source == "mbpp" and task_id == "Mbpp/793" and row["plus_input"] == {}:
                # This exact locked v0.2.0 record stores an empty mapping.
                # Upstream iterates it zero times; represent the empty suite
                # explicitly without dropping the task or its three base cases.
                row["plus_input"] = []
                data_notes.append("Pinned MBPP+ v0.2.0 task 793: empty plus_input object restored to an empty suite")
            if not all(isinstance(row[k], list) for k in ("base_input", "plus_input")):
                raise DataError(f"Non-list input collection: {task_id}")
            inputs = row["base_input"] + row["plus_input"]
            if source == "mbpp":
                inputs = _mbpp_inputs(task_id, inputs)
            elif any(not isinstance(case, list) for case in inputs):
                raise DataError(f"Malformed HumanEval positional inputs: {task_id}")
            if not inputs:
                raise DataError(f"Task has no test inputs: {task_id}")
            tolerance = row["atol"]
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
                raise DataError(f"Invalid atol: {task_id}")
            canonical_code = row["prompt"] + row["canonical_solution"]
            try:
                tree = _ast(canonical_code)
            except SyntaxError as exc:
                raise DataError(f"Unparseable reference code: {task_id}") from exc
            if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == row["entry_point"] for node in tree.body):
                raise DataError(f"Reference entry point is absent: {task_id}")
            tasks.append({"task_id": task_id, "source": source, "family_id": task_id,
                          "prompt": row["prompt"], "entry_point": row["entry_point"],
                          "canonical_code": canonical_code, "inputs": inputs,
                          "atol": tolerance,
                          "public_examples": extract_public_examples(row["prompt"], row["entry_point"]),
                          "base_input_count": len(row["base_input"]),
                          "plus_input_count": len(row["plus_input"]),
                          "data_notes": data_notes})
    return _families(tasks, None)


def _families(tasks: list[dict], overrides: dict | None) -> list[dict]:
    if not tasks:
        raise DataError("No tasks to partition")
    by_id = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in by_id:
            raise DataError(f"Missing or duplicate task ID: {task_id!r}")
        if task.get("source") not in {"humaneval", "mbpp", "synthetic"}:
            raise DataError(f"Unknown task source: {task_id}")
        if "family_id" in task and (not isinstance(task["family_id"], str) or not task["family_id"]):
            raise DataError(f"Invalid explicit family ID: {task_id}")
        by_id[task_id] = task
    parent = {task_id: task_id for task_id in by_id}

    def root(task_id):
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    def union(left, right):
        a, b = root(left), root(right)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for task_id in sorted(by_id):
        task = by_id[task_id]
        # Existing family declarations are preserved. Normalized exact matches
        # only merge; neither automated fingerprints nor manual maps split a family.
        fingerprints = [("family", str(task.get("family_id", task_id)))]
        if isinstance(task.get("prompt"), str):
            text = unicodedata.normalize("NFKC", task["prompt"])
            fingerprints.append(("prompt", "\n".join(line.rstrip() for line in text.strip().splitlines())))
        if task.get("canonical_code") and task.get("entry_point"):
            try:
                tree = _ast(task["canonical_code"])
                function = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and n.name == task["entry_point"])
                function = copy.deepcopy(function)
                entry = function.name
                class Normalize(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == entry:
                            node.id = "__entry__"
                        return node
                function.name = "__entry__"
                if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
                    function.body = function.body[1:]
                function = Normalize().visit(function)
                fingerprints.append(("function", ast.dump(function, include_attributes=False)))
            except (SyntaxError, StopIteration) as exc:
                raise DataError(f"Cannot fingerprint canonical function: {task_id}") from exc
        for fingerprint in fingerprints:
            if fingerprint in seen:
                union(task_id, seen[fingerprint])
            else:
                seen[fingerprint] = task_id
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise DataError("family_overrides must map task IDs to shared group labels")
        groups = {}
        for task_id, label in overrides.items():
            if task_id not in by_id or not isinstance(label, str) or not label:
                raise DataError(f"Invalid family override: {task_id!r}")
            if label in groups:
                union(task_id, groups[label])
            else:
                groups[label] = task_id
    return [dict(by_id[task_id], family_id="family:" + root(task_id))
            for task_id in sorted(by_id)]


def partition_tasks(tasks: list[dict], dev_count: int, panel_count: int,
                    panel_size: int, audit_min: int, seed: int,
                    family_overrides: dict | None = None) -> dict:
    """Deterministic exact-size, whole-family allocation with source stratification.

    Families with equal (size, source-count) are interchangeable for feasibility.
    Dynamic programming over those types and backtracking across bins avoids a
    greedy choice falsely declaring a feasible whole-family allocation impossible.
    """
    for key, value in {"dev_count": dev_count, "panel_count": panel_count,
                       "panel_size": panel_size, "audit_min": audit_min}.items():
        if type(value) is not int or value < 1:
            raise DataError(f"{key} must be a positive integer")
    if type(seed) is not int:
        raise DataError("seed must be an integer")
    tasks = _families(tasks, family_overrides)
    if dev_count + panel_count * panel_size + audit_min > len(tasks):
        raise DataError("Not enough tasks; revise preregistered sizes, never split families")
    bins, remainder = _allocate_family_bins(tasks, (panel_size,) * panel_count + (dev_count,), seed)
    result = {"dev": bins[-1], "panels": bins[:-1], "audit": remainder}
    _validate_partitions(result)
    return result


def _allocate_family_bins(tasks, targets, seed):
    """Jointly allocate exact whole-family bins, leaving all unused families intact.

    The pilot uses one additional exact-size audit bin. Joint allocation avoids
    drawing a subset that is itself impossible to partition despite another
    feasible subset existing in the parent development pool.
    """
    grouped = {}
    for task in tasks:
        grouped.setdefault(task["family_id"], []).append(task)
    sources = sorted({task["source"] for task in tasks})
    typed = {}
    for family, members in sorted(grouped.items()):
        kind = (len(members),) + tuple(sum(t["source"] == source for t in members)
                                      for source in sources)
        typed.setdefault(kind, []).append(family)
    kinds = sorted(typed, reverse=True)
    rng = random.Random(seed)
    for kind in kinds:
        rng.shuffle(typed[kind])
    start = tuple(len(typed[kind]) for kind in kinds)
    proportions = tuple(sum(t["source"] == source for t in tasks) / len(tasks)
                        for source in sources)

    @lru_cache(maxsize=None)
    def options(counts, target):
        @lru_cache(maxsize=None)
        def reachable(index, remaining):
            if remaining == 0:
                return True
            if index == len(kinds) or remaining < 0:
                return False
            size = kinds[index][0]
            return any(reachable(index + 1, remaining - take * size)
                       for take in range(min(counts[index], remaining // size) + 1))
        if not reachable(0, target):
            return ()
        choices = []
        def visit(index, remaining, allocation, totals):
            if index == len(kinds):
                if remaining == 0:
                    deviation = sum(abs(count - target * proportion)
                                    for count, proportion in zip(totals, proportions))
                    choices.append((deviation, tuple(allocation)))
                return
            size, *source_counts = kinds[index]
            for take in range(min(counts[index], remaining // size) + 1):
                if reachable(index + 1, remaining - take * size):
                    visit(index + 1, remaining - take * size,
                          allocation + [take],
                          tuple(count + take * extra for count, extra in zip(totals, source_counts)))
        visit(0, target, [], (0,) * len(sources))
        return tuple(allocation for _, allocation in sorted(choices))

    @lru_cache(maxsize=None)
    def solve(stage, counts):
        if stage == len(targets):
            return ()
        for allocation in options(counts, targets[stage]):
            remainder = tuple(n - used for n, used in zip(counts, allocation))
            answer = solve(stage + 1, remainder)
            if answer is not None:
                return (allocation,) + answer
        return None

    solution = solve(0, start)
    if solution is None:
        raise DataError("Exact whole-family partition is impossible; revise sizes or reviewed family mapping, never split a family")
    consumed = [0] * len(kinds)
    bins = []
    for allocation in solution:
        members = []
        for i, take in enumerate(allocation):
            for family in typed[kinds[i]][consumed[i]:consumed[i] + take]:
                members.extend(grouped[family])
            consumed[i] += take
        bins.append(sorted(members, key=lambda t: t["task_id"]))
    remainder = []
    for i, kind in enumerate(kinds):
        for family in typed[kind][consumed[i]:]:
            remainder.extend(grouped[family])
    return bins, sorted(remainder, key=lambda t: t["task_id"])


def _validate_partitions(partitions):
    if not isinstance(partitions, dict) or set(partitions) != {"dev", "panels", "audit"}:
        raise DataError("Partitions must contain exactly dev, panels, audit")
    if not isinstance(partitions["panels"], list):
        raise DataError("panels must be a list of task lists")
    ids, family_bins = set(), {}
    bins = [("dev", partitions["dev"]), ("audit", partitions["audit"])]
    bins += [(f"panel:{i}", panel) for i, panel in enumerate(partitions["panels"])]
    for name, tasks in bins:
        if not isinstance(tasks, list):
            raise DataError(f"{name} must be a task list")
        for task in tasks:
            task_id, family = task.get("task_id"), task.get("family_id")
            if not isinstance(task_id, str) or not isinstance(family, str) or task_id in ids:
                raise DataError("Missing family/task ID or repeated task across partitions")
            if family in family_bins and family_bins[family] != name:
                raise DataError(f"Family crosses partition boundary: {family}")
            if task.get("source") not in {"humaneval", "mbpp", "synthetic"}:
                raise DataError(f"Invalid source: {task_id}")
            ids.add(task_id)
            family_bins[family] = name


def _split_ids(partitions):
    return {"dev": [t["task_id"] for t in partitions["dev"]],
            "panels": [[t["task_id"] for t in p] for p in partitions["panels"]],
            "audit": [t["task_id"] for t in partitions["audit"]]}


def _counts(partitions):
    def counts(tasks):
        return {"tasks": len(tasks), "families": len({t["family_id"] for t in tasks}),
                "sources": {s: sum(t["source"] == s for t in tasks)
                            for s in sorted({t["source"] for t in tasks})}}
    return {"dev": counts(partitions["dev"]),
            "panels": [counts(panel) for panel in partitions["panels"]],
            "audit": counts(partitions["audit"])}


def _write_private(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def write_bundle(partitions: dict, output_dir: Path, metadata: dict) -> dict:
    """Create a new 0700 bundle; manifest contains counts/hashes, never task IDs."""
    from .wire import encode
    _validate_partitions(partitions)
    if not isinstance(metadata, dict):
        raise DataError("metadata must be a JSON object")
    metadata_bytes = _canonical(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output_dir, 0o700)
    salt = secrets.token_hex(32)
    payloads = {_FILES[name]: _canonical(encode(partitions[name])) for name in _FILES}
    payloads["metadata.private.json"] = metadata_bytes
    payloads["secret.private.json"] = _canonical({"salt": salt})
    manifest = {"schema": "rsi-audit-data-bundle-v1", "counts": _counts(partitions),
                "files": {name: _sha(content) for name, content in sorted(payloads.items())},
                "split_commitment_sha256": _sha(_canonical({"salt": salt, "splits": _split_ids(partitions)}))}
    for name, content in payloads.items():
        _write_private(output_dir / name, content)
    _write_private(output_dir / "public_manifest.json", _canonical(manifest))
    return manifest


def _manifest(bundle_dir):
    result = _json((Path(bundle_dir) / "public_manifest.json").read_bytes())
    if not isinstance(result, dict) or result.get("schema") != "rsi-audit-data-bundle-v1":
        raise DataError("Unsupported bundle schema")
    expected = set(_FILES.values()) | {"metadata.private.json", "secret.private.json"}
    if not isinstance(result.get("files"), dict) or set(result["files"]) != expected:
        raise DataError("Invalid bundle file manifest")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in result["files"].values()):
        raise DataError("Invalid bundle file hash")
    return result


def _read_bound(bundle_dir, name, manifest):
    path = Path(bundle_dir) / name
    if path.is_symlink():
        raise DataError(f"Bundle files must not be symlinks: {name}")
    content = path.read_bytes()
    if _sha(content) != manifest["files"][name]:
        raise DataError(f"Bundle file hash mismatch: {name}")
    return _json(content)


def read_partition(bundle_dir: Path, name: str):
    """Read exactly the named partition plus its public hash manifest.

    In particular, reading dev or panels never opens audit.json or the secret.
    Full-bundle verification is an explicit, separately authorized audit action.
    """
    from .wire import decode
    if name not in _FILES:
        raise DataError("Partition name must be dev, panels, or audit")
    manifest = _manifest(bundle_dir)
    return decode(_read_bound(bundle_dir, _FILES[name], manifest))


def verify_commitment(bundle_dir: Path) -> bool:
    """Verify all contents, including audit; call outside the search process."""
    from .wire import decode
    manifest = _manifest(bundle_dir)
    partitions = {name: decode(_read_bound(bundle_dir, filename, manifest))
                  for name, filename in _FILES.items()}
    _read_bound(bundle_dir, "metadata.private.json", manifest)
    secret = _read_bound(bundle_dir, "secret.private.json", manifest)
    if not isinstance(secret, dict) or not isinstance(secret.get("salt"), str) or not re.fullmatch(r"[0-9a-f]{64}", secret["salt"]):
        raise DataError("Invalid commitment salt")
    _validate_partitions(partitions)
    actual = _sha(_canonical({"salt": secret["salt"], "splits": _split_ids(partitions)}))
    if not secrets.compare_digest(actual, manifest.get("split_commitment_sha256", "")):
        raise DataError("Split commitment mismatch")
    if _counts(partitions) != manifest["counts"]:
        raise DataError("Manifest partition counts mismatch")
    return True


def pilot_bundle_from_dev(bundle_dir: Path, output_dir: Path, *, dev_count=16,
                          panel_count=2, panel_size=8, audit_min=8, seed=0,
                          audit_count: int | None = None,
                          metadata: dict | None = None) -> dict:
    """Build an isolated rehearsal using only the parent bundle's dev tasks.

    Specify audit_count=8 for the exact 40-task study pilot. Omitting it keeps
    the legacy behavior of assigning every unused parent-dev task to pilot audit.
    No parent confirmation, audit, secret, or metadata file is read here.
    """
    development = read_partition(bundle_dir, "dev")
    if audit_count is None:
        partitions = partition_tasks(development, dev_count, panel_count, panel_size,
                                     audit_min, seed)
    else:
        for name, value in {"dev_count": dev_count, "panel_count": panel_count,
                            "panel_size": panel_size, "audit_min": audit_min,
                            "audit_count": audit_count}.items():
            if type(value) is not int or value < 1:
                raise DataError(f"{name} must be a positive integer")
        if type(seed) is not int:
            raise DataError("seed must be an integer")
        if audit_count < audit_min:
            raise DataError("audit_count must be at least audit_min")
        targets = (panel_size,) * panel_count + (dev_count, audit_count)
        development = _families(development, None)
        if sum(targets) > len(development):
            raise DataError("Not enough parent-dev tasks for exact pilot sizes")
        bins, _unused = _allocate_family_bins(development, targets, seed)
        partitions = {"dev": bins[-2], "panels": bins[:-2], "audit": bins[-1]}
        _validate_partitions(partitions)
    private_metadata = dict(metadata or {})
    private_metadata.update({"purpose": "pilot-only-from-parent-dev", "seed": seed,
                             "parent_dev_tasks": len(development),
                             "pilot_selected_tasks": sum(len(partitions[name]) for name in ("dev", "audit"))
                                 + sum(len(panel) for panel in partitions["panels"]),
                             "parent_manifest_sha256": _sha((Path(bundle_dir) / "public_manifest.json").read_bytes())})
    return write_bundle(partitions, output_dir, private_metadata)
