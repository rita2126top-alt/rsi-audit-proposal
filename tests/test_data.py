import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_audit.data import (
    DataError, EVALPLUS_COMMIT, _mbpp_inputs, extract_public_examples,
    load_evalplus, partition_tasks, pilot_bundle_from_dev, read_partition,
    verify_commitment, write_bundle,
)


def task(index, source="mbpp", family=None):
    return {"task_id": f"{source}/{index}", "source": source,
            "family_id": family or f"unique/{index}",
            "prompt": f"Distinct specification {index}", "entry_point": "f",
            "inputs": [[(index, index + 1), {index}, complex(index, 1)]],
            "atol": 0, "public_examples": []}


def fake_assets(directory):
    he = [{"task_id": "HumanEval/0", "prompt": 'def f(x):\n    """>>> f(1)\n    2\n    """\n',
           "entry_point": "f", "canonical_solution": "    return x + 1\n",
           "base_input": [[99]], "plus_input": [[100]], "atol": 0}]
    mb = [{"task_id": "Mbpp/2", "prompt": '"""Find intersection.\nassert g((1, 2), (2, 3)) == (2,)\n"""\n',
           "entry_point": "g", "canonical_solution": "def g(a, b):\n    return tuple(set(a) & set(b))\n",
           "base_input": [[[1, 2], [2, 3]]], "plus_input": [[[9], [9]]], "atol": 0}]
    records, paths = [], []
    for name, version, rows in [("HumanEval+", "v0.1.10", he), ("MBPP+", "v0.2.0", mb)]:
        plain = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        zipped = gzip.compress(plain, mtime=0)
        path = directory / (name + ".jsonl.gz")
        path.write_bytes(zipped)
        paths.append(path)
        records.append({"name": name, "version": version, "task_count_verified": len(rows),
                        "gzip_bytes": len(zipped), "gzip_sha256": hashlib.sha256(zipped).hexdigest(),
                        "jsonl_sha256": hashlib.sha256(plain).hexdigest()})
    lock = directory / "lock.json"
    lock.write_text(json.dumps({"proposed_evaluator": {"commit": EVALPLUS_COMMIT}, "datasets": records}))
    return paths[0], paths[1], lock


class LoadingTests(unittest.TestCase):
    def test_locked_gzip_and_decompressed_payload_have_same_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            he, mb, lock = fake_assets(Path(tmp))
            loaded = load_evalplus(he, mb, lock)
            plain = Path(tmp) / "human.jsonl"
            plain.write_bytes(gzip.decompress(he.read_bytes()))
            self.assertEqual(loaded, load_evalplus(plain, mb, lock))
            human = next(t for t in loaded if t["source"] == "humaneval")
            self.assertEqual(human["public_examples"], [{"args": [1], "kwargs": {}, "expected": 2}])
            self.assertEqual(human["inputs"], [[99], [100]])
            mbpp = next(t for t in loaded if t["source"] == "mbpp")
            self.assertIsInstance(mbpp["inputs"][0][0], tuple)
            self.assertIn("def g", mbpp["canonical_code"])

    def test_hash_and_count_tampering_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            he, mb, lock = fake_assets(Path(tmp))
            original = he.read_bytes()
            he.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(DataError, "Gzip SHA"):
                load_evalplus(he, mb, lock)
            he.write_bytes(original)
            data = json.loads(lock.read_text())
            data["datasets"][0]["task_count_verified"] = 2
            lock.write_text(json.dumps(data))
            with self.assertRaisesRegex(DataError, "Task count"):
                load_evalplus(he, mb, lock)

    def test_special_mbpp_input_types_follow_upstream(self):
        self.assertEqual(_mbpp_inputs("Mbpp/124", [["0", "1j"]]), [[0.0, 1j]])
        self.assertEqual(_mbpp_inputs("Mbpp/252", [["2+3j"]]), [[2+3j]])
        self.assertEqual(_mbpp_inputs("Mbpp/722", [[{"a": [1, 2]}, 3, 4]]), [[{"a": (1, 2)}, 3, 4]])
        self.assertEqual(_mbpp_inputs("Mbpp/115", [[[[1, 2], []]]]), [[[{1, 2}, {}]]])
        self.assertEqual(_mbpp_inputs("Mbpp/580", [[[[1, 2], [3, 4]]]]), [[((1, 2), (3, 4))]])
        with self.assertRaises(DataError):
            _mbpp_inputs("Mbpp/124", [["not-float", "not-complex"]])

    def test_public_extraction_never_executes_or_reinterprets_assertions(self):
        prompt = "\n".join([
            "assert f(2, k=3) == [5]", "assert set(f(3)) == {3}",
            "assert math.isclose(f(4), 4.0, rel_tol=0.01)",
            "assert f(__import__('os').system('false')) == 0",
            ">>> f((1, 2))", "{1, 2}",
        ])
        examples = extract_public_examples(prompt, "f")
        self.assertEqual(len(examples), 2)
        self.assertIn({"args": [2], "kwargs": {"k": 3}, "expected": [5]}, examples)
        self.assertIn({"args": [(1, 2)], "kwargs": {}, "expected": {1, 2}}, examples)


class PartitionTests(unittest.TestCase):
    def test_synthetic_demo_is_explicitly_identified(self):
        tasks = [task(i, "synthetic") for i in range(40)]
        result = partition_tasks(tasks, 12, 3, 4, 16, 4)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_bundle(result, Path(tmp) / "demo", {"synthetic": True})
            self.assertEqual(manifest["counts"]["audit"]["sources"], {"synthetic": 16})
            self.assertNotIn("humaneval", manifest["counts"]["dev"]["sources"])

    def test_deterministic_stratified_and_family_disjoint(self):
        tasks = [task(i, "humaneval" if i < 12 else "mbpp") for i in range(40)]
        mapping = {"humaneval/0": "reviewed-pair", "mbpp/20": "reviewed-pair"}
        one = partition_tasks(tasks, 10, 2, 5, 20, 7, mapping)
        two = partition_tasks(list(reversed(tasks)), 10, 2, 5, 20, 7, mapping)
        self.assertEqual(one, two)
        bins = [one["dev"], *one["panels"], one["audit"]]
        places = {}
        for i, items in enumerate(bins):
            for item in items:
                places.setdefault(item["family_id"], set()).add(i)
        self.assertTrue(all(len(value) == 1 for value in places.values()))
        self.assertEqual([len(items) for items in bins], [10, 5, 5, 20])

    def test_backtracks_instead_of_greedy_false_impossibility(self):
        # A tempting 2+1 fill of the 3-task panel strands two 3-task families
        # and cannot fill dev=4. The valid solution uses a 3-task family first.
        tasks, index = [], 0
        for label, size, source in [("a", 3, "humaneval"), ("b", 3, "humaneval"),
                                    ("c", 2, "mbpp"), ("d", 1, "mbpp"), ("e", 20, "mbpp")]:
            for _ in range(size):
                tasks.append(task(index, source, label))
                index += 1
        result = partition_tasks(tasks, 4, 1, 3, 22, 9)
        self.assertEqual(len(result["dev"]), 4)
        self.assertEqual(len(result["panels"][0]), 3)
        self.assertEqual(len(result["audit"]), 22)

    def test_exact_duplicate_reference_functions_are_one_family(self):
        tasks = [task(i) for i in range(8)]
        for index, name in [(0, "f"), (1, "g")]:
            tasks[index].update(entry_point=name, canonical_code=f"def {name}(x):\n    return x + 1\n")
        result = partition_tasks(tasks, 2, 1, 2, 4, 4)
        bins = [result["dev"], *result["panels"], result["audit"]]
        positions = {t["task_id"]: i for i, group in enumerate(bins) for t in group}
        self.assertEqual(positions["mbpp/0"], positions["mbpp/1"])

    def test_impossible_whole_family_and_duplicate_ids_fail(self):
        tasks = [task(i, family=f"pair/{i//2}") for i in range(10)]
        with self.assertRaisesRegex(DataError, "whole-family"):
            partition_tasks(tasks, 3, 1, 3, 4, 0)
        with self.assertRaisesRegex(DataError, "duplicate task"):
            partition_tasks(tasks + [tasks[0]], 2, 1, 2, 2, 0)
        with self.assertRaisesRegex(DataError, "Invalid family override"):
            partition_tasks(tasks, 2, 1, 2, 2, 0, {"missing": "group"})


class BundleTests(unittest.TestCase):
    def make_bundle(self, directory):
        tasks = [task(i, "humaneval" if i < 8 else "mbpp") for i in range(24)]
        partitions = partition_tasks(tasks, 12, 2, 3, 6, 1)
        manifest = write_bundle(partitions, directory, {"private_note": "AUDIT_NEVER_PUBLIC", "seed": 1})
        return partitions, manifest

    def test_round_trip_modes_private_metadata_and_salted_commitment(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bundle"
            parts, manifest = self.make_bundle(directory)
            self.assertTrue(verify_commitment(directory))
            self.assertEqual(read_partition(directory, "dev"), parts["dev"])
            self.assertEqual(read_partition(directory, "audit"), parts["audit"])
            self.assertEqual(os.stat(directory).st_mode & 0o777, 0o700)
            for path in directory.iterdir():
                self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            public = json.dumps(manifest)
            self.assertNotIn("AUDIT_NEVER_PUBLIC", public)
            for item in parts["audit"]:
                self.assertNotIn(item["task_id"], public)
            other = Path(tmp) / "other"
            second = write_bundle(parts, other, {})
            self.assertNotEqual(manifest["split_commitment_sha256"], second["split_commitment_sha256"])
            with self.assertRaises(FileExistsError):
                write_bundle(parts, directory, {})

    def test_hash_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bundle"
            self.make_bundle(directory)
            (directory / "dev.json").write_text("[]")
            with self.assertRaisesRegex(DataError, "hash mismatch"):
                read_partition(directory, "dev")
            with self.assertRaisesRegex(DataError, "hash mismatch"):
                verify_commitment(directory)

    def test_changed_salt_fails_even_when_file_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bundle"
            self.make_bundle(directory)
            secret_path = directory / "secret.private.json"
            secret = json.loads(secret_path.read_text())
            secret["salt"] = ("0" if secret["salt"][0] != "0" else "1") + secret["salt"][1:]
            secret_path.write_text(json.dumps(secret))
            manifest_path = directory / "public_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["secret.private.json"] = hashlib.sha256(secret_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(DataError, "commitment mismatch"):
                verify_commitment(directory)

    def test_dev_reader_and_pilot_do_not_open_parent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bundle"
            parts, _ = self.make_bundle(directory)
            original = Path.read_bytes
            def guarded(path):
                if path.parent == directory and path.name in {"audit.json", "panels.json", "secret.private.json"}:
                    raise AssertionError("Pilot or dev reader accessed a forbidden parent partition")
                return original(path)
            pilot = Path(tmp) / "pilot"
            with patch.object(Path, "read_bytes", guarded):
                self.assertEqual(read_partition(directory, "dev"), parts["dev"])
                pilot_bundle_from_dev(directory, pilot, dev_count=4, panel_count=1,
                                      panel_size=2, audit_min=6, seed=3)
            self.assertTrue(verify_commitment(pilot))
            parent_dev = {t["task_id"] for t in parts["dev"]}
            pilot_tasks = read_partition(pilot, "dev") + read_partition(pilot, "audit")
            pilot_tasks += [t for panel in read_partition(pilot, "panels") for t in panel]
            self.assertEqual({t["task_id"] for t in pilot_tasks}, parent_dev)

    def test_exact_forty_task_pilot_uses_only_whole_parent_dev_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "bundle"
            tasks = [task(i, "humaneval" if i < 48 else "mbpp", f"pair/{i//2}")
                     for i in range(160)]
            parent = partition_tasks(tasks, 128, 2, 8, 16, 5)
            write_bundle(parent, directory, {"oracle_precomputed": True})
            original = Path.read_bytes
            def guarded(path):
                if path.parent == directory and path.name not in {"dev.json", "public_manifest.json"}:
                    raise AssertionError("Exact pilot opened a parent file outside dev and manifest")
                return original(path)
            pilot = Path(tmp) / "pilot"
            with patch.object(Path, "read_bytes", guarded):
                manifest = pilot_bundle_from_dev(directory, pilot, audit_count=8, seed=17)
            self.assertEqual(manifest["counts"]["dev"]["tasks"], 16)
            self.assertEqual([panel["tasks"] for panel in manifest["counts"]["panels"]], [8, 8])
            self.assertEqual(manifest["counts"]["audit"]["tasks"], 8)
            self.assertTrue(verify_commitment(pilot))
            bins = [read_partition(pilot, "dev"), *read_partition(pilot, "panels"),
                    read_partition(pilot, "audit")]
            selected = {row["task_id"] for group in bins for row in group}
            self.assertEqual(len(selected), 40)
            self.assertTrue(selected < {row["task_id"] for row in parent["dev"]})
            family_places = {}
            for index, group in enumerate(bins):
                for row in group:
                    family_places.setdefault(row["family_id"], set()).add(index)
            self.assertTrue(all(len(places) == 1 for places in family_places.values()))
            excluded_families = {row["family_id"] for row in parent["dev"] if row["task_id"] not in selected}
            self.assertFalse(excluded_families & family_places.keys())
            with self.assertRaisesRegex(DataError, "whole-family"):
                pilot_bundle_from_dev(directory, Path(tmp) / "impossible", audit_count=3, audit_min=1)


if __name__ == "__main__":
    unittest.main()
