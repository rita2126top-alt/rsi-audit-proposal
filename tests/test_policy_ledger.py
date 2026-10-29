"""Declarative-policy boundaries and persistent ledger integrity checks."""

import dataclasses
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from rsi_audit.common import IntegrityError, atomic_json, digest, read_json
from rsi_audit.ledger import Ledger
from rsi_audit.policy import Policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.valid = {"strategy_prompt": "Solve carefully", "feedback_style": "compact",
                      "repair_order": "syntax_then_examples"}
        self.count_tokens = lambda text: len(text.split())

    def test_valid_policy_and_token_limit(self):
        policy = Policy.from_dict(self.valid, self.count_tokens, max_tokens=2)
        self.assertEqual(policy.strategy_prompt, "Solve carefully")
        with self.assertRaisesRegex(ValueError, "budget"):
            Policy.from_dict(self.valid, self.count_tokens, max_tokens=1)

    def test_extra_fields_cannot_add_code_network_or_resource_access(self):
        for key, value in (("plugin", "import os"), ("network", True), ("max_calls", 100),
                           ("score_patch", "all pass"), ("test_path", "/audit")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Policy.from_dict(dict(self.valid, **{key: value}), self.count_tokens)

    def test_missing_fields_are_rejected(self):
        for key in self.valid:
            obj = dict(self.valid)
            del obj[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                Policy.from_dict(obj, self.count_tokens)

    def test_nonstring_enum_values_are_invalid_proposals_not_type_errors(self):
        for key in ("feedback_style", "repair_order"):
            for value in ([], {}, None, True, 3):
                obj = dict(self.valid, **{key: value})
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    Policy.from_dict(obj, self.count_tokens)

    def test_unknown_enum_strings_are_rejected(self):
        for key in ("feedback_style", "repair_order"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Policy.from_dict(dict(self.valid, **{key: "unregistered"}), self.count_tokens)

    def test_prompt_is_nonempty_bounded_text(self):
        for value in ("", "   ", None, [], "x" * 32769):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                Policy.from_dict(dict(self.valid, strategy_prompt=value), self.count_tokens)

    def test_duplicate_json_keys_are_rejected(self):
        text = '{"strategy_prompt":"safe","strategy_prompt":"replacement","feedback_style":"compact","repair_order":"syntax_then_examples"}'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            Policy.from_response(text, self.count_tokens)

    def test_fenced_json_is_parsed_but_prose_is_not(self):
        result = Policy.from_response("```json\n" + json.dumps(self.valid) + "\n```", self.count_tokens)
        self.assertEqual(result.to_dict(), self.valid)
        with self.assertRaises(ValueError):
            Policy.from_response("Here is the JSON: " + json.dumps(self.valid), self.count_tokens)

    def test_policy_identity_is_order_stable_and_state_is_immutable(self):
        first = Policy.from_dict(self.valid, self.count_tokens)
        second = Policy.from_dict(dict(reversed(list(self.valid.items()))), self.count_tokens)
        self.assertEqual(first.sha256, second.sha256)
        changed = Policy.from_dict(dict(self.valid, feedback_style="detailed"), self.count_tokens)
        self.assertNotEqual(first.sha256, changed.sha256)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.strategy_prompt = "replacement"


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.ledger = Ledger(self.path, create=True)
        self.ledger.append({"kind": "solve", "passed": False, "output_tokens": 10})
        self.ledger.append({"kind": "gate", "accepted": False})

    def test_round_trip_and_private_key_permissions(self):
        loaded = Ledger(self.path)
        self.assertEqual(loaded.head, self.ledger.head)
        self.assertEqual(loaded.events("gate"), [{"kind": "gate", "accepted": False}])
        self.assertEqual(stat.S_IMODE((self.path / "ledger.key").stat().st_mode), 0o600)
        self.assertEqual(len((self.path / "ledger.key").read_bytes()), 32)

    def test_changed_event_cannot_keep_original_authentication(self):
        rows = [json.loads(line) for line in (self.path / "ledger.jsonl").read_text().splitlines()]
        rows[0]["event"]["passed"] = True
        (self.path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(IntegrityError, "authentication"):
            Ledger(self.path)

    def test_recomputed_unkeyed_hash_cannot_forge_hmac(self):
        rows = [json.loads(line) for line in (self.path / "ledger.jsonl").read_text().splitlines()]
        rows[0]["event"]["passed"] = True
        rows[0]["hash"] = digest({key: rows[0][key] for key in ("sequence", "previous", "event")})
        (self.path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(IntegrityError, "authentication"):
            Ledger(self.path)

    def test_partial_record_truncation_is_rejected(self):
        path = self.path / "ledger.jsonl"
        path.write_bytes(path.read_bytes()[:-9])
        with self.assertRaisesRegex(IntegrityError, "Incomplete"):
            Ledger(self.path)

    def test_whole_record_truncation_requires_saved_head_and_count(self):
        frozen_count, frozen_head = len(self.ledger.records), self.ledger.head
        path = self.path / "ledger.jsonl"
        path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
        # A valid chain prefix is intrinsically indistinguishable from an
        # earlier complete chain. The external frozen anchor detects rollback.
        loaded = Ledger(self.path)
        with self.assertRaisesRegex(IntegrityError, "prefix"):
            loaded.assert_prefix(frozen_count, frozen_head)

    def test_reordered_records_are_rejected(self):
        path = self.path / "ledger.jsonl"
        path.write_bytes(b"".join(reversed(path.read_bytes().splitlines(keepends=True))))
        with self.assertRaisesRegex(IntegrityError, "ordering"):
            Ledger(self.path)

    def test_record_from_other_key_cannot_be_spliced_in(self):
        other = self.path / "other"
        other_ledger = Ledger(other, create=True)
        other_ledger.append({"kind": "solve", "passed": False, "output_tokens": 10})
        path = self.path / "ledger.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[0] = (other / "ledger.jsonl").read_bytes()
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(IntegrityError, "authentication"):
            Ledger(self.path)

    def test_frozen_prefix_remains_valid_when_new_authenticated_events_are_appended(self):
        count, head = len(self.ledger.records), self.ledger.head
        self.ledger.append({"kind": "phase_time", "wall_seconds": 0.1, "gpu_count": 0})
        loaded = Ledger(self.path)
        loaded.assert_prefix(count, head)
        with self.assertRaises(IntegrityError):
            loaded.assert_prefix(count, "0" * 64)

    def test_empty_frozen_prefix_uses_zero_hash(self):
        self.ledger.assert_prefix(0, "0" * 64)
        with self.assertRaises(IntegrityError):
            self.ledger.assert_prefix(0, self.ledger.head)

    def test_symlink_ledger_or_key_is_rejected(self):
        for name in ("ledger.jsonl", "ledger.key"):
            with self.subTest(name=name):
                target = self.path / name
                backup = self.path / (name + ".original")
                target.rename(backup)
                target.symlink_to(backup)
                try:
                    with self.assertRaisesRegex(IntegrityError, "symlinks"):
                        Ledger(self.path)
                finally:
                    target.unlink()
                    backup.rename(target)


class CommonPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_atomic_exclusive_write_cannot_overwrite_a_frozen_file(self):
        path = self.path / "freeze.json"
        atomic_json(path, {"frozen": True}, exclusive=True)
        with self.assertRaises(FileExistsError):
            atomic_json(path, {"frozen": False}, exclusive=True)
        self.assertEqual(read_json(path), {"frozen": True})

    def test_atomic_json_refuses_a_symlink_destination(self):
        target = self.path / "target.json"
        target.write_text('{"keep": true}')
        alias = self.path / "alias.json"
        alias.symlink_to(target)
        with self.assertRaises(IntegrityError):
            atomic_json(alias, {"keep": False})
        self.assertEqual(read_json(target), {"keep": True})

    def test_json_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        path = self.path / "bad.json"
        for text in ('{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(IntegrityError):
                    read_json(path)


if __name__ == "__main__":
    unittest.main()
